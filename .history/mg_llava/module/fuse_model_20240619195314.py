# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.ops import roi_align

# from bbox_llava.module.box_llava_model import prepare_inputs_labels_for_multimodal
from . import BoxLLaVAModel
from mmengine.model import BaseModel
from xtuner.model.utils import get_peft_model_state_dict, guess_load_checkpoint
from xtuner.model.modules import ProjectorConfig, ProjectorModel
from fairscale.nn.checkpoint import checkpoint_wrapper
from transformers import PreTrainedModel
from typing import List, Optional
from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from xtuner.tools.utils import get_stop_criteria
from transformers import  GenerationConfig
VIDEO_TOKEN_INDEX = -201


class MultiFuseObjectLLaVAModel(BoxLLaVAModel):

    def __init__(
        self,
        *args,
        pretrained_pth=None,
        projector_depth=2,
        visual_encoder_aux=None,
        frames=None,
        box_feat_size=None,
        checkpoint_fuse=False,
        fuse_model=None,
        evaluate_max_tokens=100,
        **kwargs,
    ):
        super().__init__(*args, projector_depth=2, **kwargs)
        self.evaluate_max_tokens = evaluate_max_tokens
        bbox_projector_config = ProjectorConfig(
            visual_hidden_size=2880,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth,
        )

        if visual_encoder_aux is not None:
            self.visual_encoder_aux = self._build_from_cfg_or_module(visual_encoder_aux)
            self.visual_encoder_aux.requires_grad_(False)
            self.aux_last_only = False
            if visual_encoder_aux.get('use_multi_level', False):
                self.multi_level_linear = torch.nn.ModuleList(
                    [nn.Linear(192, 1024), nn.Linear(384, 1024), nn.Linear(768, 1024), nn.Linear(1536, 1024)]
                )
                bbox_projector_config = ProjectorConfig(
                    visual_hidden_size=1024,
                    llm_hidden_size=self.llm.config.hidden_size,
                    depth=projector_depth,
                )
            if visual_encoder_aux.get('last_only', False):
                self.aux_last_only = True
                bbox_projector_config = ProjectorConfig(
                    visual_hidden_size=1536,
                    llm_hidden_size=self.llm.config.hidden_size,
                    depth=projector_depth,
                )

        else:
            self.visual_encoder_aux = None

        self.bbox_projector = ProjectorModel(bbox_projector_config).to(self.visual_encoder.dtype)
        self.frames = frames
        self.box_feat_size = box_feat_size
        if not fuse_model:
            self.fuse_module = DualPathFuseModule(
                low_res_dim=self.visual_encoder.config.hidden_size, high_res_dim=1536,
            ).to(self.visual_encoder.dtype)
            if checkpoint_fuse:
                self.fuse_module = checkpoint_wrapper(self.fuse_module)
        elif fuse_model == 1:
            projector_config = ProjectorConfig(
                visual_hidden_size=2048, llm_hidden_size=self.llm.config.hidden_size, depth=projector_depth
            )
            self.projector = ProjectorModel(projector_config).to(self.visual_encoder.dtype)
            self.fuse_module = DualPathFuseModule1(
                low_res_dim=self.visual_encoder.config.hidden_size,
                high_res_dim=1536,
            ).to(self.visual_encoder.dtype)
        elif fuse_model == 2:
            projector_config = ProjectorConfig(
                visual_hidden_size=2048, llm_hidden_size=self.llm.config.hidden_size, depth=projector_depth
            )
            self.projector = ProjectorModel(projector_config).to(self.visual_encoder.dtype)
            self.fuse_module = DualPathFuseModule2(
                low_res_dim=self.visual_encoder.config.hidden_size,
                high_res_dim=1536,
            ).to(self.visual_encoder.dtype)
        elif fuse_model == 3:
            projector_config = ProjectorConfig(
                visual_hidden_size=1024+1536, llm_hidden_size=self.llm.config.hidden_size, depth=projector_depth
            )
            self.projector = ProjectorModel(projector_config).to(self.visual_encoder.dtype)
            self.fuse_module = DualPathFuseModule3(
                low_res_dim=self.visual_encoder.config.hidden_size,
                high_res_dim=1536,
            )

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')
        # if sampler_query_num:
        #     self.use_sampler = True
        #     self.resampler = Resampler(
        #         grid_size=sampler_query_num,
        #         embed_dim=5120,  # 保持与视觉模型输出的 embed_dim 一致
        #         num_heads=1024 // 128,  # 保持与视觉模型输出的 num_heads 一致
        #         kv_dim=1024,  # 保持与视觉模型输出的 kv_dim 一致
        #     )

    def state_dict(self, *args, **kwargs):
        state_dict = BaseModel.state_dict(self, *args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(get_peft_model_state_dict(self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({k: v for k, v in state_dict.items() if 'visual_encoder.' in k})
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({k: v for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update({k: v for k, v in state_dict.items() if 'projector.' in k})

        # Step 4. bbox_projector
        to_return.update({k: v for k, v in state_dict.items() if 'bbox_projector.' in k})

        # Step 5. multi_fuse_layer
        to_return.update({k: v for k, v in state_dict.items() if 'fuse_module.' in k})

        # Step 6. multi_level_linear
        if hasattr(self, 'multi_level_linear'):
            print('muilti linear true!')
            to_return.update({k: v for k, v in state_dict.items() if 'multi_level_linear.' in k})

        # print(state_dict.keys())
        return to_return

    def _prepare_data_for_llm(self, data):
        if 'pixel_values' in data:
            visual_outputs = self.visual_encoder(
                data['pixel_values'].to(self.visual_encoder.dtype), output_hidden_states=True
            )
            if type(self.visual_encoder).__name__ == 'CLIPVisionModel':
                visual_outputs = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
            elif type(self.visual_encoder).__name__ == 'SiglipVisionModel':
                visual_outputs = visual_outputs.hidden_states[self.visual_select_layer]
            else:
                raise NotImplementedError
            if data['pixel_values_aux'][0].dim() == 4:
                is_video = True
                num_frames = data['pixel_values_aux'][0].shape[1]
            else:
                is_video = False
            if self.visual_encoder_aux is not None:
                pixels_aux = []
                for pixels in data['pixel_values_aux']:
                    if pixels.dim() == 3:
                        pixels = pixels.unsqueeze(0)
                    elif pixels.dim() == 4:
                        pixels = pixels.permute(1, 0, 2, 3)
                    pixels_aux.append(pixels)
                visual_outputs_aux = torch.cat(pixels_aux, dim=0)
                aux_output = self.visual_encoder_aux(visual_outputs_aux.to(self.visual_encoder_aux.dtype))
                visual_outputs_aux = aux_output['image_features']
                multi_level = aux_output['multi_level']
                last_feat = aux_output['last_feat']
                visual_outputs_aux.to(device=visual_outputs.device)
                visual_outputs_aux = visual_outputs_aux.float()

            fuse_features = self.fuse_module(low_res_feat=visual_outputs, high_res_feat=last_feat)
            pixel_values = self.projector(fuse_features)
            if is_video:

                b_f, n, c = pixel_values.shape
                data['pixel_values'] = pixel_values.view(-1, num_frames, n, c)
                b_f, c, h, w = visual_outputs_aux.shape
                visual_outputs_aux = visual_outputs_aux.view(-1, num_frames, c, h, w)
            else:
                data['pixel_values'] = pixel_values

            # 基于 bbox + roialign 提取 visual_outputs 特征
            # TODO 先每张图片单独处理，后面考虑并行
            bbox_visual_outputs = []
            for i, (boxes, labels) in enumerate(zip(data['gt_boxes'], data['gt_labels'])):
                # 1,c,h,w -> n,c,7,7
                if is_video:
                    out_box_feat = []
                    for j, (boxes_frame, labels_frame) in enumerate(zip(boxes, labels)):
                        out_box_feat_frame = self.align_box(
                            j, visual_outputs_aux[i], boxes_frame, pixel_values, multi_level
                        )
                        out_box_feat.append(out_box_feat_frame)
                else:
                    out_box_feat = self.align_box(i, visual_outputs_aux, boxes, pixel_values, multi_level)
                bbox_visual_outputs.append(out_box_feat)
            # b,n,c
            data['bbox_feats'] = bbox_visual_outputs
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        return data

    def align_box(self, i, visual_outputs_aux, boxes, pixel_values, multi_level=False):
        aux_size = visual_outputs_aux.shape[2]
        if self.box_feat_size:
            box_feat_size = self.box_feat_size
        else:
            box_feat_size = 56 if not self.aux_last_only else 7
        if multi_level:
            channels = 0
            multi_level_feat = visual_outputs_aux.new_zeros(
                len(multi_level), boxes.shape[0], 1024, box_feat_size, box_feat_size
            )
            for level, channels_loc in enumerate(multi_level):
                level_feature = visual_outputs_aux[i : i + 1, channels : channels + channels_loc, :, :]
                channels = channels + channels_loc
                out_box_feat = roi_align(
                    level_feature, [boxes], output_size=box_feat_size, spatial_scale=aux_size / 768
                )
                linear_loc = self.multi_level_linear[level].to(
                    device=out_box_feat.device, dtype=out_box_feat.dtype
                )
                out_box_feat = linear_loc(out_box_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                multi_level_feat[level] = out_box_feat

            out_box_feat = multi_level_feat.sum(0)

            out_box_feat = out_box_feat.to(pixel_values.dtype)
            # 通过 avg pool 变成维度为 1 的序列 -> n,c -> 1,n,c
            out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(
                1, out_box_feat.shape[0], out_box_feat.shape[1]
            )
            # 1，n，c -> n，c'
            out_box_feat = self.bbox_projector(out_box_feat)[0]
        else:
            try:
                out_box_feat = roi_align(
                    visual_outputs_aux[i : i + 1],
                    [boxes.float()],
                    output_size=box_feat_size,
                    spatial_scale=aux_size / 768,
                )
            except:
                print(visual_outputs_aux[i : i + 1].to(boxes.dtype))
                print(boxes.dtype)
                raise
            out_box_feat = out_box_feat.to(pixel_values.dtype)
            # 通过 avg pool 变成维度为 1 的序列 -> n,c -> 1,n,c
            out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(
                1, out_box_feat.shape[0], out_box_feat.shape[1]
            )
            # 1，n，c -> n，c'
            out_box_feat = self.bbox_projector(out_box_feat)[0]
            # if multi_level:
            #     channels = 0
            #     multi_level_feat = visual_outputs_aux.new_zeros(
            #         len(multi_level),visual_outputs_aux.shape[0], 1024, 192, 192
            #     )
            #     for level, channels_loc in enumerate(multi_level):
            #         level_feature = visual_outputs_aux[:, channels : channels + channels_loc, :, :]
            #         channels = channels + channels_loc
            #         linear_loc = self.multi_level_linear[level].to(
            #             device=level_feature.device, dtype=level_feature.dtype
            #         )
            #         out_feat = linear_loc(level_feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            #         multi_level_feat[level] = out_feat

            #     out_feat_sum = multi_level_feat.sum(0)

            #     visual_outputs_aux = out_feat_sum.float()

            # bbox_visual_outputs = []
            # for i, (boxes, labels) in enumerate(zip(data['gt_boxes'], data['gt_labels'])):
            #     # 1,c,h,w -> n,c,7,7
            #     out_box_feat = roi_align(
            #         visual_outputs_aux[i : i + 1], [boxes], output_size=56, spatial_scale=192 / 768
            #     )
            #     out_box_feat = out_box_feat.to(pixel_values.dtype)
            #     # 通过 avg pool 变成维度为 1 的序列 -> n,c -> 1,n,c
            #     out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(
            #         1, out_box_feat.shape[0], out_box_feat.shape[1]
            #     )
            #     # 1，n，c -> n，c'
            #     out_box_feat = self.bbox_projector(out_box_feat)[0]
            #     bbox_visual_outputs.append(out_box_feat)
        return out_box_feat

    def preparing_for_generation(self, metainfo: dict = None):
        default_generation_kwargs = dict(
            max_new_tokens=self.evaluate_max_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)

        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

def prepare_inputs_labels_for_multimodal(
    llm: PreTrainedModel,
    input_ids: torch.LongTensor = None,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    bbox_feats=None,
    gt_labels=None,
    **kwargs
):
    if pixel_values is None:
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': None,
            'labels': labels,
        }

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [
        cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]

    new_inputs_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        num_videos = (cur_input_ids == VIDEO_TOKEN_INDEX).sum()
        num = num_images + num_videos
        if num == 0:
            cur_pixel_values = pixel_values[cur_image_idx]
            cur_inputs_embeds_1 = llm.get_input_embeddings()(cur_input_ids)
            cur_inputs_embeds = torch.cat([cur_inputs_embeds_1, cur_pixel_values[0:0]], dim=0)
            new_inputs_embeds.append(cur_inputs_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1

            # =============================================
            # 在 image embedding 后面加入 bbox embedding
            cur_bbox_feats = bbox_feats[batch_idx]
            new_inputs_embeds.append(cur_bbox_feats[0:0])
            new_labels.append(
                torch.full((1,), IGNORE_INDEX, device=cur_bbox_feats.device, dtype=labels[batch_idx].dtype)
            )
            continue
        if num_images > 0:
            token_indices = (
                [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
        elif num_videos > 0:
            token_indices = (
                [-1] + torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[token_indices[i] + 1 : token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[token_indices[i] + 1 : token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_inputs_embeds = llm.get_input_embeddings()(torch.cat(cur_input_ids_noim))
        cur_inputs_embeds_no_im = torch.split(cur_inputs_embeds, split_sizes, dim=0)
        cur_new_inputs_embeds = []
        cur_new_labels = []

        if num_images > 0:
            for i in range(num_images + 1):
                cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_pixel_values = pixel_values[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_inputs_embeds.append(cur_pixel_values)
                    cur_new_labels.append(
                        torch.full(
                            (cur_pixel_values.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

                    # =============================================
                    # 在 image embedding 后面加入 bbox embedding
                    cur_bbox_feats = bbox_feats[batch_idx]  # n,c
                    cur_new_inputs_embeds.append(cur_bbox_feats)
                    cur_new_labels.append(
                        torch.full(
                            (cur_bbox_feats.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
        elif num_videos > 0:
            for i in range(num_videos + 1):
                cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_videos:
                    cur_pixel_values = pixel_values[cur_image_idx]
                    cur_image_idx += 1
                    for j in range(cur_pixel_values.shape[0]):
                        cur_new_inputs_embeds.append(cur_pixel_values[j])
                        cur_new_labels.append(
                            torch.full(
                                (cur_pixel_values[j].shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )

                        # =============================================
                        # 在 image embedding 后面加入 bbox embedding
                        cur_bbox_feats = bbox_feats[batch_idx][j]  # n,c
                        cur_new_inputs_embeds.append(cur_bbox_feats)
                        cur_new_labels.append(
                            torch.full(
                                (cur_bbox_feats.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )

        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_inputs_embeds.append(cur_new_inputs_embeds)
        new_labels.append(cur_new_labels)

    # Combine them
    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)

    new_inputs_embeds_padded = []
    new_labels_padded = torch.full(
        (batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device
    )
    attention_mask = torch.zeros(
        (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
    )
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_inputs_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        new_inputs_embeds_padded.append(
            torch.cat(
                (
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_len, cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device,
                    ),
                ),
                dim=0,
            )
        )
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(
                0, cur_len, dtype=position_ids.dtype, device=position_ids.device
            )

    new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return {
        'input_ids': None,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'past_key_values': past_key_values,
        'inputs_embeds': new_inputs_embeds,
        'labels': new_labels,
    }


class DualPathFuseModule(nn.Module):
    # change channel+gate+sum
    def __init__(self, low_res_dim, high_res_dim, zero_init=True):
        super().__init__()

        self.slow_conv = nn.Conv2d(high_res_dim, high_res_dim, 1)
        self.slow_proj = nn.Conv2d(high_res_dim, low_res_dim, 1)

        self.fast_conv = nn.Conv2d(low_res_dim, low_res_dim, 7, padding=3, groups=low_res_dim)
        self.fast_proj = nn.Conv2d(low_res_dim, low_res_dim, 1)

        self.gate = nn.Sequential(
            nn.Linear(low_res_dim * 2, low_res_dim // 2),
            nn.GELU(),
            nn.Linear(low_res_dim // 2, 1),
        )

        nn.init.xavier_uniform_(self.slow_conv.weight)
        nn.init.xavier_uniform_(self.fast_conv.weight)
        nn.init.zeros_(self.slow_conv.bias)
        nn.init.zeros_(self.fast_conv.bias)
        if zero_init:
            nn.init.zeros_(self.slow_proj.weight)
            nn.init.zeros_(self.fast_proj.weight)
        else:
            nn.init.xavier_uniform_(self.slow_proj.weight)
            nn.init.xavier_uniform_(self.fast_proj.weight)
        nn.init.zeros_(self.slow_proj.bias)
        nn.init.zeros_(self.fast_proj.bias)

    def forward(self, low_res_feat, high_res_feat, sampler=None):
        b, c, h, w = high_res_feat.shape
        _, _, d = low_res_feat.shape
        high_res_feat = self.slow_proj(F.gelu(self.slow_conv(high_res_feat)))
        high_res_feat = high_res_feat.view(b, d, -1).transpose(1, 2)
        dst_size = int(math.sqrt(low_res_feat.shape[1]))
        low_res_feat = low_res_feat.transpose(1, 2).view(b, d, dst_size, dst_size)
        low_res_feat = low_res_feat + self.fast_proj(F.gelu(self.fast_conv(low_res_feat)))
        low_res_feat = low_res_feat.view(b, d, dst_size * dst_size).transpose(1, 2)
        gate = self.gate(torch.cat([low_res_feat, high_res_feat], -1).mean(1)).unsqueeze(1)
        if not sampler:
            low_res_feat = low_res_feat + high_res_feat * gate.tanh()
        return low_res_feat


class DualPathFuseModule1(nn.Module):
    # change channel+gate+concat
    def __init__(self, low_res_dim, high_res_dim, zero_init=True):
        super().__init__()

        self.slow_conv = nn.Conv2d(high_res_dim, high_res_dim, 1)
        self.slow_proj = nn.Conv2d(high_res_dim, low_res_dim, 1)

        self.fast_conv = nn.Conv2d(low_res_dim, low_res_dim, 7, padding=3, groups=low_res_dim)
        self.fast_proj = nn.Conv2d(low_res_dim, low_res_dim, 1)

        self.gate = nn.Sequential(
            nn.Linear(low_res_dim * 2, low_res_dim // 2),
            nn.GELU(),
            nn.Linear(low_res_dim // 2, 1),
        )

        nn.init.xavier_uniform_(self.slow_conv.weight)
        nn.init.xavier_uniform_(self.fast_conv.weight)
        nn.init.zeros_(self.slow_conv.bias)
        nn.init.zeros_(self.fast_conv.bias)
        if zero_init:
            nn.init.zeros_(self.slow_proj.weight)
            nn.init.zeros_(self.fast_proj.weight)
        else:
            nn.init.xavier_uniform_(self.slow_proj.weight)
            nn.init.xavier_uniform_(self.fast_proj.weight)
        nn.init.zeros_(self.slow_proj.bias)
        nn.init.zeros_(self.fast_proj.bias)

    def forward(self, low_res_feat, high_res_feat):
        b, c, h, w = high_res_feat.shape
        _, _, d = low_res_feat.shape
        high_res_feat = self.slow_proj(F.gelu(self.slow_conv(high_res_feat)))
        high_res_feat = high_res_feat.view(b, d, -1).transpose(1, 2)
        dst_size = int(math.sqrt(low_res_feat.shape[1]))
        low_res_feat = low_res_feat.transpose(1, 2).view(b, d, dst_size, dst_size)
        low_res_feat = low_res_feat + self.fast_proj(F.gelu(self.fast_conv(low_res_feat)))
        low_res_feat = low_res_feat.view(b, d, dst_size * dst_size).transpose(1, 2)
        gate = self.gate(torch.cat([low_res_feat, high_res_feat], -1).mean(1)).unsqueeze(1)
        low_res_feat = torch.cat((low_res_feat, high_res_feat * gate.tanh()), dim=-1)
        return low_res_feat

class DualPathFuseModule2(nn.Module):
    # change channel+concat
    def __init__(self, low_res_dim, high_res_dim, zero_init=True):
        super().__init__()

        self.slow_conv = nn.Conv2d(high_res_dim, high_res_dim, 1)
        self.slow_proj = nn.Conv2d(high_res_dim, low_res_dim, 1)

        self.fast_conv = nn.Conv2d(low_res_dim, low_res_dim, 7, padding=3, groups=low_res_dim)
        self.fast_proj = nn.Conv2d(low_res_dim, low_res_dim, 1)

        nn.init.xavier_uniform_(self.slow_conv.weight)
        nn.init.xavier_uniform_(self.fast_conv.weight)
        nn.init.zeros_(self.slow_conv.bias)
        nn.init.zeros_(self.fast_conv.bias)
        if zero_init:
            nn.init.zeros_(self.slow_proj.weight)
            nn.init.zeros_(self.fast_proj.weight)
        else:
            nn.init.xavier_uniform_(self.slow_proj.weight)
            nn.init.xavier_uniform_(self.fast_proj.weight)
        nn.init.zeros_(self.slow_proj.bias)
        nn.init.zeros_(self.fast_proj.bias)

    def forward(self, low_res_feat, high_res_feat):
        b, c, h, w = high_res_feat.shape
        _, _, d = low_res_feat.shape
        high_res_feat = self.slow_proj(F.gelu(self.slow_conv(high_res_feat)))
        high_res_feat = high_res_feat.view(b, d, -1).transpose(1, 2)
        dst_size = int(math.sqrt(low_res_feat.shape[1]))
        low_res_feat = low_res_feat.transpose(1, 2).view(b, d, dst_size, dst_size)
        low_res_feat = low_res_feat + self.fast_proj(F.gelu(self.fast_conv(low_res_feat)))
        low_res_feat = low_res_feat.view(b, d, dst_size * dst_size).transpose(1, 2)
        low_res_feat = torch.cat((low_res_feat, high_res_feat), dim=-1)
        return low_res_feat


class DualPathFuseModule3(nn.Module):
    # concat
    def __init__(self, low_res_dim, high_res_dim, zero_init=True):
        super().__init__()

    def forward(self, low_res_feat, high_res_feat):
        b, c, h, w = high_res_feat.shape
        _, _, d = low_res_feat.shape
        high_res_feat = high_res_feat.view(b, c, -1).transpose(1, 2)
        low_res_feat = torch.cat((low_res_feat, high_res_feat), dim=-1)
        return low_res_feat


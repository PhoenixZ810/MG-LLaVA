# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import roi_align

# from bbox_llava.module.box_llava_model import prepare_inputs_labels_for_multimodal
from .box_model import BoxLLaVAModel
from mmengine.model import BaseModel
from xtuner.model.utils import get_peft_model_state_dict, guess_load_checkpoint
from xtuner.model.modules import ProjectorConfig, ProjectorModel

from transformers import PreTrainedModel
from typing import List, Optional
from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX

VIDEO_TOKEN_INDEX = -201


class BoxObjectLLaVAModel(BoxLLaVAModel):
    def __init__(
        self,
        *args,
        pretrained_pth=None,
        projector_depth=2,
        visual_encoder_aux=None,
        frames=None,
        box_feat_size=None,
        **kwargs,
    ):
        super().__init__(*args, projector_depth=2, **kwargs)
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

        self.bbox_projector = None
        self.frames = frames
        self.box_feat_size = box_feat_size

        self.box_fuse_module = BoxFuseModule(
            low_res_dim=self.visual_encoder.config.hidden_size,
            high_res_dim=2880,
        ).to(self.visual_encoder.dtype)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

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

        # Step 6. box_fuse_layer
        to_return.update({k: v for k, v in state_dict.items() if 'box_fuse_module.' in k})

        # Step 5. multi_level_linear
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
            pixel_values = self.projector(visual_outputs)
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
                visual_outputs_aux.to(device=pixel_values.device)
                visual_outputs_aux = visual_outputs_aux.float()

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
            # data['bbox_feats'] = bbox_visual_outputs

            fuse_box_feat = self.box_fuse_module(bbox_visual_outputs, visual_outputs)
            pixel_values = self.projector(fuse_box_feat)

            if is_video:
                b_f, n, c = pixel_values.shape
                # data['pixel_values'] = pixel_values.view(-1, num_frames, n, c)
                b_f, c, h, w = visual_outputs_aux.shape
                visual_outputs_aux = visual_outputs_aux.view(-1, num_frames, c, h, w)
            else:
                data['pixel_values'] = pixel_values

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

        else:
            out_box_feat = roi_align(
                visual_outputs_aux[i : i + 1],
                [boxes],
                output_size=box_feat_size,
                spatial_scale=aux_size / 768,
            )
        out_box_feat = out_box_feat.to(pixel_values.dtype)
        # 通过 avg pool 变成维度为 1 的序列 -> n,c -> 1,n,c
        out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(1, out_box_feat.shape[0], out_box_feat.shape[1])
        # 1，n，c -> n，c'
        # out_box_feat = self.bbox_projector(out_box_feat)[0]
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
    **kwargs,
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

            # # =============================================
            # # 在 image embedding 后面加入 bbox embedding
            # cur_bbox_feats = bbox_feats[batch_idx]
            # new_inputs_embeds.append(cur_bbox_feats[0:0])
            # new_labels.append(
            #     torch.full((1,), IGNORE_INDEX, device=cur_bbox_feats.device, dtype=labels[batch_idx].dtype)
            # )
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

                    # # =============================================
                    # # 在 image embedding 后面加入 bbox embedding
                    # cur_bbox_feats = bbox_feats[batch_idx]  # n,c
                    # cur_new_inputs_embeds.append(cur_bbox_feats)
                    # cur_new_labels.append(
                    #     torch.full(
                    #         (cur_bbox_feats.shape[0],),
                    #         IGNORE_INDEX,
                    #         device=cur_labels.device,
                    #         dtype=cur_labels.dtype,
                    #     )
                    # )
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

                        # # =============================================
                        # # 在 image embedding 后面加入 bbox embedding
                        # cur_bbox_feats = bbox_feats[batch_idx][j]  # n,c
                        # cur_new_inputs_embeds.append(cur_bbox_feats)
                        # cur_new_labels.append(
                        #     torch.full(
                        #         (cur_bbox_feats.shape[0],),
                        #         IGNORE_INDEX,
                        #         device=cur_labels.device,
                        #         dtype=cur_labels.dtype,
                        #     )
                        # )

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


class BoxFuseModule(nn.Module):
    # change channel+gate+sum
    def __init__(self, low_res_dim, high_res_dim, zero_init=True):
        super().__init__()

        self.vlm_uni_query_projector = nn.Sequential(
            nn.LayerNorm(low_res_dim),
            nn.Linear(low_res_dim, low_res_dim),
        )
        self.vlm_uni_key_projector = nn.Sequential(
            nn.LayerNorm(high_res_dim),
            nn.Linear(high_res_dim, low_res_dim),
        )
        self.vlm_uni_val_projector = nn.Sequential(
            nn.LayerNorm(high_res_dim),
            nn.Linear(high_res_dim, low_res_dim),
        )

    def forward(self, box_tensors, vit_feat):
        max_length = max(len(t[0]) for t in box_tensors)
        box_feat_padded = torch.zeros(
            (len(box_tensors), max_length, box_tensors[0].shape[-1]),
            dtype=vit_feat.dtype,
            device=vit_feat.device,
        )
        mask = torch.zeros((len(box_tensors), max_length), dtype=torch.bool, device=vit_feat.device)
        for i, tensor in enumerate(box_tensors):
            length = tensor.size(1)
            box_feat_padded[i, :length] = tensor
            mask[i, :length] = True

        # token attention
        embed_query = self.vlm_uni_query_projector(vit_feat)  # bs,576,dm
        embed_key = self.vlm_uni_key_projector(box_feat_padded)  # bs,576,4,dm
        embed_value = self.vlm_uni_val_projector(box_feat_padded)  # bs,576,4,dm
        embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
        embed_att = embed_att.masked_fill(mask.unsqueeze(1) == False, float('-inf'))
        embed_att = embed_att.nan_to_num()
        embed_feat = embed_att.softmax(-1) @ embed_value  # bs,576,dm
        box_feat_padded = vit_feat + embed_feat
        # box_feat_list = []
        # for i in range(len(box_feat_padded)):
        #     box_feat_list.append(box_feat_padded[i : i + 1, : box_tensors[i].size(1)])
        return box_feat_padded

from xtuner.model import LLaVAModel
from collections import OrderedDict
from xtuner.model.utils import (get_peft_model_state_dict, prepare_inputs_labels_for_multimodal)
from torchvision.ops import roi_align
import torch
from xtuner.model.modules import ProjectorConfig, ProjectorModel
from transformers import PreTrainedModel
from typing import List, Optional
from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX


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
        **kwargs):
    if pixel_values is None:
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': None,
            'labels': labels
        }

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [
        cur_labels[cur_attention_mask]
        for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]

    new_inputs_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_pixel_values = pixel_values[cur_image_idx]
            cur_inputs_embeds_1 = llm.get_input_embeddings()(cur_input_ids)
            cur_inputs_embeds = torch.cat(
                [cur_inputs_embeds_1, cur_pixel_values[0:0]], dim=0)
            new_inputs_embeds.append(cur_inputs_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1

            # =============================================
            # 在 image embedding 后面加入 bbox embedding
            cur_bbox_feats = bbox_feats[batch_idx]
            new_inputs_embeds.append(cur_bbox_feats[0:0])
            new_labels.append(
                torch.full((1,),
                           IGNORE_INDEX,
                           device=cur_bbox_feats.device,
                           dtype=labels[batch_idx].dtype))
            continue

        image_token_indices = [-1] + torch.where(
            cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                                  cur_input_ids.shape[0]
                              ]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] +
                                                    1:image_token_indices[i +
                                                                          1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] +
                                              1:image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_inputs_embeds = llm.get_input_embeddings()(
            torch.cat(cur_input_ids_noim))
        cur_inputs_embeds_no_im = torch.split(
            cur_inputs_embeds, split_sizes, dim=0)
        cur_new_inputs_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_pixel_values = pixel_values[cur_image_idx]
                cur_image_idx += 1
                cur_new_inputs_embeds.append(cur_pixel_values)
                cur_new_labels.append(
                    torch.full((cur_pixel_values.shape[0],),
                               IGNORE_INDEX,
                               device=cur_labels.device,
                               dtype=cur_labels.dtype))

                # =============================================
                # 在 image embedding 后面加入 bbox embedding
                cur_bbox_feats = bbox_feats[batch_idx]  # n,c
                cur_new_inputs_embeds.append(cur_bbox_feats)
                cur_new_labels.append(
                    torch.full((cur_bbox_feats.shape[0],),
                               IGNORE_INDEX,
                               device=cur_labels.device,
                               dtype=cur_labels.dtype))

        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_inputs_embeds.append(cur_new_inputs_embeds)
        new_labels.append(cur_new_labels)

    # Combine them
    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)

    new_inputs_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len),
                                   IGNORE_INDEX,
                                   dtype=new_labels[0].dtype,
                                   device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len),
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len),
                               dtype=position_ids.dtype,
                               device=position_ids.device)

    for i, (cur_new_embed,
            cur_new_labels) in enumerate(zip(new_inputs_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        new_inputs_embeds_padded.append(
            torch.cat((cur_new_embed,
                       torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                   dtype=cur_new_embed.dtype,
                                   device=cur_new_embed.device)),
                      dim=0))
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(
                0,
                cur_len,
                dtype=position_ids.dtype,
                device=position_ids.device)

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
        'labels': new_labels
    }


class BoxLLaVAModel(LLaVAModel):
    def __init__(self, *args, projector_depth=2, **kwargs):
        super().__init__(*args, projector_depth=2, **kwargs)

        bbox_projector_config = ProjectorConfig(
            visual_hidden_size=self.visual_encoder.config.hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth)
        self.bbox_projector = ProjectorModel(bbox_projector_config).to(
            self.visual_encoder.dtype)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})

        # Step 4. bbox_projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'bbox_projector.' in k})
        return to_return

    def _prepare_data_for_llm(self, data):
        if 'pixel_values' in data:
            visual_outputs = self.visual_encoder(
                data['pixel_values'].to(self.visual_encoder.dtype),
                output_hidden_states=True)
            if type(self.visual_encoder).__name__ == 'CLIPVisionModel':
                visual_outputs = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
            elif type(self.visual_encoder).__name__ == 'SiglipVisionModel':
                visual_outputs = visual_outputs.hidden_states[self.visual_select_layer]
            else:
                raise NotImplementedError
            pixel_values = self.projector(visual_outputs)
            data['pixel_values'] = pixel_values

            # 基于 bbox + roialign 提取 visual_outputs 特征
            visual_outputs = visual_outputs.reshape(visual_outputs.shape[0], 27, 27, -1)
            visual_outputs = visual_outputs.permute((0, 3, 1, 2)).float()
            # TODO 先每张图片单独处理，后面考虑并行
            bbox_visual_outputs = []
            for i, (boxes, labels) in enumerate(zip(data['gt_boxes'], data['gt_labels'])):
                # 1,c,h,w -> n,c,7,7
                out_box_feat = roi_align(visual_outputs[i:i + 1], [boxes], output_size=7, spatial_scale=27 / 384)
                out_box_feat = out_box_feat.to(pixel_values.dtype)
                # 通过 avg pool 变成维度为 1 的序列 -> n,c -> 1,n,c
                out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(1, out_box_feat.shape[0], out_box_feat.shape[1])
                # 1，n，c -> n，c'
                out_box_feat = self.bbox_projector(out_box_feat)[0]
                bbox_visual_outputs.append(out_box_feat)

            # b,n,c
            data['bbox_feats'] = bbox_visual_outputs
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        return data

import math
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import roi_align

# from bbox_llava.module.box_llava_model import prepare_inputs_labels_for_multimodal
from bbox_llava.module import BoxLLaVAModel
from xtuner.model import LLaVAModel
from mmengine.model import BaseModel
from xtuner.model.utils import get_peft_model_state_dict
from xtuner.model.modules import ProjectorConfig, ProjectorModel

from transformers import PreTrainedModel
from typing import List, Optional

from object_llava.module.object_llava_model import ObjectLLaVAModel, prepare_inputs_labels_for_multimodal


class FilterObjectLLaVAModel(ObjectLLaVAModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                visual_outputs_aux, multi_level = self.visual_encoder_aux(
                    visual_outputs_aux.to(self.visual_encoder_aux.dtype),
                )

                visual_outputs_aux.to(device=pixel_values.device)
                visual_outputs_aux = visual_outputs_aux.float()

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
            box_after_filter = self.visual_encoder_aux.class_forward(
                bbox_visual_outputs.to(self.visual_encoder_aux.dtype), data['categories']
            )
            data['bbox_feats'] = bbox_visual_outputs
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        return data

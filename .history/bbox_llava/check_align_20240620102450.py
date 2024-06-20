import os

import numpy as np
import torch
import cv2
os.environ['HF_MODULES_CACHE'] = '../'

from transformers import AutoTokenizer, CLIPImageProcessor, SiglipImageProcessor

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset import ConcatDataset
from bbox_llava.module import BoxLLaVADataset, box_collate_fn
from mmengine.visualization import Visualizer
from xtuner.dataset.samplers import LengthGroupedSampler
from mmengine.runner.runner import Runner

data_root = 'data/llava_data/'
data_path = data_root + 'LLaVA-Pretrain/blip_laion_cc_sbu_558k.json'
offline_processed_text_folder = 'data/text_cache/llava_pretrain_all'
image_folder = data_root + 'LLaVA-Pretrain/images'

prompt_template = PROMPT_TEMPLATE.vicuna
# image token 占 (336 / 14) ** 2, region token 占 1
max_length = int(2048 - (336 / 14) ** 2 - 1)

llm_name_or_path = '/mnt/petrelfs/share_data/huanghaian/model/phi-2'
visual_encoder_name_or_path = '/mnt/hwfile/llmeval/opencompass/checkpoints/llm/hf_hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/'
# visual_encoder_name_or_path = '/mnt/petrelfs/share_data/huanghaian/model/siglip-so400m-patch14-384/'
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    # added_tokens_decoder=ADD_TOKENS_DECODER,
    trust_remote_code=True,
    padding_side='right',
)

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True,
)

train_dataset = dict(
    type=BoxLLaVADataset,
    offline_processed_text_folder=offline_processed_text_folder,
    data_path=data_path,
    box_json_path='data/blip_laion_cc_sbu_558k_only_bbox_debug.json',
    iou_threshold=0.5,
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
)

class_type = train_dataset.pop('type')
obj_llava_dataset = class_type(**train_dataset)
print(len(obj_llava_dataset))

image_mean = ([0.48145466, 0.4578275, 0.40821073],)
image_std = [0.26862954, 0.26130258, 0.27577711]
mean = torch.tensor(image_mean).view(3, 1, 1)
std = torch.tensor(image_std).view(3, 1, 1)

vis = Visualizer()


def _get_adaptive_scales(areas: np.ndarray, min_area: int = 800, max_area: int = 30000) -> np.ndarray:
    scales = 0.5 + (areas - min_area) // (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


debug_dataset = True
save_path = 'debug/check_align'

if debug_dataset:
    for data in obj_llava_dataset:
        if 'gt_boxes' in data:
            pixel_values = data['pixel_values']
            pixel_values = pixel_values * std + mean
            pixel_values = pixel_values * 255
            pixel_values = torch.permute(pixel_values, (1, 2, 0))

            vis.set_image(pixel_values.numpy())

            conversation = data['conversation'][0]['input']
            print(conversation)
            print(data['conversation'][0]['output'])

            bboxes = data['gt_boxes']
            labels = data['gt_labels']

            image2colors = []
            for _ in range(len(bboxes)):
                colors = np.random.random((1, 3)) * 0.7 + 0.3
                colors = (colors * 255).astype(int).tolist()[0]
                image2colors.append(tuple(colors))

            bboxes = np.array(bboxes).reshape(-1, 4)
            positions = bboxes[:, :2] + 3
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)
            vis.draw_bboxes(bboxes, edge_colors=image2colors, line_widths=3)
            vis.draw_texts(
                labels,
                positions,
                colors='g',
                font_sizes=[int(13 * s) for s in scales],
                bboxes=[{'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}] * len(scales),
            )

            if 'mask' in data:
                mask = data['mask']
                for i, m in enumerate(mask):
                    vis.draw_polygons(m.reshape(-1, 2), edge_colors='w', face_colors=image2colors[i])

            # vis.show()
            drawn_img = vis.get_image()
            ids = data['id']
            cv2.imwrite(os.path.join(save_path, f'{ids}.jpg'), drawn_img[..., ::-1])

else:
    train_dataloader = dict(
        batch_size=4,
        num_workers=0,
        dataset=obj_llava_dataset,
        sampler=dict(type=LengthGroupedSampler, length_property='length', per_device_batch_size=4 * 1),
        collate_fn=dict(type=box_collate_fn),
    )

    train_dataloader = Runner.build_dataloader(train_dataloader)
    for i, load in enumerate(train_dataloader):
        print(load)
        break

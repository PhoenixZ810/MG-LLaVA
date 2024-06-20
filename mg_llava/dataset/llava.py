# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset_video
from xtuner.dataset.huggingface import process_hf_dataset
from xtuner.dataset.utils import expand2square


class LLaVADataset(Dataset):

    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 include_video=False):
        super().__init__()

        assert offline_processed_text_folder or (data_path and tokenizer)
        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            self.text_data = load_from_disk(offline_processed_text_folder)
        else:
            json_data = json.load(open(data_path))

            image_count = 0
            video_count = 0
            pure_conv = 0
            for item in json_data:
                if 'image' in item.keys() and item['image']:
                    image_count += 1
                    if 'video' not in item.keys():
                        item['video'] = ''

                elif 'video' in item.keys() and item['video']:
                    video_count += 1
                    if 'image' not in item.keys():
                        item['image'] = ''
                else:
                    if 'model' not in item.keys():
                        print('no image and video in this data!')
                    else:
                        pure_conv += 1
                        item['video'] = ''
                        item['image'] = ''
            print(f'initial-image {image_count}, video {video_count}, pure_conversation {pure_conv}')

            for idx in range(len(json_data)):
                if isinstance(json_data[idx]['id'], int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])
            json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
            if include_video:
                self.text_data = process_hf_dataset_video(
                    dataset=json_data,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    dataset_map_fn=dataset_map_fn,
                    template_map_fn=template_map_fn,
                    split='train',
                    max_dataset_length=max_dataset_length,
                    remove_unused_columns=False,
                    pack_to_max_length=False,
                    with_image_token=True)
            else:
                self.text_data = process_hf_dataset(
                    dataset=json_data,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    dataset_map_fn=dataset_map_fn,
                    template_map_fn=template_map_fn,
                    split='train',
                    max_dataset_length=max_dataset_length,
                    remove_unused_columns=False,
                    pack_to_max_length=False,
                    with_image_token=True)

        self.image_folder = image_folder
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values'] = image
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
        return data_dict

# from xtuner.dataset.llava import LLaVADataset
from object_llava.dataset.llava import LLaVADataset
from torchvision.ops import batched_nms
import json
import os
import copy
import torch
from PIL import Image
from xtuner.dataset.utils import expand2square
import torchvision.transforms.functional as F
from bbox_llava.module.utils import bbox_nms
from mmengine.logging import print_log
from collections import defaultdict
from ..dataset.utils import (
    load_and_transform_video,
    get_video_transform,
    adjust_short_resize_coordinates,
    adjust_center_crop_box,
    adjust_padding_boxes,
)

class ObjectLLaVADataset(LLaVADataset):

    def __init__(
        self,
        *args,
        max_length=2048,
        box_json_path,
        iou_threshold=0.2,
        image_size_aux=768,
        limit_mode='per_category',
        limit_num=100,
        video_folder=None,
        num_frames=8,
        video_pad_image_to_square=False,
        **kwargs
    ):
        super().__init__(*args, max_length=max_length, **kwargs)
        self.json_data = json.load(open(box_json_path))
        if isinstance(self.json_data, list):
            self.json_data = {str(item['id']): item for item in self.json_data}
        self.iou_threshold = iou_threshold
        self.image_size_aux = image_size_aux
        self.limit_mode = limit_mode
        self.limit_num = limit_num
        self.max_length = max_length
        self.is_clip = False
        self.video_folder = video_folder
        self.num_frames = num_frames
        if hasattr(self.image_processor, 'crop_size'):
            self.crop_size_raw = self.image_processor.crop_size.copy()
            self.image_processor.crop_size['height'] = image_size_aux
            self.image_processor.crop_size['width'] = image_size_aux
            self.image_processor.size['shortest_edge'] = image_size_aux
            self.is_clip = True
        else:
            self.crop_size_raw = self.image_processor.size.copy()
            self.image_processor.size['height'] = image_size_aux
            self.image_processor.size['width'] = image_size_aux
        self.video_pad_image_to_square = video_pad_image_to_square

    def __getitem__(self, idx):
        # print_log(idx)
        data_dict = copy.deepcopy(self.text_data[idx])
        if data_dict.get('image', None):
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            old_w, old_h = F.get_image_size(image)

            box_data = self.json_data[data_dict['id']]
            boxes = box_data['boxes']
            labels = box_data['labels']
            scores = box_data['scores']

            if self.pad_image_to_square:
                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                # print(boxes)
                boxes = adjust_padding_boxes(boxes, height=old_h, width=old_w)
                old_w, old_h = F.get_image_size(image)

            image_aux = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_h, new_w = image_aux.shape[-2:]
            data_dict['pixel_values_aux'] = image_aux

            image = image_aux.clone()
            image = torch.nn.functional.interpolate(
                image[None], size=[self.crop_size_raw['height'], self.crop_size_raw['width']], mode='bilinear', align_corners=False
            )[0]
            data_dict['pixel_values'] = image

            # nms
            boxes, labels, scores = bbox_nms(boxes, labels, scores, self.iou_threshold)
            len_text_ids = len(data_dict['input_ids'])
            close_to_max_length = len_text_ids >= self.max_length - 50
            if len_text_ids >= self.max_length:
                data_dict['input_ids'] = data_dict['input_ids'][:self.max_length]
                data_dict['labels'] = data_dict['labels'][:self.max_length]
                data_dict['length'] = self.max_length
            if boxes.shape[0]!=0:
                if self.limit_num is not None or close_to_max_length:
                    if self.limit_mode == 'per_category':
                        if self.limit_num is not None and self.limit_num>50:
                            num_per_category = 20
                        elif self.limit_num is not None and self.limit_num<50:
                            num_per_category = 5
                        if close_to_max_length:
                            num_per_category = 1
                        if len(scores) > self.limit_num or close_to_max_length:
                            object_text = []
                            label_indices = defaultdict(list)
                            for index, label in enumerate(labels):
                                label_indices[label].append(index)
                            label_indices = dict(label_indices)

                            for item in label_indices:
                                if len(label_indices[item]) > num_per_category:
                                    item_scores = scores[label_indices[item]]
                                    # item_boxes = boxes[label_indices[item]]
                                    top_scores, top_indices = item_scores.topk(num_per_category, largest=True)
                                    top_boxes_index = [label_indices[item][i] for i in top_indices.tolist()]
                                    label_indices[item] = top_boxes_index
                                    # top_boxes = item_boxes[top_indices]
                                    # label_indices[item] = top_boxes
                            boxes = [boxes[i] for item in label_indices for i in label_indices[item]]
                            try:
                                boxes = torch.stack(boxes)
                            except:
                                print(data_dict)
                                print(num_per_category)
                                print(boxes)
                                print(label_indices)
                                print(labels)
                                raise
                            labels = [labels[i] for item in label_indices for i in label_indices[item]]
                            # if len(boxes) > self.limit_num:
                            #     print_log('Warning: too many boxes in one image after filter, remain %d boxes' % self.limit_num)

                    elif self.limit_mode == 'per_image':
                        num_per_image = self.limit_num
                        if close_to_max_length:
                            num_per_image = 20
                        if len(scores) > num_per_image:
                            top_scores, top_indices = scores.topk(num_per_image, largest=True)
                            boxes = boxes[top_indices]
                            labels = [labels[i] for i in top_indices.tolist()]

                    else:
                        raise

            # TODO: 如果有裁剪或者 padding 操作，则本代码有问题
            # 坐标是原图尺度的，要映射到resize后的尺度
            if self.is_clip:
                boxes, h1, w1 = adjust_short_resize_coordinates(boxes, old_h, old_w, self.image_size_aux)
                boxes, labels = adjust_center_crop_box(boxes, labels, h1, w1, self.image_size_aux)
            else:
                boxes[:, 0::2] = boxes[:, 0::2] / old_w * new_w
                boxes[:, 1::2] = boxes[:, 1::2] / old_h * new_h
            # if self.is_clip:

            data_dict['gt_boxes'] = boxes
            data_dict['gt_labels'] = labels
            data_dict['id'] = data_dict['id']
            data_dict['modal'] = 'image'

        elif data_dict.get('video', None):
            try:
                video_file = data_dict['video'].replace('mkv', 'mp4')
                # print('video:', video_file)
                video_decode_backend = 'decord'

                video, video_aux, old_h, old_w = load_and_transform_video(
                    os.path.join(self.video_folder, video_file),
                    get_video_transform(
                        video_decode_backend=video_decode_backend,
                        num_frames=self.num_frames,
                        resolution=self.crop_size_raw['height'],
                        padding=self.video_pad_image_to_square,
                    ),
                    transform_aux=get_video_transform(
                        video_decode_backend=video_decode_backend,
                        num_frames=self.num_frames,
                        resolution=self.image_size_aux,
                        padding=self.video_pad_image_to_square,
                    ),
                    video_decode_backend=video_decode_backend,
                    num_frames=self.num_frames,
                )
            except Exception as e:
                print_log(f'Error in video processing: {e}', 'current')
                print_log(data_dict['video'].replace('mkv', 'mp4'),' current')
                for i in range(max(idx-100000, 0), idx):
                    data_dict = copy.deepcopy(self.text_data[i])
                    if data_dict.get('video', None):
                        video_file = data_dict['video'].replace('mkv', 'mp4')
                        # print('video:', video_file)
                        video_decode_backend = 'decord'

                        video, video_aux, old_h, old_w = load_and_transform_video(
                            os.path.join(self.video_folder, video_file),
                            get_video_transform(
                                video_decode_backend=video_decode_backend,
                                num_frames=self.num_frames,
                                resolution=self.crop_size_raw['height'],
                                padding=self.video_pad_image_to_square,
                            ),
                            transform_aux=get_video_transform(
                                video_decode_backend=video_decode_backend,
                                num_frames=self.num_frames,
                                resolution=self.image_size_aux,
                                padding=self.video_pad_image_to_square,
                            ),
                            video_decode_backend=video_decode_backend,
                            num_frames=self.num_frames,
                        )
                        break
            # print(f'success get video')
            data_dict['pixel_values'] = video
            data_dict['pixel_values_aux'] = video_aux

            data_dict['gt_boxes'] = []
            data_dict['gt_labels'] = []

            box_video = self.json_data[data_dict['id']]
            origin_old_h, origin_old_w = old_h, old_w
            for box_data in box_video['frames']:
                boxes = box_data['boxes']
                labels = box_data['labels']
                scores = box_data['scores']
                if self.video_pad_image_to_square:
                    boxes = adjust_padding_boxes(boxes, height=origin_old_h, width=origin_old_w)
                    padding_size = max(old_h, old_w)
                    old_h, old_w = padding_size, padding_size
                # nms
                boxes, labels, scores = bbox_nms(boxes, labels, scores, self.iou_threshold)
                len_text_ids = len(data_dict['input_ids'])
                close_to_max_length = len_text_ids >= self.max_length - 50
                if len_text_ids >= self.max_length:
                    data_dict['input_ids'] = data_dict['input_ids'][: self.max_length]
                    data_dict['labels'] = data_dict['labels'][: self.max_length]
                    data_dict['length'] = self.max_length
                if self.limit_num is not None or close_to_max_length:
                    if self.limit_mode == 'per_category':
                        num_per_category = 20
                        if close_to_max_length:
                            num_per_category = 1
                        if len(scores) > self.limit_num or close_to_max_length:
                            object_text = []
                            label_indices = defaultdict(list)
                            for index, label in enumerate(labels):
                                label_indices[label].append(index)
                            label_indices = dict(label_indices)

                            for item in label_indices:
                                if len(label_indices[item]) > num_per_category:
                                    item_scores = scores[label_indices[item]]
                                    # item_boxes = boxes[label_indices[item]]
                                    top_scores, top_indices = item_scores.topk(num_per_category, largest=True)
                                    top_boxes_index = [label_indices[item][i] for i in top_indices.tolist()]
                                    label_indices[item] = top_boxes_index
                                    # top_boxes = item_boxes[top_indices]
                                    # label_indices[item] = top_boxes
                            boxes = [boxes[i] for item in label_indices for i in label_indices[item]]
                            try:
                                boxes = torch.stack(boxes)
                            except:
                                print(num_per_category)
                                print(boxes)
                                print(label_indices)
                                print(labels)
                                raise
                            labels = [labels[i] for item in label_indices for i in label_indices[item]]
                            if len(boxes) > self.limit_num:
                                print_log(
                                    'Warning: too many boxes in one image after filter, remain %d boxes'
                                    % self.limit_num
                                )

                    elif self.limit_mode == 'per_image':
                        num_per_image = self.limit_num
                        if close_to_max_length:
                            num_per_image = 20
                        if len(scores) > num_per_image:
                            top_scores, top_indices = scores.topk(num_per_image, largest=True)
                            boxes = boxes[top_indices]
                            labels = [labels[i] for i in top_indices.tolist()]

                    else:
                        raise

                # 坐标是原图尺度的，要映射到resize后的尺度
                boxes, h1, w1 = adjust_short_resize_coordinates(boxes, old_h, old_w, self.image_size_aux)
                boxes, labels = adjust_center_crop_box(boxes, labels, h1, w1, self.image_size_aux)

                data_dict['gt_boxes'].append(boxes)
                data_dict['gt_labels'].append(labels)
            data_dict['id'] = data_dict['id']
            data_dict['modal'] = 'video'

        else:
            data_dict['pixel_values'] = torch.zeros(3, self.crop_size_raw['height'], self.crop_size_raw['width'])
            data_dict['pixel_values_aux'] = torch.zeros(3, self.image_size_aux, self.image_size_aux)
            # 100 无所谓
            data_dict['gt_boxes'] = torch.tensor([0, 0, 100, 100]).reshape(1, 4).float()
            data_dict['gt_labels'] = torch.tensor([0]).reshape(1)
            data_dict['id'] = ''
            data_dict['modal'] = 'text'
        return data_dict

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if not data_dict.get('image', None) and not data_dict.get('video', None):
                cur_len = -cur_len
            elif data_dict.get('video', None):
                cur_len = cur_len+100000
            length_list.append(cur_len)
        return length_list

    def get_data_info(self, idx):
        data = self.text_data[idx]
        if data.get('image', None):
            return 'image'
        elif data.get('video', None):
            return 'video'
        else:
            return 'text'

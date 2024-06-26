import torch
import os
import json
from collections import defaultdict
from mmengine.logging import print_log
from mg_llava.dataset.utils import (
    load_and_transform_video,
    get_video_transform,
    adjust_short_resize_coordinates,
    adjust_center_crop_box,
    adjust_padding_boxes,
    bbox_nms,
)

DEFAULT_VIDEO_TOKEN = '<video>'
VIDEO_TOKEN_INDEX = -201
class VideoObjectLLaVAProxyEvalDataset:

    def __init__(
        self,
        eval_dataset,
        box_json_path,
        iou_threshold=0.2,
        image_size_aux=768,
        limit_num=None,
        limit_mode='per_category',
        num_frames=8,
    ):
        self.eval_ds = eval_dataset

        self.json_data = json.load(open(box_json_path))
        self.video_format = self.json_data[0]['video'].split('.')[-1]
        self.json_data = {item['video'].split('.')[-2]: item for item in self.json_data}
        self.iou_threshold = iou_threshold
        self.image_size_aux = image_size_aux
        self.limit_num = limit_num
        self.limit_mode = limit_mode
        self.is_clip = False
        if hasattr(self.eval_ds.image_processor, 'crop_size'):
            self.crop_size_raw = self.eval_ds.image_processor.crop_size.copy()
            self.eval_ds.image_processor.crop_size['height'] = image_size_aux
            self.eval_ds.image_processor.crop_size['width'] = image_size_aux
            self.eval_ds.image_processor.size['shortest_edge'] = image_size_aux
            self.is_clip = True
        else:
            self.crop_size_raw = self.eval_ds.image_processor.size.copy()
            self.eval_ds.image_processor.size['height'] = image_size_aux
            self.eval_ds.image_processor.size['width'] = image_size_aux
        self.num_frames = num_frames

    def getitem(self, idx, data):
        data_dict = {'img_id': data['id']}
        data_dict['question']=  data['question']
        data_dict['answer'] = data['answer']

        # 1 prepare text
        text = data['question']
        text = DEFAULT_VIDEO_TOKEN + '\n' + text

        if self.eval_ds.use_system:
            inputs = self.eval_ds.template.get('SYSTEM', '{system}').format(system='')
        else:
            inputs = ''
        inputs += self.eval_ds.template['INSTRUCTION'].format(input=text, round=1)

        # 2 tokenize inputs
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_VIDEO_TOKEN)):
            if idx == 0:
                cur_encode = self.eval_ds.tokenizer.encode(chunk)
            else:
                cur_encode = self.eval_ds.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(VIDEO_TOKEN_INDEX)
        ids = torch.tensor(ids)
        data_dict['input_ids'] = ids

        video_file = data['video']+'.'+self.video_format
        # print('video:', video_file)
        video_decode_backend = 'decord'

        video, video_aux, old_h, old_w = load_and_transform_video(
            os.path.join(self.eval_ds.video_folder, video_file),
            get_video_transform(
                video_decode_backend=video_decode_backend,
                num_frames=self.num_frames,
                resolution=self.crop_size_raw['height'],
                padding=self.eval_ds.pad_image_to_square,
            ),
            transform_aux=get_video_transform(
                video_decode_backend=video_decode_backend,
                num_frames=self.num_frames,
                resolution=self.image_size_aux,
                padding=self.eval_ds.pad_image_to_square,
            ),
            video_decode_backend=video_decode_backend,
            num_frames=self.num_frames,
        )

        # print(f'success get video')
        data_dict['pixel_values'] = video
        data_dict['pixel_values_aux'] = video_aux

        data_dict['gt_boxes'] = []
        data_dict['gt_labels'] = []

        box_video = self.json_data[data['video']]
        origin_old_h, origin_old_w = old_h, old_w
        for box_data in box_video['frames']:
            boxes = box_data['boxes']
            labels = box_data['labels']
            scores = box_data['scores']
            if self.eval_ds.pad_image_to_square:
                boxes = adjust_padding_boxes(boxes, height=origin_old_h, width=origin_old_w)
                padding_size = max(old_h, old_w)
                old_h, old_w = padding_size, padding_size
            # nms
            boxes, labels, scores = bbox_nms(boxes, labels, scores, self.iou_threshold)
            if self.limit_num is not None:
                if self.limit_mode == 'per_category':
                    num_per_category = 20
                    if len(scores) > self.limit_num:
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

        return data_dict

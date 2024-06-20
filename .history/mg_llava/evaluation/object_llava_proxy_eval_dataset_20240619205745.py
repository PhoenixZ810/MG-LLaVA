from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
import torch
from PIL import Image
import os
from xtuner.tools.utils import is_cn_string
import torchvision.transforms.functional as F
from bbox_llava.module import bbox_nms
import json
from mg_llava.dataset.utils import adjust_short_resize_coordinates, adjust_center_crop_box, adjust_padding_boxes,
from xtuner.dataset.utils import expand2square
from collections import defaultdict

class ObjectLLaVAProxyEvalDataset:

    def __init__(
        self,
        eval_dataset,
        box_json_path,
        iou_threshold=0.2,
        image_size_aux=768,
        limit_num=None,
        limit_mode='per_category',
        add_bos=False
    ):
        self.eval_ds = eval_dataset

        self.json_data = json.load(open(box_json_path))
        self.json_data = {item['id']: item for item in self.json_data}
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
        # add_bos = True
        self.add_bos = add_bos
    def getitem(self, idx, data):
        data_dict = {'img_id': data['img_id']}

        # 1 prepare text
        if self.eval_ds.metainfo['name'] == 'multiple_choice':
            # MultipleChoiceDataset
            data_dict['index'] = data['index']
            if data['context'] is not None:
                text = data['context'] + '\n' + data['question'] + '\n' + data['options']
            else:
                text = data['question'] + '\n' + data['options']
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

            if is_cn_string(text):
                text = text + '请直接回答选项字母。'
            else:
                text = text + ("Answer with the option's letter from the " 'given choices directly.')
        elif self.eval_ds.metainfo['name'] in ['chartqa', 'gvqa']:
            # TODO prompt are different of vlmevalkit
            text = data['question'] + '\nAnswer the question using a single word or phrase.'
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
        elif self.eval_ds.metainfo['name'] == 'tallyqa':
            text = data['question']
            text = text + "\nAnswer the question using a single number."
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
        elif self.eval_ds.metainfo['name'] in ['hallusion', 'pope']:
            # TODO prompt are different of vlmevalkit
            text = data['question'] + '\nPlease answer the question with yes or no.'
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
        else:
            text = data['question']
            if self.eval_ds.metainfo['name']=='mme':
                text = data['question'].replace('Please answer yes or no.', 'Please answer the question only a single word yes or no.')
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if self.eval_ds.use_system:
            inputs = self.eval_ds.template.get('SYSTEM', '{system}').format(system='')
        else:
            inputs = ''
        inputs += self.eval_ds.template['INSTRUCTION'].format(input=text, round=1)

        # 2 tokenize inputs
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                # if not self.add_bos:
                #     cur_encode = self.eval_ds.tokenizer.encode(chunk)
                # else:
                # add bos token
                bos_token_id = self.eval_ds.tokenizer.bos_token_id
                cur_encode = [bos_token_id]
                cur_encode += self.eval_ds.tokenizer.encode(chunk, add_special_tokens=False)
            else:
                cur_encode = self.eval_ds.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids)
        data_dict['input_ids'] = ids

        # 3 process image
        if self.eval_ds.metainfo['name'] in ['mme', 'textvqa', 'gqa', 'tallyqa']:
            # MMEDataset or TextVQADataset
            image = Image.open(os.path.join(self.eval_ds.image_folder, data['image_path'])).convert('RGB')
        else:
            image = self.eval_ds.get_image(data['img']).convert('RGB')
        old_w, old_h = F.get_image_size(image)

        if self.eval_ds.metainfo['name'] == 'textvqa':
            box_data = self.json_data[data['question_id']]
        else:
            box_data = self.json_data[data['index']]
        boxes = box_data['boxes']
        labels = box_data['labels']
        scores = box_data['scores']

        if self.eval_ds.pad_image_to_square:
            image = expand2square(image, tuple(int(x * 255) for x in self.eval_ds.image_processor.image_mean))
            # print(boxes)
            boxes = adjust_padding_boxes(boxes, height=old_h, width=old_w)
            old_w, old_h = F.get_image_size(image)

        image_aux = self.eval_ds.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        new_h, new_w = image_aux.shape[-2:]
        data_dict['pixel_values_aux'] = image_aux

        image = image_aux.clone()
        image = torch.nn.functional.interpolate(
            image[None], size=[self.crop_size_raw['height'], self.crop_size_raw['width']], mode='bilinear', align_corners=False
        )[0]
        data_dict['pixel_values'] = image

        # nms
        boxes, labels, scores = bbox_nms(boxes, labels, scores, self.iou_threshold)

        # if self.limit_num!=None:
        #     if len(boxes) > self.limit_num:
        #         top_scores, top_indices = scores.topk(20, largest=True)
        #         boxes = boxes[top_indices]
        #         labels = [labels[i] for i in top_indices.tolist()]
        if self.limit_num is not None:
            if self.limit_mode == 'per_category':
                if self.limit_num is not None and self.limit_num > 50:
                    num_per_category = 20
                elif self.limit_num is not None and self.limit_num < 50:
                    num_per_category = 5

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
                    # if len(boxes) > self.limit_num:
                    #     print_log('Warning: too many boxes in one image after filter, remain %d boxes' % self.limit_num)

            elif self.limit_mode == 'per_image':
                num_per_image = self.limit_num
                if len(scores) > num_per_image:
                    top_scores, top_indices = scores.topk(num_per_image, largest=True)
                    boxes = boxes[top_indices]
                    labels = [labels[i] for i in top_indices.tolist()]

        # TODO: 如果有裁剪或者 padding 操作，则本代码有问题
        # 坐标是原图尺度的，要映射到resize后的尺度
        if self.is_clip:
            boxes, h1, w1 = adjust_short_resize_coordinates(boxes, old_h, old_w, self.image_size_aux)
            boxes, labels = adjust_center_crop_box(boxes, labels, h1, w1, self.image_size_aux)
        else:
            boxes[:, 0::2] = boxes[:, 0::2] / old_w * new_w
            boxes[:, 1::2] = boxes[:, 1::2] / old_h * new_h

        data_dict['gt_boxes'] = boxes
        data_dict['gt_labels'] = labels

        return data_dict


class ChatObjectLLaVAProxyEvalDataset:

    def __init__(
        self,
        eval_dataset,
        box_json_path,
        iou_threshold=0.2,
        image_size_aux=768,
        limit_num=None,
        limit_mode='per_category',
        add_bos=False,
    ):
        self.eval_ds = eval_dataset

        self.json_data = json.load(open(box_json_path))
        self.json_data = {item['id']: item for item in self.json_data}
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
        # add_bos = True
        self.add_bos = add_bos

    def getitem(self, idx, data):
        data_dict = {'img_id': data['img_id']}

        # 1 prepare text
        if self.eval_ds.metainfo['name'] == 'multiple_choice':
            # MultipleChoiceDataset
            data_dict['index'] = data['index']
            if data['context'] is not None:
                text = data['context'] + '\n' + data['question'] + '\n' + data['options']
            else:
                text = data['question'] + '\n' + data['options']
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

            # if is_cn_string(text):
            #     text = text + '请直接回答选项字母。'
            # else:
            #     text = text + ("Answer with the option's letter from the " 'given choices directly.')
        elif self.eval_ds.metainfo['name'] in ['chartqa', 'gvqa']:
            # TODO prompt are different of vlmevalkit
            # text = data['question'] + '\nAnswer the question using a single word or phrase.'
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
        elif self.eval_ds.metainfo['name'] == 'tallyqa':
            text = data['question']
            # text = text + "\nDescribe them in details."
            text = "\nDescribe them in details."
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
        elif self.eval_ds.metainfo['name'] in ['hallusion', 'pope']:
            # TODO prompt are different of vlmevalkit
            # text = data['question'] + '\nPlease answer the question with yes or no.'
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
        else:
            text = data['question']
            # if self.eval_ds.metainfo['name'] == 'mme':
            #     text = data['question'].replace(
            #         'Please answer yes or no.', 'Please answer the question only a single word yes or no.'
            #     )
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if self.eval_ds.use_system:
            inputs = self.eval_ds.template.get('SYSTEM', '{system}').format(system='')
        else:
            inputs = ''
        inputs += self.eval_ds.template['INSTRUCTION'].format(input=text, round=1)

        # 2 tokenize inputs
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                # if not self.add_bos:
                #     cur_encode = self.eval_ds.tokenizer.encode(chunk)
                # else:
                # add bos token
                bos_token_id = self.eval_ds.tokenizer.bos_token_id
                cur_encode = [bos_token_id]
                cur_encode += self.eval_ds.tokenizer.encode(chunk, add_special_tokens=False)
            else:
                cur_encode = self.eval_ds.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids)
        data_dict['input_ids'] = ids

        # 3 process image
        if self.eval_ds.metainfo['name'] in ['mme', 'textvqa', 'gqa', 'tallyqa']:
            # MMEDataset or TextVQADataset
            image = Image.open(os.path.join(self.eval_ds.image_folder, data['image_path'])).convert('RGB')
        else:
            image = self.eval_ds.get_image(data['img']).convert('RGB')
        old_w, old_h = F.get_image_size(image)

        if self.eval_ds.metainfo['name'] == 'textvqa':
            box_data = self.json_data[data['question_id']]
        else:
            box_data = self.json_data[data['index']]
        boxes = box_data['boxes']
        labels = box_data['labels']
        scores = box_data['scores']

        if self.eval_ds.pad_image_to_square:
            image = expand2square(image, tuple(int(x * 255) for x in self.eval_ds.image_processor.image_mean))
            # print(boxes)
            boxes = adjust_padding_boxes(boxes, height=old_h, width=old_w)
            old_w, old_h = F.get_image_size(image)

        image_aux = self.eval_ds.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        new_h, new_w = image_aux.shape[-2:]
        data_dict['pixel_values_aux'] = image_aux

        image = image_aux.clone()
        image = torch.nn.functional.interpolate(
            image[None],
            size=[self.crop_size_raw['height'], self.crop_size_raw['width']],
            mode='bilinear',
            align_corners=False,
        )[0]
        data_dict['pixel_values'] = image

        # nms
        boxes, labels, scores = bbox_nms(boxes, labels, scores, self.iou_threshold)

        # if self.limit_num!=None:
        #     if len(boxes) > self.limit_num:
        #         top_scores, top_indices = scores.topk(20, largest=True)
        #         boxes = boxes[top_indices]
        #         labels = [labels[i] for i in top_indices.tolist()]
        if self.limit_num is not None:
            if self.limit_mode == 'per_category':
                if self.limit_num is not None and self.limit_num > 50:
                    num_per_category = 20
                elif self.limit_num is not None and self.limit_num < 50:
                    num_per_category = 5

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
                    # if len(boxes) > self.limit_num:
                    #     print_log('Warning: too many boxes in one image after filter, remain %d boxes' % self.limit_num)

            elif self.limit_mode == 'per_image':
                num_per_image = self.limit_num
                if len(scores) > num_per_image:
                    top_scores, top_indices = scores.topk(num_per_image, largest=True)
                    boxes = boxes[top_indices]
                    labels = [labels[i] for i in top_indices.tolist()]

        # TODO: 如果有裁剪或者 padding 操作，则本代码有问题
        # 坐标是原图尺度的，要映射到resize后的尺度
        if self.is_clip:
            boxes, h1, w1 = adjust_short_resize_coordinates(boxes, old_h, old_w, self.image_size_aux)
            boxes, labels = adjust_center_crop_box(boxes, labels, h1, w1, self.image_size_aux)
        else:
            boxes[:, 0::2] = boxes[:, 0::2] / old_w * new_w
            boxes[:, 1::2] = boxes[:, 1::2] / old_h * new_h

        data_dict['gt_boxes'] = boxes
        data_dict['gt_labels'] = labels

        return data_dict

# Copyright (c) OpenMMLab. All rights reserved.
import base64
import copy
import io
from io import BytesIO
from itertools import chain

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import batched_nms

import math
import numpy as np
import cv2
import requests
from PIL import Image

from mmengine.model import BaseModule, ModuleList
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX

DEFAULT_VIDEO_TOKEN = '<video>'
VIDEO_TOKEN_INDEX = -201
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import (
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
    CenterCropVideo,
)
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge('torch')


def get_bos_eos_token_ids(tokenizer):
    if tokenizer.__class__.__name__ in ['QWenTokenizer', 'QWen2Tokenizer', 'Qwen2TokenizerFast']:
        bos_token_id = []
        eos_token_id = tokenizer.eos_token_id
        assert eos_token_id is not None, 'Please set eos_token for Qwen tokenizer!'
    elif tokenizer.__class__.__name__ == 'ChatGLMTokenizer':
        bos_token_id = [64790, 64792]
        eos_token_id = tokenizer.eos_token_id
    else:
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
    if isinstance(bos_token_id, int):
        bos_token_id = [bos_token_id]
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    return bos_token_id, eos_token_id


def encode_fn(example, tokenizer, max_length, input_ids_with_output=True, with_image_token=False):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        if DEFAULT_IMAGE_TOKEN in input and with_image_token:
            chunk_encode = [
                tokenizer.encode(chunk, add_special_tokens=False)
                for chunk in input.split(DEFAULT_IMAGE_TOKEN)
            ]
            assert len(chunk_encode) == 2
            input_encode = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_encode.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    input_encode.append(IMAGE_TOKEN_INDEX)
        else:
            input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output
            output_with_loss = single_turn_conversation.get('output_with_loss', True)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}


def encode_fn_qformer(example, tokenizer, qformer_tokenizer, max_length, input_ids_with_output=True, with_image_token=False):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True

    qformer_text = ''
    for msg in example['conversations']:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                promp = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
            else:
                promp = msg['value'].strip()
            qformer_text+=promp
    qformer_text_encoding = qformer_tokenizer(
        text=qformer_text,
        truncation=True,
        return_tensors="pt",
        )

    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        if DEFAULT_IMAGE_TOKEN in input and with_image_token:
            chunk_encode = [
                tokenizer.encode(chunk, add_special_tokens=False)
                for chunk in input.split(DEFAULT_IMAGE_TOKEN)
            ]
            assert len(chunk_encode) == 2
            input_encode = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_encode.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    input_encode.append(IMAGE_TOKEN_INDEX)
        else:
            input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output
            output_with_loss = single_turn_conversation.get('output_with_loss', True)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {
        'input_ids': input_ids,
        'labels': labels,
        'qformer_input_ids': qformer_text_encoding["qformer_input_ids"],
        'qformer_attention_mask': qformer_text_encoding["qformer_attention_mask"],
    }


class Packer:
    """Pack multiple pieces of data into one."""

    def __init__(self, chunk_size=2048, use_varlen_attn=False, drop_last=False):
        self.chunk_size = chunk_size
        self.residual = {'input_ids': [], 'labels': []}
        self.use_varlen_attn = use_varlen_attn
        self.drop_last = drop_last
        if use_varlen_attn:
            self.residual_cumulative_len = [0]

    def get_cumulative_len(self, chunk_num):
        ptr_l = 0
        cumulative_len = []
        for chunk_idx in range(chunk_num):
            length_train = (chunk_idx + 1) * self.chunk_size
            ptr_r = np.searchsorted(self.residual_cumulative_len, length_train, side='left')
            if self.residual_cumulative_len[ptr_r] == length_train:
                cumulative_len_cur = self.residual_cumulative_len[ptr_l : ptr_r + 1]
                ptr_l = ptr_r + 1
            else:
                cumulative_len_cur = self.residual_cumulative_len[ptr_l:ptr_r] + [length_train]
                ptr_l = ptr_r
            cumulative_len_cur = [num - chunk_idx * self.chunk_size for num in cumulative_len_cur]
            if cumulative_len_cur[0] != 0:
                cumulative_len_cur = [0] + cumulative_len_cur

            cumulative_len.append(cumulative_len_cur)

        self.residual_cumulative_len = [num - length_train for num in self.residual_cumulative_len[ptr_l:]]
        if len(self.residual_cumulative_len) == 0:
            self.residual_cumulative_len = [0]
        elif self.residual_cumulative_len[0] != 0:
            self.residual_cumulative_len = [0] + self.residual_cumulative_len

        return cumulative_len

    def get_position_ids(self, cumulative_len):
        position_ids = []
        for cumulative_len_cur in cumulative_len:
            index_cur = []
            for i in range(len(cumulative_len_cur) - 1):
                index_cur.extend(list(range(cumulative_len_cur[i + 1] - cumulative_len_cur[i])))  # noqa: W504
            position_ids.append(index_cur)
        return position_ids

    def __call__(self, batch):
        concatenated_samples = {k: v + list(chain(*batch[k])) for k, v in self.residual.items()}

        if self.use_varlen_attn:
            for input_id in batch['input_ids']:
                self.residual_cumulative_len.append(self.residual_cumulative_len[-1] + len(input_id))

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)  # noqa: W504
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {k: v[(chunk_num * self.chunk_size) :] for k, v in concatenated_samples.items()}

            if self.use_varlen_attn:
                cumulative_len = self.get_cumulative_len(chunk_num)
                result['cumulative_len'] = cumulative_len
                result['position_ids'] = self.get_position_ids(cumulative_len)
        else:
            if self.drop_last:
                result = {k: [] for k, v in concatenated_samples.items()}
            else:
                result = {k: [v] for k, v in concatenated_samples.items()}

            self.residual = {k: [] for k in concatenated_samples.keys()}

            if self.use_varlen_attn:
                result['cumulative_len'] = [] if self.drop_last else [self.residual_cumulative_len]
                result['position_ids'] = (
                    [] if self.drop_last else self.get_position_ids([self.residual_cumulative_len])
                )
                self.residual_cumulative_len = [0]

        return result


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


# ----------------------------------------------------------------------
# ref: https://github.com/haotian-liu/LLaVA
def select_best_resolution(original_size, possible_resolutions):
    """Selects the best resolution from a list of possible resolutions based on
    the original size.

    Args:
        original_size (tuple): The original size of the image in the format
            (width, height).
        possible_resolutions (list): A list of possible resolutions in
            the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution, pad_mean):
    """Resize and pad an image to a target resolution while maintaining aspect
    ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of
            the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), pad_mean)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    # 居中 padding
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def process_anyres_image(
    image,
    processor,
    possible_resolutions,
    patch_size,
    shortest_edge,
    pad_mean=(0, 0, 0),
    orig_img_pad_to_square=False,
):
    """Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        possible_resolutions (str): A string representation of a list of
            possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution, pad_mean)

    patches = divide_to_patches(image_padded, patch_size)

    if orig_img_pad_to_square:
        # 不是居中 padding
        image = expand2square(image, pad_mean)

    image_original_resize = image.resize((shortest_edge, shortest_edge))

    image_patches = [image_original_resize] + patches
    image_patches = [
        processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
        for image_patch in image_patches
    ]
    return torch.stack(image_patches, dim=0)


def get_anyres_image_grid_shape(image_size, possible_resolutions, patch_size):
    """Calculate the shape of the image patch grid after the preprocessing for
    images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format
            (width, height).
        possible_resolutions (list): A string representation of a list of
            possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def unpad_image(tensor, original_size):
    """Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


# ----------------------------------------------------------------------

def bbox_nms(boxes, labels, scores, iou_threshold=0.2):
    char_to_index = {char: index for index, char in enumerate(set(labels))}
    index_to_char = {index: char for char, index in char_to_index.items()}
    index_list = [char_to_index[char] for char in labels]
    boxes = torch.tensor(boxes).reshape(-1, 4)
    scores = torch.tensor(scores)
    labels = torch.tensor(index_list)
    keep = batched_nms(boxes.float(), scores, labels, iou_threshold)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    return boxes.float(), [index_to_char[label.item()] for label in labels], scores


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def get_video_transform(video_decode_backend, num_frames, resolution, padding=False):

    if video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=resolution),
                    CenterCropVideo(resolution),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif video_decode_backend == 'decord':
        if padding:
            transform = Compose(
                [
                    # UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    padding_video(mean=OPENAI_DATASET_MEAN),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=resolution),
                    CenterCropVideo(resolution),
                    # RandomHorizontalFlipVideo(p=0.5),
                ]
            )
        else:
            transform = Compose(
                [
                    # UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=resolution),
                    CenterCropVideo(resolution),
                    # RandomHorizontalFlipVideo(p=0.5),
                ]
            )

    elif video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=resolution),
                CenterCropVideo(resolution),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform


def load_and_transform_video(
    video_path,
    transform,
    num_frames,
    transform_aux=None,
    video_decode_backend='opencv',
    clip_start_sec=0.0,
    clip_end_sec=None,
):
    video_outputs_aux = None
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        # print(duration)
        frame_id_list = np.linspace(0, duration - 1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        # print(f'path:{video_path}, {video_data.shape}')
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        h, w = video_data.shape[2], video_data.shape[3]
        if transform_aux is not None:
            video_data_aux = video_data.clone()
            video_outputs_aux = transform_aux(video_data_aux)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration - 5, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret:
                raise ValueError(f'video error at {video_path}')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs, video_outputs_aux, h, w


class padding_video:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, video):
        # video: (C, T, H, W)
        target_size = max(video.size(2), video.size(3))

        padding_top = (target_size - video.size(2)) // 2
        padding_bottom = target_size - video.size(2) - padding_top
        padding_left = (target_size - video.size(3)) // 2
        padding_right = target_size - video.size(3) - padding_left

        padded_video = torch.zeros(video.size(0), video.size(1), target_size, target_size)

        for c in range(video.size(0)):
            padded_video[c] = self.mean[c]

        padded_video[
            :, :, padding_top : padding_top + video.size(2), padding_left : padding_left + video.size(3)
        ] = video

        return padded_video

    def __repr__(self) -> str:
        return f"class padding_video"


def adjust_padding_boxes(boxes, height, width):
    new_boxes_list = []
    for box in boxes:
        if width > height:
            delta = (width - height) // 2
            new_box = (box[0], box[1] + delta, box[2], box[3] + delta)
        elif height > width:
            delta = (height - width) // 2
            new_box = (box[0] + delta, box[1], box[2] + delta, box[3])
        else:
            new_box = box
        new_boxes_list.append(new_box)
    return new_boxes_list


def adjust_short_resize_coordinates(original_boxes, original_height, original_width, new_short_edge_length):
    # 计算缩放比例
    scale = new_short_edge_length / min(original_width, original_height)

    # 调整bounding box坐标
    # x_min, y_min, x_max, y_max = box
    # new_x_min = int(x_min * scale)
    # new_y_min = int(y_min * scale)
    # new_x_max = int(x_max * scale)
    # new_y_max = int(y_max * scale)
    boxes = original_boxes * scale
    boxes = torch.round(boxes).int()
    new_h = int(original_height * scale)
    new_w = int(original_width * scale)
    return boxes, new_h, new_w


def adjust_center_crop_box(original_boxes, original_labels, original_height, original_width, crop_size):
    # 裁剪尺寸
    crop_height = crop_size
    crop_width = crop_size

    # 计算裁剪的起始点
    start_x = (original_width - crop_width) // 2
    start_y = (original_height - crop_height) // 2

    # 调整box坐标
    new_boxes = []
    new_labels = []
    for i in range(original_boxes.size(0)):
        original_box = original_boxes[i]
        x_min, y_min, x_max, y_max = original_box.tolist()
        new_x_min = max(x_min - start_x, 0)
        new_y_min = max(y_min - start_y, 0)
        new_x_max = min(x_max - start_x, crop_width)
        new_y_max = min(y_max - start_y, crop_height)
        if new_x_max < 0 or new_y_max < 0 or new_x_min > crop_width or new_y_min > crop_height:
            continue
        new_boxes.append(torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max]))
        new_labels.append(original_labels[i])
    # 返回调整后的box坐标
    if new_boxes:
        stacked_boxes = torch.stack(new_boxes).float()
    else:
        stacked_boxes = torch.stack([torch.zeros(4)]).float()
    return stacked_boxes, new_labels


def find_noun_phrases(caption: str) -> list:
    """Find noun phrases in a caption using nltk.
    Args:
        caption (str): The caption to analyze.

    Returns:
        list: List of noun phrases found in the caption.

    Examples:
        >>> caption = 'There is two cat and a remote in the picture'
        >>> find_noun_phrases(caption) # ['cat', 'a remote', 'the picture']
    """
    try:
        import nltk

        nltk.download('punkt', download_dir='~/nltk_data')
        nltk.download('averaged_perceptron_tagger', download_dir='~/nltk_data')
    except ImportError:
        raise RuntimeError('nltk is not installed, please install it by: ' 'pip install nltk.')

    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}'
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = []
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))
    filtered_list = [s for s in noun_phrases if 'image' not in s]
    return filtered_list


def box_to_position_embedding(boxes, image_width, image_height):
    # boxes: Tensor of shape (N, 4) where N is the number of boxes
    # Each box is represented by (x_min, y_min, x_max, y_max)

    # Convert to center coordinates and sizes
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]

    # Normalize coordinates by image dimensions
    center_x /= image_width
    center_y /= image_height
    width /= image_width
    height /= image_height

    # Stack and return position embeddings
    position_embeddings = torch.stack((center_x, center_y, width, height), dim=1)
    return position_embeddings


def coordinate_to_encoding(
    coord_tensor: Tensor, num_feats: int = 128, temperature: int = 10000, scale: float = 2 * math.pi
):
    """Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(2)
    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(coord_tensor.size(-1)))
    return pos  # [n, num_feat/2, 8]


class MLP(BaseModule):
    """Very simple multi-layer perceptron (also called FFN) with relu. Mostly
    used in DETR series detectors.

    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers. As the last
            layer of MLP only contains FFN (Linear).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = ModuleList(torch.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.

        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

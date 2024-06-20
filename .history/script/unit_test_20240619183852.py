from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    CLIPImageProcessor,
    CLIPVisionModel,
)

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.collate_fns import mm_collate_fn
from object_llava.module import ObjectLLaVAModel, ObjectLLaVADataset, OpenCLIPVisionTower
from mg_llava.evaluation.m4c_evaluator import EvalAIAnswerProcessor
from bbox_llava.module.utils import bbox_nms
from xtuner.dataset.samplers import LengthGroupedSampler
from mmengine.runner.runner import Runner
from mmengine import Config
from xtuner.registry import BUILDER

import os
import os.path as osp
import json
import numpy as np
import torch
import cv2
from mmengine.visualization import Visualizer
from tqdm import tqdm
from multiprocessing import Pool
import random
from functools import partial
from PIL import Image

from torch.utils.data import DataLoader, Dataset

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
# llm_name_or_path = '/mnt/petrelfs/share_data/huanghaian/model/phi-2'
llm_name_or_path = '/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5/'
# visual_encoder_name_or_path = '/mnt/petrelfs/share_data/huanghaian/model/siglip-so400m-patch14-384'
visual_encoder_name_or_path = '/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/'
visual_encoder_aux_name = 'model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup'
visual_encoder_aux_path = '/mnt/petrelfs/share_data/zhaoxiangyu/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/snapshots/39918dfbdf69ccd2172e6510a430e92337ee23e1/'
# --------------------------------------------------------------------------------------------
# Data
# data_root = './data/llava_data/'
# data_path = data_root + 'LLaVA-Pretrain/blip_laion_cc_sbu_558k.json'
# offline_processed_text_folder = 'data/text_cache/llava_pretrain_all'
# image_folder = data_root + 'LLaVA-Pretrain/images'

# data_root = '/mnt/petrelfs/share_data/huanghaian/llava_data/'
# data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
# offline_processed_text_folder = 'data/text_cache/llava_sft_all'
# # data_path = 'debug/llava_v1_5_mix5k.json'
# # offline_processed_text_folder = 'debug/llava_sft_5k'
# image_folder = data_root + 'llava_images'

# prompt_template = PROMPT_TEMPLATE.vicuna
# max_length = int(2048 - (384 // 14) ** 2 -100)

# --------------------------------------------------------------------------------------------

box_json_path = '/mnt/petrelfs/share_data/zhaoxiangyu/pretrain_valley_llava_126k_only_box.json'
data_path = '/mnt/petrelfs/share_data/zhaoxiangyu/valley_llavaimage_fixed.json'
offline_processed_text_folder = 'data/text_cache/video_llava_pretrain_all'
# offline_processed_text_folder = None
image_folder = '/mnt/petrelfs/zhaoxiangyu/code_new/xtuner/data/llava_pretrain'
video_folder = '/mnt/petrelfs/share_data/duanhaodong/'
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = int(2048 - (336 // 14) ** 2 -100)

# --------------------------------------------------------------------------------------------
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right',
)

# image_processor = dict(
#     type=SiglipImageProcessor.from_pretrained,
#     pretrained_model_name_or_path=visual_encoder_name_or_path,
#     trust_remote_code=True,
# )

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True,
)

model = dict(
    type=ObjectLLaVAModel,
    tokenizer=tokenizer,
    template=prompt_template,
    image_processor=image_processor,
    freeze_llm=True,
    freeze_visual_encoder=True,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
    ),
    visual_encoder=dict(
        type=SiglipVisionModel.from_pretrained, pretrained_model_name_or_path=visual_encoder_name_or_path
    ),
    visual_encoder_aux=dict(
        type=OpenCLIPVisionTower,
        vision_tower=visual_encoder_aux_name,
        vision_tower_path=visual_encoder_aux_path,
        optimize_vision_tower_aux=False,
    ),
)
# model_type = model.pop('type')
# model = model_type(**model)

visual_encoder_aux = dict(
    type=OpenCLIPVisionTower,
    vision_tower=visual_encoder_aux_name,
    vision_tower_path=visual_encoder_aux_path,
    optimize_vision_tower_aux=False,
    use_multi_level=True,
)
# visual_encoder = visual_encoder_aux.pop('type')
# visual_encoder = visual_encoder(**visual_encoder_aux)

# pretrain_dataset = dict(
#     type=ObjectLLaVADataset,
#     data_path=data_path,
#     offline_processed_text_folder=offline_processed_text_folder,
#     box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/pretrain_valley_llava_126k_only_box.json',
#     image_folder=image_folder,
#     tokenizer=tokenizer,
#     image_processor=image_processor,
#     dataset_map_fn=llava_map_fn,
#     template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     pad_image_to_square=False,
#     image_size_aux=768,
#     limit_num=20
# )

pretrain_dataset = dict(
    type=ObjectLLaVADataset,
    data_path=data_path,
    offline_processed_text_folder=offline_processed_text_folder,
    box_json_path=box_json_path,
    image_folder=image_folder,
    video_folder=video_folder,
    include_video=True,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    image_size_aux=768,
)

sft_dataset = dict(
    type=ObjectLLaVADataset,
    offline_processed_text_folder=None,
    box_json_path='allava-instruct-vflan4v_203k_only_box.json',
    data_path='/mnt/petrelfs/share_data/zhaoxiangyu/allava_vflan/ALLaVA-Instruct-VFLAN-4V-fixed.json',
    image_folder='/mnt/petrelfs/share_data/zhaoxiangyu',
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
    image_size_aux=768,
    limit_num=100,
)
train_dataloader = dict(
    batch_size=1,
    num_workers=16,
    pin_memory=True,
    dataset=sft_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=1,
    ),
    collate_fn=dict(type=mm_collate_fn, extra_collate_keys=['gt_boxes', 'gt_labels', 'pixel_values_aux', 'id']),
)

# class_type = sft_dataset.pop('type')
# obj_llava_dataset = class_type(**sft_dataset)

# --------------------------------------------------------------------------------------------------------
# train_dataloader = Runner.build_dataloader(train_dataloader)
# all_num_tokens = []
# all_img_names = []
# sum_num_tokens = 0
# num = 0
# for load in tqdm(train_dataloader):
#     # print(load)
#     text_num_tokens = load['data']['input_ids'].shape[1]
#     image_num_tokens = load['data']['gt_boxes'][0].shape[0]
#     num_tokens = text_num_tokens + image_num_tokens
#     num_list = [image_num_tokens, text_num_tokens, num_tokens, load['data']['id'][0]]
#     if num ==0:
#         print(num_list)
#     all_num_tokens.append(num_list)
#     if load['data']['id'][0]:
#         sum_num_tokens += num_tokens
#         num+=1
# with open('all_num_tokens_new1-1-300.txt', 'w') as f:
#     for item in all_num_tokens:
#         f.write("%s\n" % item)
# average_num_tokens = sum_num_tokens / num
# print(f'average_num_tokens={average_num_tokens}')

# break

# for data in obj_llava_dataset:
#     print(data)
#     break

# ------------------------------------------------------------------------------------------------------
image_mean = ([0.48145466, 0.4578275, 0.40821073],)
image_std = [0.26862954, 0.26130258, 0.27577711]
mean = torch.tensor(image_mean).view(3, 1, 1)
std = torch.tensor(image_std).view(3, 1, 1)
vis = Visualizer()


def _get_adaptive_scales(areas: np.ndarray, min_area: int = 800, max_area: int = 30000) -> np.ndarray:
    scales = 0.5 + (areas - min_area) // (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def visual_super_boxes_image(data, save_path):
    # for data in tqdm(obj_llava_dataset, total = len(obj_llava_dataset)):
    if 'gt_boxes' in data:
        bboxes = data['gt_boxes']
        labels = data['gt_labels']
        bboxes[:, 0::2] = bboxes[:, 0::2] / 768 * 336
        bboxes[:, 1::2] = bboxes[:, 1::2] / 768 * 336
        if len(bboxes)>0:
            pixel_values = data['pixel_values']
            pixel_values = pixel_values * std + mean
            pixel_values = pixel_values * 255
            pixel_values = torch.permute(pixel_values, (1, 2, 0))

            vis.set_image(pixel_values.numpy())

            conversation = data['conversation'][0]['input']
            print(conversation)
            print(data['conversation'][0]['output'])

            image2colors = []
            for _ in range(len(bboxes)):
                colors = np.random.random((1, 3)) * 0.7 + 0.3
                colors = (colors * 255).astype(int).tolist()[0]
                image2colors.append(tuple(colors))

            bboxes = np.array(bboxes).reshape(-1, 4)
            positions = bboxes[:, :2] + 3
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)
            if labels:
                vis.draw_bboxes(bboxes, edge_colors=image2colors, line_widths=3)
                vis.draw_texts(
                    labels,
                    positions,
                    colors='g',
                    font_sizes=[int(13 * s) for s in scales],
                    bboxes=[{'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}] * len(scales),
                )

            # vis.show()
            drawn_img = vis.get_image()
            ids = data['id']
            # ids = data['img_id']
            cv2.imwrite(os.path.join(save_path, f'{ids}.jpg'), drawn_img[..., ::-1])
            print(f'{ids}.jpg created!')
            return True
        else:
            print('no_boxes')
            return False
    else:
        return False


def visual_super_boxes_video(data, save_path):
    # for data in tqdm(obj_llava_dataset, total = len(obj_llava_dataset)):
    if 'gt_boxes' in data:
        video_bboxes = data['gt_boxes']
        video_labels = data['gt_labels']
        for idx,bboxes in enumerate(video_bboxes):
            bboxes[:, 0::2] = bboxes[:, 0::2] / 768 * 336
            bboxes[:, 1::2] = bboxes[:, 1::2] / 768 * 336
            labels = video_labels[idx]
            if len(bboxes) > 0:
                pixel_values = data['pixel_values'][:,idx,:,:]
                pixel_values = pixel_values * std + mean
                pixel_values = pixel_values * 255
                pixel_values = torch.permute(pixel_values, (1, 2, 0))

                vis.set_image(pixel_values.numpy())

                # conversation = data['conversation'][0]['input']
                # print(conversation)
                # print(data['conversation'][0]['output'])

                image2colors = []
                for _ in range(len(bboxes)):
                    colors = np.random.random((1, 3)) * 0.7 + 0.3
                    colors = (colors * 255).astype(int).tolist()[0]
                    image2colors.append(tuple(colors))

                bboxes = np.array(bboxes).reshape(-1, 4)
                positions = bboxes[:, :2] + 3
                areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                scales = _get_adaptive_scales(areas)
                if labels:
                    vis.draw_bboxes(bboxes, edge_colors=image2colors, line_widths=3)
                    vis.draw_texts(
                        labels,
                        positions,
                        colors='g',
                        font_sizes=[int(13 * s) for s in scales],
                        bboxes=[{'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}] * len(scales),
                    )

                # vis.show()
                drawn_img = vis.get_image()
                # ids = data['id']
                ids = data['img_id']
                cv2.imwrite(os.path.join(save_path, f'{ids}_{idx}.jpg'), drawn_img[..., ::-1])
        return True

    #         else:
    #             return False
    # else:
    #     return False

# ---------------------------------------------------------------------------------------------
# from object_llava.evaluation import ObjectLLaVAProxyEvalDataset
# from xtuner.dataset.evaluation import (
#     MMEDataset,
#     MultipleChoiceDataset,
#     POPEDataset,
#     HallusionDataset,
#     TextVQADataset,
# )


# seed_dataset = dict(
#     type=MultipleChoiceDataset,
#     proxy_eval_dataset=dict(
#         type=ObjectLLaVAProxyEvalDataset,
#         box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/SEEDBench_IMG_only_bbox.json',
#         image_size_aux=768,
#         limit_num=100,
#     ),
#     data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/SEEDBench_IMG.tsv',
#     prompt_template=PROMPT_TEMPLATE.vicuna,
#     tokenizer=tokenizer,
#     image_processor=image_processor,
#     pad_image_to_square=True,
# )
# save_path = 'debug/visualiztion/vis_seed'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# class_type = seed_dataset.pop('type')
# test_dataset = class_type(**seed_dataset)
# # for i in tqdm(test_dataset, total=len(test_dataset)):
# #     data = test_dataset[i]
# #     if data['index']==79570:
# #         try:
# #             visual_super_boxes_image(data, save_path)
# #         except:
# #             # print(data['gt_boxes'])
# #             raise
# for i in range(15,50):
#     data = test_dataset[i]
#     try:
#         visual_super_boxes_image(data, save_path)
#     except:
#         # print(data['gt_boxes'])
#         raise
# -----------------------------------------------------------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 确保CPU和GPU结果一致
    torch.backends.cudnn.deterministic = True
    # 可能会减慢训练速度，但是为了可复现性可以选择开启
    torch.backends.cudnn.benchmark = False
setup_seed(2024)
cfg = Config.fromfile('fuse_object_llava/config/vicuna/fuse_more_vicuna7b_clip_L_14_336_sft_padding.py')
train_dataloader = Runner.build_dataloader(cfg.train_dataloader)
for load in tqdm(train_dataloader):
    pass

# -----------------------------------------------------------------------------
# save_path = 'debug/visualiztion/vis_video_msvd_padding'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# cfg = Config.fromfile('object_llava/config/video_vicuna7b_clip_L_14_336_sft.py')
# obj_llava_dataset = BUILDER.build(cfg.train_dataloader.dataset)
# for i in range(15):
#     data = obj_llava_dataset[i]
# try:
#     visual_super_boxes_video(data, save_path)
# except:
#     # print(data['gt_boxes'])
#     raise

# ---------------------------------------------------------------
# save_path = 'debug/visualiztion/vis_changefilter_clip_padding'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# cfg = Config.fromfile('object_llava/config/vicuna7b_clip_L_14_336_sft_padding.py')
# obj_llava_dataset = BUILDER.build(cfg.train_dataloader.dataset)
# for i in range(15):
#     data = obj_llava_dataset[i]
# try:
#     visual_super_boxes_image(data, save_path)
# except:
#     # print(data['gt_boxes'])
#     raise

# -----------------------------------------------------------------------------
# save_path = 'debug/visualiztion/allava_sft_nofilter'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# sft_dataset = dict(
#     type=ObjectLLaVADataset,
#     offline_processed_text_folder='debug/allava_sft_5000',
#     box_json_path='allava-instruct-vflan4v_203k_only_box.json',
#     data_path='debug/allava_sft_5000.json',
#     image_folder='/mnt/petrelfs/share_data/zhaoxiangyu',
#     tokenizer=tokenizer,
#     image_processor=image_processor,
#     dataset_map_fn=llava_map_fn,
#     template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     pad_image_to_square=False,
#     image_size_aux=768,
#     limit_num=100,
# )
# class_type = sft_dataset.pop('type')
# obj_llava_dataset = class_type(**sft_dataset)

# for i in range(15):
#     data = obj_llava_dataset[i]
#     # import pdb;pdb.set_trace()
#     try:
#         visual_super_boxes_image(data, save_path)
#     except:
#         # print(data['gt_boxes'])
#         raise
# --------------------------------------------------------------------------------------
# class_type = pretrain_dataset.pop('type')
# obj_llava_dataset = class_type(**pretrain_dataset)

# num=0
# with Pool(processes=4) as pool:
#     iters = pool.imap(func=visual_super_boxes, iterable=obj_llava_dataset)
#     for iter in tqdm(iters, total=len(obj_llava_dataset)):
#         if iter:
#             num+=1
#     print(num)
# -------------------------------------------------------------------------------------------------
# from xtuner.dataset.evaluation import (
#     MMEDataset,
#     MultipleChoiceDataset,
#     POPEDataset,
#     HallusionDataset,
#     TextVQADataset,
# )
# from object_llava.evaluation import ObjectLLaVAProxyEvalDataset
# image_size_aux = 768
# limit_num = 100
# test_dataset = dict(
#         type=MultipleChoiceDataset,
#         proxy_eval_dataset=dict(
#             type=ObjectLLaVAProxyEvalDataset,
#             box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/MMBench_DEV_EN_only_bbox.json',
#             image_size_aux=image_size_aux,
#             limit_num=limit_num,
#         ),
#         data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/MMBench_DEV_EN.tsv',
#         prompt_template=PROMPT_TEMPLATE.vicuna,
#         tokenizer=tokenizer,
#         image_processor=image_processor,
#         pad_image_to_square=False,
#     )
# class_type = test_dataset.pop('type')
# obj_llava_dataset = class_type(**test_dataset)

# for i in range(15):
#     data = obj_llava_dataset[i]
#     try:
#         visual_super_boxes_image(data)
#     except:
#         # print(data['gt_boxes'])
#         raise
# ---------------------------------------------------------------------------------------------------------
# visual_aux_model = OpenCLIPVisionTower(
#     vision_tower=visual_encoder_aux_name,
#     vision_tower_path=visual_encoder_aux_path,
#     optimize_vision_tower_aux=False,
# )
# image_test = torch.zeros(1, 3, 768, 768)
# result = visual_aux_model(image_test)

# ---------------------------------------------------------------------------------------------------
# def evaluate():
#     answers_file = '/mnt/petrelfs/zhaoxiangyu/code_new/box_xtuner/work_dirs/objectllava_phi2_2_sigclip_L_p14_384_e1_gpu8_finetune_coco80/llava_vqav2_testdev_balanced_merge.jsonl'
#     test_split = '/mnt/petrelfs/share_data/zhaoxiangyu/vqav2_llava_eval/llava_vqav2_mscoco_test2015.jsonl'
#     dst = '/mnt/petrelfs/zhaoxiangyu/code_new/box_xtuner/work_dirs/objectllava_phi2_2_sigclip_L_p14_384_e1_gpu8_finetune_coco80/vqav2_testdev_balanced_predictions.json'

#     results = []
#     error_line = 0
#     for line_idx, line in enumerate(open(answers_file)):
#         try:
#             results.append(json.loads(line))
#         except:
#             error_line += 1

#     results = {x['question_id']: x['text'] for x in results}
#     test_split = [json.loads(line) for line in open(test_split)]
#     split_ids = set([x['question_id'] for x in test_split])

#     print(f'total results: {len(results)}, total split: {len(test_split)}, error_line:{error_line}')

#     all_answers = []

#     answer_processor = EvalAIAnswerProcessor()

#     for x in test_split:
#         if x['question_id'] not in results:
#             all_answers.append({'question_id': x['question_id'], 'answer': ''})
#         else:
#             all_answers.append(
#                 {'question_id': x['question_id'], 'answer': answer_processor(results[x['question_id']])}
#             )

#     with open(dst, 'w') as f:
#         json.dump(all_answers, open(dst, 'w'))

# evaluate()
# --------------------------------------------------------------------------------------------
# with open('/mnt/petrelfs/share_data/zhaoxiangyu/vqav2_llava_eval/vqav2_box.json', 'r') as f:
#     j = json.load(f)
#     json_data = {item['id']: item for item in j}
# data = json_data[436166001]
# with open('/mnt/petrelfs/share_data/zhaoxiangyu/llava_v1_5_mix665k_only_box.json', 'r') as f:
#     j = json.load(f)
#     json_data = {item['id']: item for item in j}
# data = json_data[436166001]
# bboxes = data['boxes']
# print(len(bboxes))
# # bboxes[:, 0::2] = bboxes[:, 0::2] / 768 * 384
# # bboxes[:, 1::2] = bboxes[:, 1::2] / 768 * 384
# labels = data['labels']
# scores = data['scores']
# if len(bboxes)>0:
#     # pixel_values = Image.open('/mnt/petrelfs/share_data/huanghaian/llava_data/vg/VG_100K/1160042.jpg').convert('RGB')
#     pixel_values = Image.open(
#         '/mnt/petrelfs/share_data/huanghaian/llava_data/llava_images/vg/VG_100K/1160042.jpg'
#     ).convert('RGB')
#     pixel_values = np.array(pixel_values)
#     # pixel_values = pixel_values * std + mean
#     # pixel_values = pixel_values * 255
#     # pixel_values = torch.permute(pixel_values, (1, 2, 0))
#     bboxes, labels, scores = bbox_nms(bboxes, labels, scores, 0.2)
#     text = [labels[i] + str(scores[i]) for i in range(len(labels))]
#     print(text)
#     vis.set_image(pixel_values)

#     image2colors = []
#     for _ in range(len(bboxes)):
#         colors = np.random.random((1, 3)) * 0.7 + 0.3
#         colors = (colors * 255).astype(int).tolist()[0]
#         image2colors.append(tuple(colors))

#     bboxes = np.array(bboxes).reshape(-1, 4)
#     print(bboxes.shape)
#     positions = bboxes[:, :2] + 3
#     areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
#     scales = _get_adaptive_scales(areas)
#     vis.draw_bboxes(bboxes, edge_colors=image2colors, line_widths=3)
#     vis.draw_texts(
#         text,
#         positions,
#         colors='g',
#         font_sizes=[int(13 * s) for s in scales],
#         bboxes=[{'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}] * len(scales),
#     )

#     # vis.show()
#     drawn_img = vis.get_image()
#     cv2.imwrite(os.path.join(save_path, f'COCO_test2015_000000436166.jpg'), drawn_img[..., ::-1])
# -----------------------------------------------------------------------------------------------------
# visual_encoder_name_or_path = '/mnt/petrelfs/share_data/basemodel/checkpoints/llm/hf_hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/'
# image_processor = dict(
#     type=CLIPImageProcessor.from_pretrained,
#     pretrained_model_name_or_path=visual_encoder_name_or_path,
#     trust_remote_code=True,
# )
# visual_encoder = (
#     dict(type=CLIPVisionModel.from_pretrained, pretrained_model_name_or_path=visual_encoder_name_or_path),
# )
# image_processor = CLIPImageProcessor.from_pretrained(
#     pretrained_model_name_or_path=visual_encoder_name_or_path, trust_remote_code=True
# )
# image = Image.open('/mnt/petrelfs/share_data/huanghaian/llava_data/llava_images/vg/VG_100K/1160042.jpg').convert('RGB')
# img = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
# ---------------------------------------------------------------------------------------
# from fuse_object_llava.module.instruct_blip import InstructBLIPQFormerTokenizer
# q_tokenizer = InstructBLIPQFormerTokenizer(
#     '/mnt/petrelfs/share_data/zhaoxiangyu/models--Salesforce--instructblip-vicuna-7b/snapshots/52ba0cb2c44d96b2fcceed4e84141dc40d2b6a92'
# )
# text = ['what is the color of the s?', 'what is the color of the k?', 'what is the color of the y?']
# ids = q_tokenizer(
#     text,
#     truncation=True,
#     padding="max_length",
#     return_tensors="pt",
# )
# print(ids)

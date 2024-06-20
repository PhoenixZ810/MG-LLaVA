# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
)
from xtuner.dataset.collate_fns import mm_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.samplers import LengthGroupedSampler
# from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.evaluation import (
    MMEDataset,
    MultipleChoiceDataset,
    POPEDataset,
    HallusionDataset,
    TextVQADataset,
)
from bbox_llava.evaluation.gqa_llava_eval_dataset import GQADataset
from bbox_llava.evaluation.vqav2_llava_eval_dataset import VQAv2Dataset
from xtuner.dataset import ConcatDataset
from xtuner.engine.runner import TrainLoop, ValLoop, TestLoop
from mmengine.dataset import DefaultSampler
from object_llava.dataset.video_length_grouped import VideoLengthGroupedSampler
from object_llava.dataset.video_collate_fn import video_collate_fn
from object_llava.dataset.batch_sampler import VideoImageSeperateBatchSampler
from mg_llava.evaluation.video.video_qa_dataset import VIDEOQADataset
from mg_llava.evaluation.video.video_object_llava_proxy_eval_dataset import VideoObjectLLaVAProxyEvalDataset
from object_llava.module import ObjectLLaVAModel, MGLLaVADataset, OpenCLIPVisionTower

from mg_llava.evaluation.video.hooks import DatasetInfoHook

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = '/mnt/hwfile/llmeval/opencompass/checkpoints/llm/hf_hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5/'
visual_encoder_name_or_path = '/mnt/hwfile/llmeval/opencompass/checkpoints/llm/hf_hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/'
visual_encoder_aux_name = 'model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup'
visual_encoder_aux_path = '/mnt/petrelfs/share_data/zhaoxiangyu/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/snapshots/39918dfbdf69ccd2172e6510a430e92337ee23e1/'
# Specify the pretrained pth
# pretrained_pth = 'work_dirs/ram_owl_pretrain/iter_2181.pth'
pretrained_pth = 'work_dirs/video_vicuna7b_clip_L_14_336_pretrain/iter_24000.pth'

# Data
box_json_path = '/mnt/petrelfs/share_data/zhaoxiangyu/sft_llava_video_mixed_only_box.json'
data_root = '/mnt/petrelfs/share_data/huanghaian/llava_data/'
data_path = '/mnt/petrelfs/share_data/fangxinyu/xtuner_llava_video_sft/videochatgpt_llavaimage_tune_modify_shuffle_v2.json'
# data_path = '/mnt/petrelfs/zhaoxiangyu/code_new/xtuner/data/train_json/videochatgpt_llavaimage_tune_sampledMinor.json'
# data_path = 'data/video_chatgpt_7k.json'
offline_processed_text_folder = 'data/text_cache/video_llava_sft_all'
# offline_processed_text_folder = '/mnt/petrelfs/zhaoxiangyu/code_new/xtuner/data/train_cache/sft_cache_minor'
# offline_processed_text_folder = 'data/text_cache/video_llava_sft_pure_video7000'
# offline_processed_text_folder = None
image_folder = '/mnt/petrelfs/zhaoxiangyu/code_new/xtuner/data'
video_folder = '/mnt/petrelfs/zhaoxiangyu/code_new/xtuner/data'
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = int(2048 - (336 // 14) ** 2)
image_size_aux = 768
limit_num = 100
num_frames = 8
# Scheduler & Optimizer
batch_size = 16  # per_device
video_batch_size = 2
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 2000
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''
evaluation_images = 'https://llava-vl.github.io/static/images/view.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture']

pad_image_to_square = False
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right',
)

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True,
)

model = dict(
    type=ObjectLLaVAModel,
    freeze_llm=False,
    freeze_visual_encoder=True,
    pretrained_pth=pretrained_pth,
    tokenizer=tokenizer,
    template=prompt_template,
    image_processor=image_processor,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
    ),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained, pretrained_model_name_or_path=visual_encoder_name_or_path
    ),
    visual_encoder_aux=dict(
        type=OpenCLIPVisionTower,
        vision_tower=visual_encoder_aux_name,
        vision_tower_path=visual_encoder_aux_path,
        optimize_vision_tower_aux=False,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=MGLLaVADataset,
    offline_processed_text_folder=offline_processed_text_folder,
    box_json_path=box_json_path,
    data_path=data_path,
    image_folder=image_folder,
    video_folder=video_folder,
    include_video=True,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=pad_image_to_square,
    image_size_aux=image_size_aux,
    limit_num=limit_num,
)  # ........

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=llava_dataset,
    sampler=dict(
        type=VideoLengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    batch_sampler=dict(
        type=VideoImageSeperateBatchSampler, video_batch_size=video_batch_size, drop_last=True
    ),
    collate_fn=dict(type=video_collate_fn, extra_collate_keys=['gt_boxes', 'gt_labels', 'pixel_values_aux']),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16',
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, val_interval=save_steps)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    # dict(
    #     type=EvaluateChatHook,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     every_n_iters=evaluation_freq,
    #     evaluation_inputs=evaluation_inputs,
    #     evaluation_images=evaluation_images,
    #     system=SYSTEM,
    #     prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,  # can save disk memory mmengine >=0.10.3
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

# ==================== val and test cfg =======================
val_dataset = [
    dict(
        type=MMEDataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/MME_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/MME.tsv',
        image_folder='/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
]

test_dataset = [
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/MMBench_DEV_EN_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/MMBench_DEV_EN.tsv',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/MMBench_TEST_EN_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/MMBench_TEST_EN.tsv',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/SEEDBench_IMG_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/SEEDBench_IMG.tsv',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/ScienceQA_VAL_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/ScienceQA_VAL.tsv',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/ScienceQA_TEST_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/ScienceQA_TEST.tsv',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/MMMU_DEV_VAL_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/MMMU_DEV_VAL.tsv',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/AI2D_TEST_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/AI2D_TEST.tsv',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=TextVQADataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/llava_textvqa_val_v051_ocr_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/huanghaian/orig_llava_eval/textvqa/llava_textvqa_val_v051_ocr.jsonl',
        ann_file='/mnt/petrelfs/share_data/huanghaian/text_vqa/TextVQA_0.5.1_val.json',
        image_folder='/mnt/petrelfs/share_data/huanghaian/text_vqa/train_images',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=MMEDataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/MME_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/MME.tsv',
        image_folder='/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        # for_llava_prompt=True, # 开了后，perception 会掉
        pad_image_to_square=pad_image_to_square,
    ),
    # dict(
    #     type=HallusionDataset,
    #     proxy_eval_dataset=dict(
    #         type=ObjectLLaVAProxyEvalDataset,
    #         box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/HallusionBench_only_bbox.json',
    #         image_size_aux=image_size_aux,
    #         limit_num=limit_num,
    #     ),
    #     data_file='/mnt/petrelfs/share_data/huanghaian/LMUData/HallusionBench.tsv',
    #     prompt_template=PROMPT_TEMPLATE.vicuna,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     pad_image_to_square=pad_image_to_square,
    # ),
    # dict(
    #     type=POPEDataset,
    #     proxy_eval_dataset=dict(
    #         type=ObjectLLaVAProxyEvalDataset,
    #         box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/pope/coco_pope_adversarial_only_bbox.json',
    #         image_size_aux=image_size_aux,
    #         limit_num=limit_num,
    #     ),
    #     data_file=[
    #         '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_adversarial.json',
    #         '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_popular.json',
    #         '/mnt/petrelfs/share_data/linzhihao/dataset/POPE/coco_pope_random.json',
    #     ],
    #     coco_val_path='/mnt/petrelfs/share_data/linzhihao/dataset/coco/val2014/',
    #     prompt_template=PROMPT_TEMPLATE.vicuna,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     pad_image_to_square=pad_image_to_square,
    # ),
    dict(
        type=MultipleChoiceDataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/LUMDdata_box/MMStar_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='/mnt/petrelfs/share_data/zhaoxiangyu/datasets--Lin-Chen--MMStar/snapshots/mmstar/MMStar.tsv',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=GQADataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/gqa_llava_eval/gqa_box.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        question_file='/mnt/petrelfs/share_data/zhaoxiangyu/gqa_llava_eval/llava_gqa_testdev_balanced.jsonl',
        answer_file='llava_gqa_testdev_balanced_merge.jsonl',
        prediction_file='testdev_balanced_predictions.json',
        test_question_file='/mnt/petrelfs/share_data/zhaoxiangyu/gqa_llava_eval/testdev_balanced_questions.json',
        image_folder='/mnt/petrelfs/share_data/basemodel/dataset/multimodality/gqa/images',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    dict(
        type=VQAv2Dataset,
        proxy_eval_dataset=dict(
            type=ObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/vqav2_llava_eval/vqav2_box.json',
            image_size_aux=image_size_aux,
        ),
        question_file='/mnt/petrelfs/share_data/zhaoxiangyu/vqav2_llava_eval/llava_vqav2_mscoco_test-dev2015.jsonl',
        answer_file='llava_vqav2_testdev_balanced_merge.jsonl',
        test_file='/mnt/petrelfs/share_data/zhaoxiangyu/vqav2_llava_eval/llava_vqav2_mscoco_test2015.jsonl',
        prediction_file='vqav2_testdev_balanced_predictions.json',
        image_folder='/mnt/petrelfs/share_data/zhaoxiangyu/vqav2_test2015',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
]

video_test_dataset = [
    dict(
        type=VIDEOQADataset,
        proxy_eval_dataset=dict(
            type=VideoObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/MSVD_frame8_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
            num_frames=8,
        ),
        gt_question_file='/mnt/petrelfs/share_data/fangxinyu/video_eval/video-llava-GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_q.json',
        # gt_question_file='debug/msvd_q_debug.json',
        gt_answer_file='/mnt/petrelfs/share_data/fangxinyu/video_eval/video-llava-GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_a.json',
        pred_file='MSVD_qa.json',
        video_folder='/mnt/petrelfs/share_data/fangxinyu/video_eval/video-llava-GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/videos',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=pad_image_to_square,
    ),
    # dict(
    #     type=VIDEOQADataset,
    #     proxy_eval_dataset=dict(
    #         type=VideoObjectLLaVAProxyEvalDataset,
    #         box_json_path='/mnt/petrelfs/share_data/zhaoxiangyu/MSRVTT_frame8_only_bbox.json',
    #         image_size_aux=image_size_aux,
    #         limit_num=limit_num,
    #         num_frames=8,
    #     ),
    #     gt_question_file='/mnt/petrelfs/share_data/fangxinyu/video_eval/video-llava-GPT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA/test_q.json',
    #     # gt_question_file='debug/msvd_q_debug.json',
    #     gt_answer_file='/mnt/petrelfs/share_data/fangxinyu/video_eval/video-llava-GPT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA/test_a.json',
    #     pred_file='MSRVTT_qa.json',
    #     video_folder='/mnt/petrelfs/share_data/fangxinyu/video_eval/video-llava-GPT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA/videos/all',
    #     prompt_template=PROMPT_TEMPLATE.vicuna,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     pad_image_to_square=pad_image_to_square,
    # )
]

# TODO: We are not currently using val_evaluator
# Don't support num_workers > 0
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=val_dataset),
    collate_fn=dict(
        type=mm_collate_fn, extra_collate_keys=['gt_boxes', 'gt_labels', 'img_id', 'pixel_values_aux']
    ),
)
val_evaluator = dict()
val_cfg = dict(type=ValLoop)

# TODO: We are not currently using test_evaluator
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=0,
#     drop_last=False,
#     sampler=dict(type=DefaultSampler, shuffle=False),
#     dataset=dict(type=ConcatDataset, datasets=test_dataset),
#     collate_fn=dict(
#         type=mm_collate_fn, extra_collate_keys=['gt_boxes', 'gt_labels', 'img_id', 'pixel_values_aux']
#     ),
# )

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=video_test_dataset),
    collate_fn=dict(
        type=video_collate_fn,
        extra_collate_keys=['gt_boxes', 'gt_labels', 'img_id', 'pixel_values_aux', 'question', 'answer'],
    ),
)

test_evaluator = val_evaluator
test_cfg = dict(type=TestLoop, select_metric='first')

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
from xtuner.dataset.evaluation import MMEDataset

from xtuner.dataset import ConcatDataset
from xtuner.engine.runner import TrainLoop, ValLoop, TestLoop
from mmengine.dataset import DefaultSampler
from mg_llava.dataset import VideoLengthGroupedSampler, video_collate_fn, VideoImageSeperateBatchSampler
from mg_llava.evaluation import (
    VIDEOQADataset,
    VideoObjectLLaVAProxyEvalDataset,
    DatasetInfoHook,
    MGLLaVAProxyEvalDataset
)
from mg_llava.module import MultiFuseObjectLLaVAModel, MGLLaVADataset, OpenCLIPVisionTower

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = 'PATH_TO_LLM'
visual_encoder_name_or_path = 'PATH_TO_CLIP-ViT_MODEL'
visual_encoder_aux_name = 'model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup'
visual_encoder_aux_path = 'PATH_TO_CLIP-ConvNext-MODEL'
# Specify the pretrained pth
pretrained_pth = 'work_dirs/fuse_video_vicuna7b_clip_L_14_336_pretrain/iter_24000.pth'

# Data
box_json_path = '/mnt/hwfile/mm_dev/zhaoxiangyu/sft_llava_video_mixed_only_box.json'
data_path = '/mnt/petrelfs/share_data/fangxinyu/xtuner_llava_video_sft/videochatgpt_llavaimage_tune_modify_shuffle_v2.json'
offline_processed_text_folder = None # 'PATH_TO_OFFLINE_FOLDER'
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

pad_image_to_square = True
video_pad_image_to_square = False
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
    type=MultiFuseObjectLLaVAModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    pretrained_pth=pretrained_pth,
    tokenizer=tokenizer,
    template=prompt_template,
    image_processor=image_processor,
    evaluate_max_tokens=50,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        ),
    ),
    llm_lora=dict(
        type=LoraConfig, r=512, lora_alpha=256, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'
    ),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained, pretrained_model_name_or_path=visual_encoder_name_or_path
    ),
    visual_encoder_aux=dict(
        type=OpenCLIPVisionTower,
        vision_tower=visual_encoder_aux_name,
        vision_tower_path=visual_encoder_aux_path,
        optimize_vision_tower_aux=False,
        use_last_feat=True,
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
    video_pad_image_to_square=video_pad_image_to_square,
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
            type=MGLLaVAProxyEvalDataset,
            box_json_path='PATH_TO_MME_BBOX_JSON',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
        ),
        data_file='PATH_TO_MME_TSV',
        image_folder='/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release',
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
            box_json_path='/mnt/hwfile/mm_dev/zhaoxiangyu/MSVD_frame8_only_bbox.json',
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
        pad_image_to_square=video_pad_image_to_square,
    ),
    dict(
        type=VIDEOQADataset,
        proxy_eval_dataset=dict(
            type=VideoObjectLLaVAProxyEvalDataset,
            box_json_path='/mnt/hwfile/mm_dev/zhaoxiangyu/MSRVTT_frame8_only_bbox.json',
            image_size_aux=image_size_aux,
            limit_num=limit_num,
            num_frames=8,
        ),
        gt_question_file='/mnt/petrelfs/share_data/fangxinyu/video_eval/video-llava-GPT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA/test_q.json',
        # gt_question_file='debug/msvd_q_debug.json',
        gt_answer_file='/mnt/petrelfs/share_data/fangxinyu/video_eval/video-llava-GPT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA/test_a.json',
        pred_file='MSRVTT_qa.json',
        video_folder='/mnt/petrelfs/share_data/fangxinyu/video_eval/video-llava-GPT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA/videos/all',
        prompt_template=PROMPT_TEMPLATE.vicuna,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=video_pad_image_to_square,
    )
]

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

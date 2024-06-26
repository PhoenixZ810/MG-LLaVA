# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel)

from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import mm_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.model import LLaVAModel
from peft import LoraConfig
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.dataset.evaluation import MMEDataset, MultipleChoiceDataset, POPEDataset, \
    HallusionDataset, TextVQADataset, GQADataset, VQAv2Dataset, ChartQADataset
from xtuner.dataset import ConcatDataset
from xtuner.engine.runner import TrainLoop, ValLoop, TestLoop
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = '/mnt/petrelfs/share_data/fanqi/meta-llama/Meta-Llama-3-8B-Instruct'
visual_encoder_name_or_path = 'model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
pretrained_pth = 'work_dirs/llava_llama3_8b_chat_clip_e1_gpu8_pretrain/iter_2181.pth'

# Data
data_root = '/mnt/petrelfs/share_data/huanghaian/llava_data/'
data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
image_folder = data_root + 'llava_images'
prompt_template = PROMPT_TEMPLATE.llama3_chat
max_length = int(2048 - (336 // 14) ** 2)

# Scheduler & Optimizer
batch_size = 8  # per_device
accumulative_counts = 2
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 1000
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 1000
SYSTEM = ''
evaluation_images = 'https://llava-vl.github.io/static/images/view.jpg'
evaluation_inputs = ['Please describe this picture']

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

model = dict(
    type=LLaVAModel,
    tokenizer=tokenizer,
    template=prompt_template,
    image_processor=image_processor,
    freeze_llm=False,
    freeze_visual_encoder=True,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        # to speed inference
        # attn_implementation='sdpa',
        trust_remote_code=True),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path),
    visual_encoder_lora=dict(
        type=LoraConfig, r=64, lora_alpha=16, lora_dropout=0.05, bias='none')
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder='/mnt/petrelfs/huanghaian/code/xtuner/llama3_8b_finetune',
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=llava_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=mm_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, val_interval=save_steps)
#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        image_processor=image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template)
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
        max_keep_ckpts=save_total_limit),
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
        type=GQADataset,
        data_file='gqa/llava_gqa_testdev_balanced.jsonl',
        ann_file='gqa/testdev_balanced_questions.json',
        image_folder='gqa/images',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
]

test_dataset = [
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MMBench_DEV_EN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/SEEDBench_IMG.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/ScienceQA_VAL.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/ScienceQA_TEST.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MMMU_DEV_VAL.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/AI2D_TEST.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=TextVQADataset,
        data_file='textvqa/llava_textvqa_val_v051_ocr.jsonl',
        ann_file='text_vqa/TextVQA_0.5.1_val.json',
        image_folder='text_vqa/train_images',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MMEDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MME.tsv',
        image_folder='/mnt/petrelfs/share_data/duanhaodong/data/mme/MME_Benchmark_release',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,

        pad_image_to_square=True),
    dict(
        type=HallusionDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/HallusionBench.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=POPEDataset,
        data_file=[
            'POPE/coco_pope_adversarial.json',
            'POPE/coco_pope_popular.json',
            'POPE/coco_pope_random.json'
        ],
        coco_val_path='coco/val2014/',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=GQADataset,
        data_file='gqa/llava_gqa_testdev_balanced.jsonl',
        ann_file='gqa/testdev_balanced_questions.json',
        image_folder='gqa/images',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='PATH_TO_MMSTAR_TSV',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MMBench_DEV_CN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/CCBench.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MMBench_TEST_CN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    dict(
        type=MultipleChoiceDataset,
        data_file='/mnt/petrelfs/huanghaian/code/xtuner/LMUData/MMBench_TEST_EN.tsv',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True),
    # dict(
    #     type=VQAv2Dataset,
    #     data_file='vqa/llava_vqav2_mscoco_test-dev2015.jsonl',
    #     test_file='vqa/llava_vqav2_mscoco_test2015.jsonl',
    #     image_folder='vqa/vqav2_test2015',
    #     prompt_template=PROMPT_TEMPLATE.vicuna,
    #     tokenizer=tokenizer,
    #     image_processor=image_processor,
    #     pad_image_to_square=True),
    dict(
        type=ChartQADataset,
        data_file=['LMUData/ChartQA/ChartQA Dataset/test/test_human.json',
                   'LMUData/ChartQA/ChartQA Dataset/test/test_augmented.json'],
        image_folder='LMUData/ChartQA/ChartQA Dataset/test/png',
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        image_processor=image_processor,
        pad_image_to_square=True
    )
]

# TODO: We are not currently using val_evaluator
# Don't support num_workers > 0
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=val_dataset),
    collate_fn=dict(type=mm_collate_fn, extra_collate_keys=['img_id']))
val_evaluator = dict()
val_cfg = dict(type=ValLoop)

# TODO: We are not currently using test_evaluator
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=test_dataset),
    collate_fn=dict(type=mm_collate_fn, extra_collate_keys=['img_id'])
)

test_evaluator = val_evaluator
test_cfg = dict(type=TestLoop, select_metric='first')
JOB_NAME=$1
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

# srun -p s1_mm_dev \
#     --job-name=${JOB_NAME} \
#     --quotatype=auto \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     python xtuner/tools/test.py \
#     fuse_object_llava/config/llamavid/llvid_vicuna7b_clip_L_14_336_sft_padding.py \
#     --checkpoint work_dirs/llvid_vicuna7b_clip_L_14_336_sft_padding/iter_10395.pth \
#     --launcher="slurm" \

# srun -p llmit \
#     --job-name=${JOB_NAME} \
#     --quotatype=auto \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     python xtuner/tools/test.py \
#     fuse_object_llava/config/vicuna/fuse_vicuna13b_clip_L_14_336_sft_padding.py \
#     --checkpoint work_dirs/fuse_vicuna13b_clip_L_14_336_sft_padding/iter_21000.pth \
#     --launcher="slurm"

# srun -p s1_mm_dev \
#     --job-name=${JOB_NAME} \
#     --quotatype=auto \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     python fuse_object_llava/module/test_34B/special_test.py \
#     fuse_object_llava/config/vicuna/fuse_vicuna13b_clip_L_14_336_sft_padding.py \
#     --work-dir work_dirs/fuse_vicuna13b_clip_L_14_336_sft_padding \
#     --launcher="slurm" \
#     --deepspeed deepspeed_zero3_debug

# srun -p llmit \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     python xtuner/tools/test.py \
#     object_llava/config/vicuna7b_clip_L_14_336_sft_padding.py \
#     --checkpoint work_dirs/vicuna7b_clip_L_14_336_sft_padding/iter_5000_w_initial.pth \
#     --launcher="slurm" \

# srun -p llmit \
#     --job-name=${JOB_NAME} \
#     --quotatype=auto \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     python fuse_object_llava/module/test_34B/special_test.py \
#     fuse_object_llava/config/Yi-34B/fuse_more_34B_test.py \
#     --work-dir work_dirs/fuse_more_yi34B_clip_L_14_336_sft_padding \
#     --launcher="slurm" \
#     --deepspeed deepspeed_zero3_debug

srun -p llmit \
    --job-name=${JOB_NAME} \
    --quotatype=auto \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python xtuner/tools/test.py \
    fuse_object_llava/config/vicuna/fuse_vicuna7b_clip_L_14_336_sft_padding.py \
    --checkpoint work_dirs/fuse_more_vicuna7b_clip_L_14_336_sft_padding/iter_10805.pth \
    --work-dir debug/appendjson \
    --launcher="slurm" \

# srun -p llmit \
#     --job-name=${JOB_NAME} \
#     --quotatype=auto \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     python xtuner/tools/test.py \
#     object_llava/config/llava_v15_7b/llava_v15_7b_finetune.py \
#     --checkpoint work_dirs/llava_v15_7b_finetune/iter_5198.pth \
#     --work-dir debug/appendjson \
#     --launcher="slurm" \


# srun -p llmit \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:1 \
#     python xtuner/tools/test.py \
#     object_llava/config/llava_v15_phi3/llava_v15_phi3_finetune.py \
#     --checkpoint work_dirs/llava_v15_phi3_finetune/iter_5198.pth \
#     --work-dir debug/appendjson \
#     --launcher="slurm" \

# srun -p llmit \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:1 \
#     python xtuner/tools/test.py \
#     fuse_object_llava/config/phi3/fuse_phi3_clip_L_14_336_sft_padding.py \
#     --checkpoint work_dirs/fuse_phi3_clip_L_14_336_sft_padding/iter_5198.pth \
#     --work-dir debug/appendjson \
#     --launcher="slurm" \
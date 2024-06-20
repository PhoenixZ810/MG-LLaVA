JOB_NAME=$1
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}
export MASTER_PORT=12345
srun -p llmit \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -x SH-IDC1-10-140-1-142,SH-IDC1-10-140-0-152 \
    python xtuner/tools/train.py \
    fuse_object_llava/config/llamavid/llvid_vicuna7b_clip_L_14_336_pretrain_padding.py \
    --seed 2024 \
    --launcher="slurm" \
    --deepspeed deepspeed_zero2


# srun -p s1_mm_dev \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     python xtuner/tools/train.py \
#     xtuner/configs/llava/phi2_2_7b_clip_large_p14_336/llava_phi2_2_7b_clip_L_p14_336_e1_gpu8_finetune.py \
#     --launcher="slurm" \
#     --deepspeed deepspeed_zero2

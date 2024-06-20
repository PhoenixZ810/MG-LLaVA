set -e

JOB_NAME=$1
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}
# export MASTER_PORT=12345
# srun -p llmit \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --quotatype=auto \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     -x SH-IDC1-10-140-0-145,SH-IDC1-10-140-0-143,SH-IDC1-10-140-0-142 \
#     python xtuner/tools/train.py \
#     fuse_object_llava/config/Yi-34B/fuse_more_yi34B_clip_L_14_336_pretrain_padding.py \
#     --seed 2024 \
#     --launcher="slurm" \
#     --deepspeed deepspeed_zero3

srun -p llmit \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -x SH-IDC1-10-140-1-87 \
    python xtuner/tools/train.py \
    fuse_object_llava/config/Yi-34B/fuse_more_yi34B_clip_L_14_336_sft_padding.py \
    --seed 2024 \
    --launcher="slurm" \
    --deepspeed deepspeed_zero3

srun -p llmit \
    --job-name=${JOB_NAME} \
    --quotatype=auto \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python xtuner/tools/test.py \
    fuse_object_llava/config/Yi-34B/fuse_more_yi34B_clip_L_14_336_sft_padding.py \
    --checkpoint work_dirs/fuse_more_yi34B_clip_L_14_336_sft_padding/iter_5198.pth \
    --launcher="slurm" \
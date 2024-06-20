set -e

JOB_NAME=$1
GPUS=${GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

srun -p llmit \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python xtuner/tools/train.py \
    mg_llava/config/llava_video/llava_video_vicuna7b_clip_L_14_336_lora.py \
    --seed 2024 \
    --launcher="slurm" \
    --deepspeed deepspeed_zero2

srun -p llmit \
    --job-name=${JOB_NAME} \
    --quotatype=auto \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python xtuner/tools/test.py \
    mg_llava/config/llava_video/llava_video_vicuna7b_clip_L_14_336_lora.py \
    --checkpoint work_dirs/llava_video_vicuna7b_clip_L_14_336_lora/iter_5724.pth \
    --launcher="slurm" \
    # --resume work_dirs/objectllava_phi2_2_7b_sigclip_L_p14_384_e1_gpu8_finetune/iter_5000.pth \
    # --work-dir work_dirs/object_llava_finetune_limit20 \
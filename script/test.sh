JOB_NAME=$1
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

srun -p PARTITION_NAME \
    --job-name=${JOB_NAME} \
    --quotatype=auto \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python xtuner/tools/test.py \
    PATH_TO_CONFIG \
    --checkpoint PATH_TO_PTH \
    --launcher="slurm" \

# Evaluation for Yi-1.5-34B
# srun -p llmit \
#     --job-name=${JOB_NAME} \
#     --quotatype=auto \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks=${GPUS} \
#     --ntasks-per-node=${GPUS_PER_NODE} \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --kill-on-bad-exit=1 \
#     python fuse_object_llava/module/test_34B/special_test.py \
#     mg_llava/config/Yi-34B/fuse_more_34B_test.py \
#     --work-dir work_dirs/fuse_more_yi34B_clip_L_14_336_sft_padding \
#     --launcher="slurm" \
#     --deepspeed deepspeed_zero3_debug
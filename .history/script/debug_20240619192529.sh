# export MASTER_PORT=12345
export CUDA_LAUNCH_BLOCKING=1
srun -p llmit \
    --gres=gpu:1 \
    python \
    -m debugpy --wait-for-client --listen 0.0.0:5678 \
    xtuner/tools/train.py \
    object_llava/config/box2/box2_vicuna7b_clip_L_14_336_pretrain_padding.py \
    --launcher="slurm" \
    # xtuner/tools/train.py \
    # mg_llava/config/Yi-34B/fuse_more_yi34B_clip_L_14_336_pretrain_padding.py \
    # --launcher="slurm" \

    # script/unit_test.py
        # --quotatype=auto \


    # xtuner/tools/test.py \
    # object_llava/config/llava_v15_7b/llava_v15_official.py \
    # --checkpoint work_dirs/video_vicuna7b_clip_L_14_336_sft/iter_11448.pth \



    # debug/openclip.py
    # xtuner/tools/train.py \
    # object_llava/config/video_vicuna7b_clip_L_14_336_sft.py \
    # --gres=gpu:1 \
    # --checkpoint work_dirs/vicuna7b_clip_L_14_336_sft/iter_5000.pth
        # --quotatype=auto \
    # debug/debug_dataloader.py
    # --gres=gpu:1 \
    #    --cpus-per-task=64 \
    # script/unit_test.py

    # object_llava/config/objectllava_phi2_2_7b_sigclip_L_p14_384_e1_gpu8_finetune_limit_multi.py


    # xtuner/tools/train.py \
    # object_llava/config/objectllava_phi2_2_7b_sigclip_L_p14_384_e1_gpu8_finetune_limit_multi.py
    # object_llava/config/objectllava_phi2_2_7b_clip_L_p14_336_e1_gpu8_pretrain.py
    # xtuner/tools/test.py \
    # bbox_llava/configs/ram_owl_finetune.py \
    # --checkpoint work_dirs/ram_owl_finetune/iter_5000_1.pth
    # --cpus-per-task=128 \
    # script/unit_test.py


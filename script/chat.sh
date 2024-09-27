srun -p mllm_1 \
    --gres=gpu:1 \
    python mg_llava/module/chat.py \
    'PATH TO MG-LLaVA-Vicuna-7B MODEL' \
    --llm_name_or_path 'PATH TO Vicuna1.5-7B LLM' \
    --visual_encoder_clip 'PATH TO CLIP MODEL' \
    --visual_encoder_convnext 'PATH TO ConvNext MODEL' \
    --ram_model 'PATH TO RAM MODEL' \
    --owl_vit_model 'PATH TO OWL-VIT-2 MODEL' \
    --prompt-template 'vicuna' \
    --image examples/example.jpg

# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
    GenerationConfig,
)
from transformers.generation.streamers import TextStreamer

from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import guess_load_checkpoint
from mmengine.model import is_model_wrapper
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE, SYSTEM_TEMPLATE
from xtuner.registry import BUILDER
from mmengine.config import Config
from mg_llava.module import MultiFuseObjectLLaVAModel, OpenCLIPVisionTower

TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument('model_name_or_path', help='Hugging Face model name or path')
    parser.add_argument('--llm_name_or_path', help='Hugging Face model name or path')
    parser.add_argument('--visual_encoder_clip', help='Hugging Face model name or path')
    parser.add_argument('--visual_encoder_convnext', help='Hugging Face model name or path')
    parser.add_argument('--ram_model', help='model name or path for ram')
    parser.add_argument('--owl_vit_model', help='model name or path for owl_vit')
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument('--adapter', default=None, help='adapter name or path')
    adapter_group.add_argument('--llava', default=None, help='llava name or path')
    parser.add_argument('--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument('--image', default=None, help='image')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under ' 'a specific `dtype`.',
    )
    parser.add_argument(
        '--prompt-template', choices=PROMPT_TEMPLATE.keys(), default=None, help='Specify a prompt template'
    )
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument('--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template', choices=SYSTEM_TEMPLATE.keys(), default=None, help='Specify a system template'
    )
    parser.add_argument('--bits', type=int, choices=[4, 8, None], default=None, help='LLM bits')
    parser.add_argument('--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins', nargs='+', choices=['calculate', 'solve', 'search'], help='Specify plugins to use'
    )
    parser.add_argument('--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument('--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument('--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).',
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to ' 'keep for top-k-filtering.',
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.',
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.',
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible text generation')
    args = parser.parse_args()
    return args


def get_input():
    """Helper function for getting input from users."""
    sentinel = ''  # ends when this string is seen
    result = None
    while result is None:
        print(('\ndouble enter to end input (EXIT: exit chat, ' 'RESET: reset history) >>> '), end='')
        try:
            result = '\n'.join(iter(input, sentinel))
        except UnicodeDecodeError:
            print('Invalid characters detected. Please enter again.')
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # build llm
    prompt_template = PROMPT_TEMPLATE[args.prompt_template]

    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=args.llm_name_or_path,
        trust_remote_code=True,
        padding_side='right',
    )

    image_processor = dict(
        type=CLIPImageProcessor.from_pretrained,
        pretrained_model_name_or_path=args.visual_encoder_clip,
        trust_remote_code=True,
    )
    model_dict = dict(
        type=MultiFuseObjectLLaVAModel,
        tokenizer=tokenizer,
        template=prompt_template,
        image_processor=image_processor,
        llm=dict(
            type=AutoModelForCausalLM.from_pretrained,
            pretrained_model_name_or_path=args.llm_name_or_path,
            trust_remote_code=True,
        ),
        visual_encoder=dict(
            type=CLIPVisionModel.from_pretrained,
            pretrained_model_name_or_path=args.visual_encoder_clip,
        ),
        visual_encoder_aux=dict(
            type=OpenCLIPVisionTower,
            vision_tower='model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup',
            vision_tower_path=args.visual_encoder_convnext,
            optimize_vision_tower_aux=False,
            use_last_feat=True,
        ),
    )
    model = Config(model_dict)
    model = BUILDER.build(model).to(dtype=torch.bfloat16).cuda().eval()
    state_dict = guess_load_checkpoint(args.model_name_or_path)
    if is_model_wrapper(model):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    tokenizer = model.tokenizer
    image_processor = model.image_processor

    stop_words = args.stop_words
    sep = ''
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
        sep = template.get('SEP', '')
    stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=stop_words)

    if args.no_streamer:
        streamer = None
    else:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        # streamer = sys.stdout

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ),
    )

    n_turn = 0
    inputs = ''

    if args.image is not None:
        model.init_box_generator(args.ram_model, args.owl_vit_model)
        box_data = model.box_generator(args.image)
        processed_dict = model.chat_preprocess_multi_modal(args.image, box_data)

    print('Log: Start chatting!')
    while True:
        text = get_input()
        while text.strip() == 'RESET':
            print('Log: History responses have been removed!')
            n_turn = 0
            inputs = ''
            text = get_input()
        if text.strip() == 'EXIT':
            print('Log: Exit!')
            exit(0)

        if args.image is not None and n_turn == 0:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if args.prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[args.prompt_template]
            if 'SYSTEM' in template and n_turn == 0:
                system_text = None
                if args.system_template is not None:
                    system_text = SYSTEM_TEMPLATE[args.system_template].format(
                        round=n_turn + 1, bot_name=args.bot_name
                    )
                elif args.system is not None:
                    system_text = args.system
                if system_text is not None:
                    prompt_text += template['SYSTEM'].format(
                        system=system_text, round=n_turn + 1, bot_name=args.bot_name
                    )
            prompt_text += template['INSTRUCTION'].format(
                input=text, round=n_turn + 1, bot_name=args.bot_name
            )
        else:
            prompt_text = text

        inputs += prompt_text

        if args.image is None:
            inputs, generate_output = model.chat(
                inputs, None, gen_config=gen_config, streamer=streamer, stop_criteria=stop_criteria
            )
            # inputs = tokenizer.decode(generate_output[0], skip_special_tokens=True).strip()
        else:
            inputs, generate_output = model.chat(
                inputs,
                args.image,
                processed_dict,
                generation_cfg=gen_config,
                streamer=streamer,
                stop_criteria=stop_criteria,
            )
            # inputs += tokenizer.decode(generate_output[0], skip_special_tokens=True).strip()

        n_turn += 1
        inputs += sep
        if len(generate_output[0]) >= args.max_new_tokens:
            print(
                'Remove the memory of history responses, since '
                f'it exceeds the length limitation {args.max_new_tokens}.'
            )
            n_turn = 0
            inputs = ''


if __name__ == '__main__':
    main()

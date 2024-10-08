<div align="center">
  <img src="images/MG.png" alt="Your Image" width="70px" style="float: left; margin-right: 1px;"/>
</div>

# MG-LLaVA: Towards Multi-Granularity Visual Instruction Tuning

 Xiangyu Zhao, [Xiangtai Li](https://lxtgh.github.io), [Haodong Duan](https://kennymckormick.github.io/), [Haian Huang](https://github.com/hhaAndroid), [Yining Li](https://scholar.google.com.hk/citations?user=y_cp1sUAAAAJ&hl=en), [Kai Chen](https://scholar.google.com.hk/citations?user=eGD0b7IAAAAJ&hl=en), Hua Yang

  <p align="center">
    <a href='https://arxiv.org/abs/2406.17770'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://phoenixz810.github.io/MGLLaVA/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=webpack' alt='Project Page'>
    </a>
    <a href='https://www.youtube.com/watch?v=WI-65Jk0j50'>
      <img src='https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=YouTube' alt='Video'>
    </a>
  </p>
<br />
<div align="center">
  <img src="images/teaser.png" width="90%">
</div>

## 🎉 News
- **\[2024/09\]** MG-LLaVA inference code released! Please refer to [Inference](#inference) for more details.
- **\[2024/08\]** MG-LLaVA now supports the evalution of [MMVet](https://github.com/yuweihao/MM-Vet), [LLaVA-Bench-in-the-wild](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_Bench.md), [MMVP](https://github.com/tsb0601/MMVP), and [MathVista](https://github.com/lupantech/MathVista) benchmarks!
- **\[2024/06\]** Our [paper](https://arxiv.org/abs/2406.17770), [code](https://github.com/PhoenixZ810/MG-LLaVA) and [weights](https://huggingface.co/PhoenixZ/MG-LLaVA) are all released.

## 📖 Introduction

we present MG-LLaVA, an innovative MLLM that enhances the model's visual processing capabilities by incorporating a multi-granularity vision flow, which includes low-resolution, high-resolution, and object-centric features. We propose the integration of an additional high-resolution visual encoder to capture fine-grained details, which are then fused with base visual features through a Conv-Gate fusion network. To further refine the model's object recognition abilities, we incorporate object-level features derived from bounding boxes identified by offline detectors. Being trained solely on publicly available multimodal data through instruction tuning,
MG-LLaVA demonstrates exceptional perception skills.

<div align="center">
  <img src="images/framework.png" width="80%">
</div>

## 🔥 Main Results
<!-- <div align="center">
  <img src="images/Main-results1.png" width="60%">
</div> -->


<div align="center">
  <img src="images/More_result.png" width="95%">
</div>


## 🛠️ Quick Start

### Installation

- It is recommended to build a Python-3.10 virtual environment using conda

  ```bash
  conda create --name mgllava-env python=3.10 -y
  conda activate mgllava-env
  ```

- Install XTuner from source

  ```shell
  git clone https://github.com/PhoenixZ810/MG-LLaVA.git
  cd MG-LLaVA
  pip install -e '.[all]'
  ```

### Data Preparation

Please refer to [dataset_prepare.md](dataset_prepare.md).

### Model Weights
Our checkpoints are available at [ModelZoo](https://huggingface.co/PhoenixZ/MG-LLaVA).

### Before Train
MG-LLaVA employed several LLMs ranged from 3.8B to 34B, including [Phi-3-3.8B](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), [Vicuna1.5-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5), [Vicuna1.5-13B](https://huggingface.co/lmsys/vicuna-13b-v1.5), [llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), and [Yi1.5-34B](https://huggingface.co/01-ai/Yi-1.5-34B-Chat). We employ [CLIP-Large-336](https://huggingface.co/openai/clip-vit-large-patch14-336) and [CLIP-ConvNext-320-d](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup) as vision encoders, you should download both the LLM and CLIP checkpoints before training.

The training process is similar to the original XTuner. Before training, you should check the [configs](mg_llava/config) and modify the following variables to your own settings. You can also modify the [configs](mg_llava/config) to train the model with your own settings.
  ```shell
  # Path of LLM and CLIP
  llm_name_or_path
  visual_encoder_name_or_path
  visual_encoder_aux_path
  prompt_template

  # Data
  data_path
  box_json_path
  image_folder
  offline_processed_text_folder(optional)

  # Training
  pretrained_pth(Fine-Tuning)
  ```
Before training, you can use the following command to preprocess the text data to speed up the training process. You can preprocess the text data by running the following command:

```shell
python xtuner/tools/process_untokenized_llava_data.py CONFIG --save-folder TEXT-PATH
```
and then set the `offline_processed_text_folder` in the config file to `TEXT-PATH`.

### Train & Evaluation
MG-LLaVA follows a two-stage training process, the entire training process takes approximately 23 hours when using the Vicuna1.5-7B model using 8×A100 GPUs. For example, to train the MG-LLaVA model with Vicuna1.5-7B, you can use the following command:


- **Entire Pipeline**: Pretraining + Fine-tuning + Evaluation

  ```shell
  bash script/train_vicuna7B.sh
  ```

If you want to train our model step by step, you can follow the instructions below:

- **Step 1**, start pretraining.
  ```shell
  bash script/train_pretrain.sh mg_llava/config/vicuna/fuse_vicuna7b_clip_L_14_336_pretrain_padding.py
  ```

- **Step 2**, start fine-tuning.

  ```shell
  bash script/train_sft.sh mg_llava/config/vicuna/fuse_vicuna7b_clip_L_14_336_sft_padding.py
  ```

  - `--deepspeed` means using [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 to optimize the training. XTuner comes with several integrated strategies including ZeRO-1, ZeRO-2, and ZeRO-3. If you wish to disable this feature, simply remove this argument.

  - For more examples, please see [finetune.md](./docs/en/user_guides/finetune.md).

- **Step 3**, evaluation. The evaluation metrics are specified in the sft configuration, including MMBench, SEED, SQA, AI2D, TextVQA, POPE, GQA, VQAv2, and additional ones. Please refer to [evaluation.md](evaluation.md).

  You can convert the saved PTH model (if using DeepSpeed, it will be a directory) to Hugging Face model, by

  ```shell
  xtuner convert pth_to_hf CONFIG_NAME_OR_PATH CHECKPOINT SAVE_PATH
  ```
<a id="inference"></a>
### Inference
Before inference, you need to download MG-LLaVA [checkpoints](https://huggingface.co/PhoenixZ/MG-LLaVA) and corresponding LLM model. In addition, [CLIP-Large-336](https://huggingface.co/openai/clip-vit-large-patch14-336), [CLIP-ConvNext-320-d](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup), [RAM](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth) and [OWL-VIT-2](https://huggingface.co/google/owlv2-large-patch14-ensemble) are also required.


The code for inference is available at [chat.py](mg_llava/module/chat.py). You can use the following command to run the inference code in [chat.sh](script/chat.sh) and chat with MG-LLaVA.

```
srun -p mllm_1 \
    --gres=gpu:1 \
    python mg_llava/module/chat.py \
    PATH TO MG-LLaVA-Vicuna-7B MODEL \
    --llm_name_or_path 'PATH TO Vicuna1.5-7B LLM' \
    --visual_encoder_clip 'PATH TO CLIP MODEL' \
    --visual_encoder_convnext 'PATH TO ConvNext MODEL' \
    --ram_model 'PATH TO RAM MODEL' \
    --owl_vit_model 'PATH TO OWL-VIT-2 MODEL' \
    --prompt-template 'vicuna' \
    --image examples/example.jpg
```
## Citation
If you find MG-LLaVA useful, please cite using this BibTeX:
```bibtex
@article{zhao2024mg,
  title={MG-LLaVA: Towards Multi-Granularity Visual Instruction Tuning},
  author={Zhao, Xiangyu and Li, Xiangtai and Duan, Haodong and Huang, Haian and Li, Yining and Chen, Kai and Yang, Hua},
  journal={arXiv preprint arXiv:2406.17770},
  year={2024}
}
```
## Acknowledgement
- [Xtuner](https://github.com/InternLM/xtuner): the codebase we built upon.
- [LLaVA](https://github.com/haotian-liu/LLaVA): the base model structure.



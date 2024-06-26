<div align="center">
  <img src="images/MG.png" alt="Your Image" width="70px" style="float: left; margin-right: 1px;"/>
</div>

# MG-LLaVA: Towards Multi-Granularity Visual Instruction Tuning

 Xiangyu Zhao, [Xiangtai Li](https://hliu.cc), [Haodong Duan](https://sivakm.github.io/), [Haian Huang](https://gregmeyer.info), [Yining Li](https://scholar.google.com/citations?user=i7U4YogAAAAJ&hl=en), [Kai Chen](https://scholar.google.com/citations?user=_UJsz3AAAAAJ&hl=en), Hua Yang

  <p align="center">
    <a href='https://arxiv.org/abs/2406.17770'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://phoenixz810.github.io/MGLLaVA/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=webpack' alt='Project Page'>
    </a>
    <a href='https://github.com/PhoenixZ810/MG-LLaVA'>
      <img src='https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=YouTube' alt='Video'>
    </a>
  </p>
<br />
<div align="center">
  <img src="images/teaser.png" width="90%">
</div>

## 🎉 News

- **\[2024/06\]** Our [Paper](https://arxiv.org/abs/2406.17770) and code are released. The checkpoints will be released soon.

## 📖 Introduction

we present MG-LLaVA, an innovative MLLM that enhances the model's visual processing capabilities by incorporating a multi-granularity vision flow, which includes low-resolution, high-resolution, and object-centric features. We propose the integration of an additional high-resolution visual encoder to capture fine-grained details, which are then fused with base visual features through a Conv-Gate fusion network. To further refine the model's object recognition abilities, we incorporate object-level features derived from bounding boxes identified by offline detectors. Being trained solely on publicly available multimodal data through instruction tuning,
MG-LLaVA demonstrates exceptional perception skills.

<div align="center">
  <img src="images/framework.png" width="80%">
</div>

## 🔥 Main Results
<div align="center">
  <img src="images/Main-results1.png" width="60%">
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

### Before Train
MG-LLaVA employed several LLMs ranged from 3.8B to 34B, including Phi-3-3.8B, Vicuna1.5-7B, Vicuna1.5-13B, llama3-8B, and Yi1.5-34B. We employ CLIP-Large-336m and CLIP-ConvNext-320-d as vision encoders, you should download both the LLM and CLIP checkpoints before training.

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

### Train
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

- **Step 3**, evaluation. Before evaluation, you should modify the [test config]() and run the following command:
  ```shell
  bash script/test.sh
  ```
  You can convert the saved PTH model (if using DeepSpeed, it will be a directory) to Hugging Face model, by

  ```shell
  xtuner convert pth_to_hf CONFIG_NAME_OR_PATH CHECKPOINT SAVE_PATH
  ```

## Model Weights
Our checkpoints are available at [ModelZoo](https://huggingface.co/PhoenixZ/MG-LLaVA).



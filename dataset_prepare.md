# Dataset Preparation

## Pretraining

During the image-based training stage, our dataset comprises 558K image-caption pairs from LAION-CCSBU and 708k image-caption pairs from ALLaVA-4V-Caption dataset, culminating in a total of 1.26M image-caption pairs for pretraining.

## Fine-tuning

The datasets employed for instruction-tuning encompass 665K mixture dataset from LLaVA-Instruct, 692k instructions from ALLaVA-4V-Instruction dataset, and an additional 25k instructions derived from a combination of ShareGPT4V , DocVQA, DVQA and AI2D, with a total number of more than 1.3M image-text conversations.

## Download Images for Training
[LLaVA-1.5 pretrain images](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) -> ```data/LLaVA-Pretrain/images```

[ALLaVA-4V-LAION and ALLaVA-4V-Vison-FLAN images](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) -> ```data/allava_laion/images```, ```data/allava_vflan/images```


[COCO](http://images.cocodataset.org/zips/train2017.zip) -> ```data/coco/train2017```

[GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) -> ```data/gqa/images```

[OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) -> ```data/ocr_vqa/images```

[TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) -> ```data/textvqa/train_images```

[VG-Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [VG-Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) -> ```data/vg/VG_100K, data/vg/VG_100K_2```

The Web-Celebrity, Web-Landmark, WikiArt,  Share-TextVQA in [ShareGPT-4V](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax).

[AI2D](https://allenai.org/data/diagrams) -> ```data/ai2d/images```

[DocVQA](https://www.docvqa.org/datasets/docvqa) -> ```data/docvqa/images```

[DVQA](https://github.com/kushalkafle/DVQA_dataset) -> ```data/dvqa/images```

The complete structure is as follows:
```text
MG-LLAVA
├── data
│   ├── LLaVA-Pretrain
│   │   ├── images
│   ├── ai2d
|   |   ├── images
│   ├── allava_laion
|   |   ├── images
│   ├── allava_vflan
|   |   ├── images
│   ├── coco
|   |   ├── train2017
│   ├── docvqa
|   |   ├── images
│   ├── dvqa
|   |   ├── images
│   ├── gqa
|   |   ├── images
│   ├── ocr_vqa
|   |   ├── images
│   ├── share_textvqa
|   |   ├── images
│   ├── textvqa
|   |   ├── train_images
│   ├── vg
|   |   ├── VG_100K
|   |   ├── VG_100K_2
│   ├── web-celebrity
|   |   ├── images
│   ├── web-landmark
|   |   ├── images
│   ├── wikiart
|   |   ├── images
```

## Download Annotations Files
We employ RAM-Plus and OWL-ViT to generate bounding boxes for training and evaluation. Our [trainging annotation files](https://huggingface.co/PhoenixZ/MG-LLaVA/tree/main/train_json) and [bounding box annotation files](https://huggingface.co/PhoenixZ/MG-LLaVA/tree/main/bbox_json/train) are available in Hugging Face. Please download and modify ```data_path``` and ```box_json_path``` in your config file.

If you want to generate the bounding boxes by yourself, you can refer to the [image_offline_to_bbox.py](mg_llava/bbox_generation/image_offline_to_bbox.py) by downloading [RAM](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth), [OWL-ViT2](https://huggingface.co/google/owlv2-large-patch14-ensemble) and  modifying the ```data_file```, ```image_folder```, ```save_json_path``` ,then run the following command:
```shell
torchrun --nproc_per_node=8 mg_llava/bbox_generation/image_offline_to_bbox.py
```

## Download Data for Evaluation
Most of the evaluation benchmarks utilized in our paper can be found in [LLaVa](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

The bounding box annotation files for evaluation are available in [Hugging Face](https://huggingface.co/PhoenixZ/MG-LLaVA/tree/main/bbox_json/eval).
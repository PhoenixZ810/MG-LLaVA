<div align="center">
  <img src="images/MG.png" alt="Your Image" width="35px" style="float: left; margin-right: 10px;"/>
</div>

# MG-LLaVA: Towards Multi-Granularity Visual Instruction Tuning

 Xiangyu Zhao, [Xiangtai Li](https://hliu.cc), [Haodong Duan](https://sivakm.github.io/), [Haian Huang](https://gregmeyer.info), [Yining Li](https://scholar.google.com/citations?user=i7U4YogAAAAJ&hl=en), [Kai Chen](https://scholar.google.com/citations?user=_UJsz3AAAAAJ&hl=en), Hua Yang

<div align="center">
  <img src="images/teaser.png" width="90%">
</div>

## 🎉 News

- **\[2024/06\]** Paper and code are to be released!

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

## 🖊️ Citation
If you find this work useful, please consider citing:
```bibtex
@article{mgllava,
  title={Towards Semantic Equivalence of Tokenization in Multimodal LLM},
  author={Zhao, Xiangyu and Li, Xiangtai and Duan, Haodong and Huang, Haian and Li, Yining and Chen, Kai and Yang, Hua},
  journal={arXiv preprint},
  year={2024}
}
```



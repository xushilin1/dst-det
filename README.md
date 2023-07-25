<!-- # DST-Det -->

<br />
<p align="center">
  <h1 align="center">Open-Vocabulary Object Detection via
Dynamic Self-Training </h1>
  <p align="center">
    <!-- ICCV, 2023 -->
    <br />
    <a href=""><strong>Shilin Xu*</strong></a>
    ·
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li*</strong></a>
    ·
    <a href="http://zhangwenwei.cn/"><strong>Wenwei Zhang</strong></a>
    ·
    <a href=""><strong>Size Wu</strong></a>
    ·
    <a href="https://sites.google.com/view/guangliangcheng"><strong>Guangliang Cheng</strong></a>
    <br />
    <a><strong>Yunhai Tong</strong></a>
    .
    <a href="https://www.mmlab-ntu.com/person/ccloy/"><strong>Chen Change Loy</strong></a>
  </p>
<br />




## Abstract

This paper presents a novel method for open-vocabulary object detection (OVOD) that aims to detect objects \textit{beyond} the set of categories observed during training. 
Our approach proposes a dynamic self-training strategy that leverages the zero-shot classification capabilities of pre-trained vision-language models, such as CLIP, to classify proposals as novel classes directly. Unlike previous works that ignore novel classes during training and rely solely on the region proposal network (RPN) for novel object detection, our method selectively filters proposals based on specific design criteria. The resulting set of identified proposals serves as pseudo labels for novel classes during the training phase, enabling our self-training strategy to improve the recall and accuracy of novel classes in a self-training manner without requiring additional annotations or datasets. Empirical evaluations on the LVIS and COCO datasets demonstrate significant improvements over the baseline performance without incurring additional parameters or computational costs during inference. Notably, our method achieves a 1.7\% improvement over the previous F-VLM method on the LVIS validation set. Moreover, combined with offline pseudo label generation, our method improves the strong baselines over 10 \% mAP on COCO. 
![teaser](./assets/figs/teaser.png)



## Visualization Results
### COCO
<details open>
<summary>Demo</summary>

![vis_demo_1](assets/figs/coco_vis.png) 



## Citation
If you think DST-Det is helpful in your research, please consider referring DST-Det:
```bibtex
@article{xu2023dst-det,
  title={Open-Vocabulary Object Detection via
Dynamic Self-Training},
  author={Shilin Xu, Xiangtai Li, Wenwei Zhang, Size Wu, Guangliang Cheng, Yunhai Tong, Chen Change Loy},
  journal={arXiv pre-print},
  year={2023},
}
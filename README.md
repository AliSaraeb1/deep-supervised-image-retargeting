# Deep Supervised Image Retargeting

This repository contains a PyTorch implementation of the method from

> Yijing Mei, Xiaojie Guo, Di Sun, Gang Pan, and Jiawan Zhang,  
> **“Deep Supervised Image Retargeting”**, IEEE TCSVT, 2021.  
> [https://ieeexplore.ieee.org/document/9428129](https://ieeexplore.ieee.org/document/9428129)

Please see the notebook (training + evaluation) for the implementation code and details of implementation (see below for quick test):

- [`deep_supervised_image_retargeting.ipynb`](./deep_supervised_image_retargeting.ipynb)

For the purpose of quick evaluation and testing (~ 3 minutes), please see 

- [`Instructions for OSC testing.pdf`](./Instructions for OSC testing.pdf)

To avoid copyright issues, we don't provide the images that the model outputted, but we make the metric tables (confirming the success of the method: higher means better) that the model outputted available here

- [`mrgan_run.out`](./mrgan_run.out)

We make the pretrained checkpoints available for download and use here:

-  for the loss_mode="ours": [mrgan_tired_best.pth](https://www.dropbox.com/scl/fo/lh7vrzxtx689q4a2xw6s6/AELD_F9VpQ_QGqg_XqvgvBQ?rlkey=bj6un44lara1xhpry8cqnh6kr&st=6h28l8fr&dl=0) 
-  for the loss_mode="no_Lm_tv": [mrgan_no_Lm_tv_best3.pth](https://www.dropbox.com/scl/fo/lh7vrzxtx689q4a2xw6s6/AELD_F9VpQ_QGqg_XqvgvBQ?rlkey=bj6un44lara1xhpry8cqnh6kr&st=6h28l8fr&dl=0)

The TIReD dataset was made available by the authors at:  
- [https://github.com/TIReD2020/TIReD](https://github.com/TIReD2020/TIReD)


© 2025 Ali Saraeb. All rights reserved.  


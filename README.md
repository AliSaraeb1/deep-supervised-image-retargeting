# Deep Supervised Image Retargeting

This repository contains a PyTorch implementation of the method from

> Yijing Mei, Xiaojie Guo, Di Sun, Gang Pan, and Jiawan Zhang,  
> **“Deep Supervised Image Retargeting”**, IEEE TCSVT, 2021.  
> [https://ieeexplore.ieee.org/document/9428129](https://ieeexplore.ieee.org/document/9428129)

VERY IMPORTANT: Please read the following short pdf for answers to asked questions and for the purpose of quick evaluation and testing (no training, use pretrained models ~ 3 minutes for testing) 

- [`Instructions_for_OSC_pretrained_testing_and_answers_to_required_questions.pdf`](./Instructions_for_OSC_pretrained_testing_and_answers_to_required_questions.pdf)

VERY IMPORTANT: Please see the notebook (training + evaluation) for the implementation code and details of implementation:

- [`deep_supervised_image_retargeting.ipynb`](./deep_supervised_image_retargeting.ipynb)

To avoid copyright issues, we don't provide the images that the model outputted, but we make the metric tables (confirming the success of the method: higher means better) that the model outputted available here

- [`mrgan_run.out`](./mrgan_run.out)

We make the pretrained checkpoints available for download and use here:

-  for the loss_mode="ours": [mrgan_tired_best.pth](https://www.dropbox.com/scl/fo/lh7vrzxtx689q4a2xw6s6/AELD_F9VpQ_QGqg_XqvgvBQ?rlkey=bj6un44lara1xhpry8cqnh6kr&st=6h28l8fr&dl=0) 
-  for the loss_mode="no_Lm_tv": [mrgan_no_Lm_tv_best3.pth](https://www.dropbox.com/scl/fo/lh7vrzxtx689q4a2xw6s6/AELD_F9VpQ_QGqg_XqvgvBQ?rlkey=bj6un44lara1xhpry8cqnh6kr&st=6h28l8fr&dl=0)

The TIReD dataset was made available by the authors at:  
- [https://github.com/TIReD2020/TIReD](https://github.com/TIReD2020/TIReD)


© 2025 Ali Saraeb. All rights reserved.  


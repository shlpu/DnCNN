# About

* This is a reimplementation of paper [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981) using Matlab Built-in deeplearning toolbox
* The [original version](https://github.com/cszn/DnCNN) uses Matconvnet


# Requirements

* Matlab R2018a only

# Usage

1. set up your configurations as in main*.m
2. use `Project.run_project(opts)` to run the project

# Notes
1. **About time:** Due to the implementation of Matlab BatchNorm layers, which doesn't support testing before training is finished, I have to make an trade-off (i.e., force finalizing training for each epoch, which is time-consuming) in order to test the performance(PSNR and SSIM) of each snapshot of the network, hence the total training time is about *10 hours for 50 epochs*, which is longer than the original ones(6 hours)

# Todos:
1. Fine-tuned network models
2. more experiments on different training options

---

Have fun fine-tuning :)
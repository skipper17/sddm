# ILVR + ADM

This is the implementation of [ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2108.02938) (ICCV 2021 Oral).

This repository is heavily based on [improved diffusion](https://github.com/openai/improved-diffusion) and [guided diffusion](https://github.com/openai/guided-diffusion).
We use [PyTorch-Resizer](https://github.com/assafshocher/PyTorch-Resizer) for resizing function.

## Overview

ILVR is a learning-free method for controlling the generation of unconditional DDPMs. ILVR refines each generation step with low-frequency component of purturbed reference image. Our method enables various tasks (image translation, paint-to-image, editing with scribbles) with only a single model trained on a target dataset. 

![image](https://user-images.githubusercontent.com/36615789/133278340-48050da2-192b-4851-87ab-ba090545886a.png)


## Download pre-trained models
Create a folder `models/` and download model checkpoints into it.
Here are the unconditional models trained on FFHQ and AFHQ-dog:

 * 256x256 FFHQ: [ffhq_10m.pt](https://drive.google.com/file/d/117Y6Z6-Hg6TMZVIXMmgYbpZy7QvTXign/view?usp=sharing)
 * 256x256 AFHQ-dog: [afhq_dog_4m.pt](https://drive.google.com/file/d/14OG_o3aa8Hxmfu36IIRyOgRwEP6ngLdo/view?usp=sharing)

These models have seen 10M and 4M images respectively.
You may also try with models from [guided diffusion](https://github.com/openai/guided-diffusion).


## ILVR Sampling
First, set PYTHONPATH variable to point to the root of the repository.

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Then, place your input image into a folder `ref_imgs/`.

Run the `ilvr_sample.py` script. Specify the folder where you want to save the output in `--save_dir`.

Here, we provide flags for sampling from above models.
Feel free to change `--down_N` and `--range_t` to adapt downsampling factor and conditioning range from the paper.

Refer to [improved diffusion](https://github.com/openai/improved-diffusion) for `--timestep_respacing` flag.

```
python scripts/ilvr_sample.py  --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 --model_path models/ffhq_10m.pt --base_samples ref_imgs/face --down_N 32 --range_t 20 --save_dir output
```

ILVR sampling is implemented in `p_sample_loop_progressive` of `guided-diffusion/gaussian_diffusion.py`


## Results

These are samples generated with N=8 and 16:

![a](gif/full_face8_small.gif)

![b](gif/full_face16_small.gif)

These are cat-to-dog samples generated with N=32:

![c](gif/full_cat2dog_small.gif)


## Note
This repo is re-implemention of our method on [guided diffusion](https://github.com/openai/guided-diffusion). Our initial implementation of the paper is based on [denoising-diffusion-pytorch](https://github.com/rosinality/denoising-diffusion-pytorch).

## Edit
mainly edit the gauss_diffusion/gauss_diffusion.py and the scripts/ae_sample.py

## Contribulation
* 提出了流形优化的角度统一了对diffusion model统计量的约束
* 首次将动态梯度结合的算法应用在多energy指导的diffusion model
* 提出了blockadain模块用于更好地做图形的统计量迁移， 有更高的自由度
## For Paper
* 证明部分
    * dynamic perspective (是个trivial的体力活)
        * 一个证明是保证能落在对应噪声的分布上
        * 多个ref梯度之间是动态merge的
    * manifold optimization (可行性已经完成证明, 只剩下trivial的体力活. 不需要修改block adain)
        * 提出的blockadain用流形优化的formulation写出来, 已完成, 是$S^{n-2}$的超球
    * 可以退化到ILVR和EGSDE, 并且给出了相应的稳定性的解释
        * 做一阶统计量迁移就是ILVR, 抛弃ILVR仅仅使用energy且放弃dynamic的合并方案就是EGSDE
        * EGSDE的专家系统理解, 如果能迁移证明要迁移证明
    * 考虑流形约束和梯度的串联效应

## TODO
* 实验部分 在21天内完成
    * 框架代码的实现
    * 框架代码的调参
    * 实验的设计和结果
        * 以下实验使用cat2dog 或者 male2female
        * 验证流形迁移梯度的重要性(数据实验)来说明这样的分割是科学的 自我对照
            * 计算流形迁移梯度的模长

        * 使用动态梯度合并和不使用的对比, 来说明这样的算法是有效的 自我对照

        * 不同block大小的对比, 梯度合并是否按block分割的对比 不同时间步的对比 自我对照

        * 以下实验使用多个数据集
        * 跟之前方法 score的方法 (主要是SDEDIT, ILVR, EGSDE, 还有gan based的方法) 实验的对比
            * 指标有FID, ssim, psnr, l2

* 理论部分 在10天内完成
    * 确定证明要采用的符号并熟悉相关语言, 需要考虑之前工作的连续性
    * 确定要证明的点和框架
    * 按照上一章节的内容补充证明框架

* 撰写论文 在一个月内完成初版

## 流程
* 给定ref image, 首先根据block统计量采样一个gauss noise 并映射到流形M_{T}上
* 之后循环 t: T ~ 1:
    * 获得Diffusion的梯度g_d, 分解为子流形切空间的分量g_{d1}, 和剩余的与之垂直的分量g_{d2}
    * 通过Frank Wolfe算法做流形内梯度的合并, 更新生成的目标
    * 映射回流形M_{t}
    * 使用流形迁移梯度g_{d2}和blockadain将其映射到流形M_{t-1}

## useful bash command
``` python
# 运行实验程序
python scripts/ae_sample.py  --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 --model_path models/afhq_dog_4m.pt --base_samples /home/sunsk/data/afhq_v2/test/cat  --range_t 40 --save_dir myoutput/tmp2neural

# 运行fid的测试
python -m pytorch_fid ~/data/afhq_v2/test/dog/ ~/Projects/mycond_adm/myoutput/test/ --device cuda:1
```
## 运行的stack
```
main()
p_sample_loop()
p_sample_loop_progressive() # 遍历t,  go one step and remap
p_sample()
    p_mean_variance() # 调用model, 得到均值和方差

    condition_mean() # 计算条件均值和方差

```
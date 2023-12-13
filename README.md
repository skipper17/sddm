# SDDM

This is the implementation of [SDDM: Score-Decomposed Diffusion Models on Manifolds for Unpaired Image-to-Image Translation](https://openreview.net/pdf?id=J4w91xRPBY) (ICML 2023 Poster).

This repository is heavily based on [ILVR](https://github.com/jychoi118/ilvr_adm), [improved diffusion](https://github.com/openai/improved-diffusion) and [guided diffusion](https://github.com/openai/guided-diffusion).
We use [PyTorch-Resizer](https://github.com/assafshocher/PyTorch-Resizer) for resizing function.

## Overview

SDDM derives manifolds to make the distributions of adjacent time steps separable and decompose the score function or energy guidance into an image “denoising” part and a content “refinement” part. To refine the image in the same noise level, we equalize the refinement parts of the score function and energy guidance, which permits multiobjective optimization on the manifold. We also leverage the block adaptive instance normalization module to construct manifolds with lower dimensions but still concentrated with the perturbed reference image. SDDM outperforms existing SBDM-based methods with much fewer diffusion steps on several I2I benchmarks.

<img width="1375" alt="image" src="https://github.com/skipper17/mycond_adm/assets/36984150/07b045d1-0fa8-4aec-9289-fd8e98a6392d">


## Download pre-trained models
Refer to [ILVR](https://github.com/jychoi118/ilvr_adm)
You may also try with models from [guided diffusion](https://github.com/openai/guided-diffusion).


## SDDM Sampling
First, set PYTHONPATH variable to point to the root of the repository.

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Then, place your input image into a folder `ref_imgs/`.

Run the `ae_sample.py` script. Specify the folder where you want to save the output in `--save_dir`.

Here, we provide flags for sampling from above models.

Refer to [improved diffusion](https://github.com/openai/improved-diffusion) for `--timestep_respacing` flag.

```
python scripts/ae_sample.py  --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma False --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 --model_path models/afhq.pt --base_samples ref_imgs --range_t 50 --area 16 --detail_merge False --save_dir output
```

SDDM sampling is implemented in `guided-diffusion/gaussian_diffusion.py` and `scripts/ae_sample.py`


## 流程
* 给定ref image, 首先根据block统计量采样一个gauss noise 并映射到流形M_{T}上
* 之后循环 t: T ~ 1:
    * 获得Diffusion的梯度g_d, 分解为子流形切空间的分量g_{d1}, 和剩余的与之垂直的分量g_{d2}
    * 通过Frank Wolfe算法做流形内梯度的合并, 更新生成的目标
    * 映射回流形M_{t}
    * 使用流形迁移梯度g_{d2}和blockadain将其映射到流形M_{t-1}

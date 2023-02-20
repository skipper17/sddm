"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from tool.utils import dict2namespace
import yaml

from guided_diffusion.gaussian_diffusion import StageVGG, block_adaIN, calc_mean_std, blockzation
from guided_diffusion.ddpm import Model
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils, models

# added
def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir="myoutput/")
    
    os.makedirs(os.path.expanduser(args.save_dir), exist_ok=True)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.diffusionmodel == 'DDPM':
        with open("guided_diffusion/ddpm.yml", "r") as f:
            config_ = yaml.safe_load(f)
        config = dict2namespace(config_)
        # config.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
        model = Model(config)
        model.to(dist_util.dev())
        model = th.nn.DataParallel(model)
        states = th.load(args.model_path)
        model.load_state_dict(states, strict=True)
        print("load success")
    else:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading energy_guidance...")
    stagevgg = StageVGG()

    stagevgg.to(dist_util.dev())
    # if args.use_fp16:
    #     stagevgg.convert_to_fp16()
    stagevgg.eval()
    # path = "/home/sunsk/Models/resnet50/resnet50-19c8e357.pth"
    # path = "/home/sunsk/Models/resnet50/resnet50-0676ba61.pth"
    cosmodel = models.resnet50(pretrained=True)
    # cosmodel.load_state_dict(th.load(path))
    cosmodel = th.nn.Sequential(*(list(cosmodel.children())[:5])).to(dist_util.dev()) # replace with better neural model for similarity
    cosmodel.eval()
    cos = th.nn.CosineSimilarity(dim = 1, eps = 1e-6).to(dist_util.dev())

    # supply simple gradients
    def cond_fn(x, t, ref_img=None):
        assert ref_img is not None
        batchsize = x.shape[0]

        with th.enable_grad():
            # feature
            x_in = x.detach().requires_grad_(True)
            y_feat = stagevgg(ref_img)
            x_feat = stagevgg(x_in)
            target_feat = block_adaIN(x_feat, y_feat, blocknum=args.area)
            gap = (x_feat - target_feat) ** 2

            ## original image feature
            # target = block_adaIN(x_in, ref_img, blocknum=1)
            # gap = (x_in - target) ** 2

            # # cos similarity
            # deepfeature1 = cosmodel(x_in)
            # deepfeature2 = cosmodel(ref_img)
            # grad2 = batchsize * th.autograd.grad(cos(deepfeature1, deepfeature2).mean(), x_in)[0] * args.classifier_scale
            grad = th.autograd.grad(gap.sum(), x_in)[0] * args.classifier_scale
            return [-grad]#, grad2]

    logger.log("loading data...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("sampling...")
    count = 0
    while count * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        condition_kwargs = {}
        if "ref_img" in model_kwargs.keys():
            condition_kwargs["ref_mean"], condition_kwargs["ref_std"] = calc_mean_std(blockzation(model_kwargs["ref_img"], args.area))
            condition_kwargs["area"] = args.area
            condition_kwargs["range_t"] = args.range_t
            condition_kwargs["detail_merge"] = args.detail_merge
        # to calculate the mean and var, in shape of [batch, channel, blocknum, blocknum, 1 , 1]
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            condition_kwargs=condition_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev()
        )

        for i in range(args.batch_size):
            out_path = os.path.join(args.save_dir,
                                    f"{str(count * args.batch_size + i).zfill(5)}.png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

        count += 1
        logger.log(f"created {count * args.batch_size} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1000,
        batch_size=4,
        range_t=0,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_dir="",
        classifier_scale=1,
        vggdepth = 1,
        area = 16,
        affine = False, # keep image features affine
        losstype = "KL", # "MSE" or "KL"
        init_with_blockadain = True,
        detail_merge=False,
        diffusionmodel="ADM",
        )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
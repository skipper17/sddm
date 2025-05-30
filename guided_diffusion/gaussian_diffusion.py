"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
from cv2 import norm

import numpy as np
import torch as th
import torch.nn as nn

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.trigger = False
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.sqrt_recip_alphas = np.sqrt(1.0 / alphas)
        self.weight_energy = self.betas / np.sqrt(alphas)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        # assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_sample_sta(self, mean, var, t):
        """
        get the corresponding mean and variance for a given number of diffusion steps.

        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param mean: the mean of the ref img.
        :param var: the var of the ref img.
        :return: the corresponding mean and var of the ref img in time t.
        """
        assert mean.shape == var.shape
        a = _extract_into_tensor(self.sqrt_alphas_cumprod, t, mean.shape)
        b = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, var.shape)

        return a * mean, a ** 2 * var + b ** 2

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        # for v,k in model_kwargs.items():
        #     print(v)
        #     try:
        #         print(k.shape)
        #     except AttributeError:
        #         print(k)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None, condition_kwargs=None, onManifold=False):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        # if model_kwargs is not None and "ref_img" in model_kwargs:
        assert model_kwargs is not None
        assert condition_kwargs is not None
        assert "ref_img" in model_kwargs
        assert "ref_mean" in condition_kwargs
        assert "ref_std" in condition_kwargs

        ref_noisyimg = self.q_sample(model_kwargs["ref_img"], t, 0)
        gradients = cond_fn(x, self._scale_timesteps(t), ref_img=model_kwargs["ref_img"], ref_noisyimg=ref_noisyimg) # tuple or list

        if t[0] > condition_kwargs["range_t"]:

            mean_t, var_t = self.q_sample_sta(condition_kwargs["ref_mean"], condition_kwargs["ref_std"] ** 2, t)
            std_t = th.sqrt(var_t)
            f = _extract_into_tensor(self.sqrt_recip_alphas, t, x.shape) * x - x
            dg = (p_mean_var["mean"] - _extract_into_tensor(self.sqrt_recip_alphas, t, x.shape) * x).float() # the grad diffusion gives
            dg_m, dg_o = divide_gradient(x, dg, mean_t, condition_kwargs["area"]) 
            
            for i in range(len(gradients)):
                gradients[i], _ = divide_gradient(x,gradients[i], mean_t, condition_kwargs["area"])
                # gradients[i] = gradients[i] * p_mean_var["variance"]
                gradients[i] = gradients[i] * _extract_into_tensor(self.weight_energy, t, x.shape) / gradients[i].norm(dim=[2,3],keepdim= True) * dg_m.norm(dim=[2,3], keepdim= True) * 25
                if condition_kwargs["detail_merge"]:
                    gradients[i] = blockzation(gradients[i], condition_kwargs["area"])
            

            # #print diffusion gradient in sub-manifold and other part 
            # print(t[0])
            # print(dg_m.norm())
            # print(dg_o.norm())

            # print(f.norm())
            # print((dg_o+f).norm())
            # print(gradients[0].norm())
            # print((dg_m * f).sum())
            # print(((dg_o.reshape(dg_o.shape[0],-1) ** 2).sum(dim = -1) / (dg.reshape(dg.shape[0],-1) ** 2).sum(dim = -1)).mean())
            if not condition_kwargs["detail_merge"]:
                gradients = th.stack([dg_m , *gradients], -3) # batch, channel, number, h, w
                gradient = frank_wolfe_solver(gradients, ind_dim=2)
            else:
                gradients = th.stack([blockzation(dg_m, condition_kwargs["area"]), *gradients], -3)  # batch, channel, block, block, number, h/block, w/block
                gradient = unblockzation(frank_wolfe_solver(gradients,ind_dim=4))
            # gradient = dg_m + gradients[0]
            # sub-mainfold_t restore
            # middle = block_adaIN(x+gradient, is_simplied=True, style_mean=mean_t, style_std=std_t, blocknum=condition_kwargs["area"])
            middle = x + gradient 

            if onManifold:
                return block_adaIN(middle, is_simplied= True, style_mean=mean_t, style_std=std_t, blocknum=condition_kwargs["area"])
            # sub-mainfold_{t-1} restore
            # ref_mean, ref_var = self.q_sample_sta(condition_kwargs["ref_mean"], condition_kwargs["ref_std"] ** 2, t)
            # ref_std = th.sqrt(ref_var)
            # final = block_adaIN(middle+dg_o, is_simplied= True, style_mean=ref_mean, style_std=ref_std, blocknum=condition_kwargs["area"])
            final = middle + dg_o + f
            new_mean = (
                # p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float() # EDIT
                final.float()
                # p_mean_var["mean"].float()
            )
        else:
            # # MOO
            # f = _extract_into_tensor(self.sqrt_recip_alphas, t, x.shape) * x - x
            # dg = (p_mean_var["mean"] - _extract_into_tensor(self.sqrt_recip_alphas, t, x.shape) * x).float() # the grad diffusion gives

            # for i in range(len(gradients)):
            #     gradients[i] = gradients[i] * _extract_into_tensor(self.weight_energy, t, x.shape) / gradients[i].norm(dim=[2,3],keepdim= True) * dg.norm(dim=[2,3], keepdim= True) * 25
            #     if condition_kwargs["detail_merge"]:
            #         gradients[i] = blockzation(gradients[i], condition_kwargs["area"])

            # if not condition_kwargs["detail_merge"]:
            #     gradients = th.stack([dg , *gradients], -3) # batch, channel, number, h, w
            #     gradient = frank_wolfe_solver(gradients, ind_dim=2)
            # else:
            #     gradients = th.stack([blockzation(dg, condition_kwargs["area"]), *gradients], -3)  # batch, channel, block, block, number, h/block, w/block
            #     gradient = unblockzation(frank_wolfe_solver(gradients,ind_dim=4))

            # middle = x + gradient
            # final = middle + f

            # Direct ADD
            # weight_t = _extract_into_tensor(self.weight_energy, t, x.shape)
            # li = 2# 0.5
            # ls = 500# 700
            # final = p_mean_var["mean"] + ls * weight_t * gradients[0] + li * weight_t * gradients[1]

            # if not self.trigger:
            #     new_mean = ( (0.5 * final + 0.5* ref_noisyimg).float())
            #     self.trigger = True
            #     return new_mean
            # new_mean = ( final.float())
            new_mean = ( p_mean_var["mean"].float())
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        condition_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            # last time refine 10 times
            if  t[0] == condition_kwargs["range_t"] + 1:
                for refinetimes in range(4):
                    out = self.p_mean_variance(
                            model,
                            x,
                            t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            model_kwargs=model_kwargs,
                        )
                    x = self.condition_mean(
                        cond_fn, out, x, t, model_kwargs=model_kwargs, condition_kwargs=condition_kwargs
                    )    

            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs, condition_kwargs=condition_kwargs
            )
            # out["mean"] = cond_fn(out["mean"], self._scale_timesteps(t), **model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        condition_kwargs=None,
        device=None,
        progress=False,
        resizers=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            condition_kwargs=condition_kwargs,
            device=device,
            progress=progress,
            resizers=resizers,
        ):
            final = sample

        return final["sample"]



    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        condition_kwargs=None,
        device=None,
        progress=False,
        resizers=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        # # Here to do Normalize
        # img = adaptive_instance_normalization(img, is_simplied=True, style_mean=th.zeros(1, device=device), style_std=th.ones(1, device=device))
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        if resizers is not None:
            down, up = resizers

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    condition_kwargs=condition_kwargs,
                )
                #### ILVR #### 做blockadain ,lazy version
                # if resizers is not None:
                if i > condition_kwargs["range_t"]:
                    # out["sample"] = out["sample"] - up(down(out["sample"])) + up(
                    #     down(self.q_sample(model_kwargs["ref_img"], t, th.randn(*shape, device=device))))
                    # out["sample"] = block_adaIN(out["sample"],self.q_sample(model_kwargs["ref_img"], t, th.randn(*shape, device=device)), blocknum=16)
                    ref_mean, ref_var = self.q_sample_sta(condition_kwargs["ref_mean"], condition_kwargs["ref_std"] ** 2, t - 1)
                    ref_std = th.sqrt(ref_var)
                    out["sample"] = block_adaIN(out["sample"], is_simplied= True, style_mean=ref_mean, style_std=ref_std, blocknum=condition_kwargs["area"])

                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # assert (len(size) == 4)
    other = size[:-2]
    feat_var = feat.reshape(*other, -1).var(dim=-1) + eps
    feat_std = feat_var.sqrt().reshape(*other, 1, 1)
    feat_mean = feat.reshape(*other, -1).mean(dim=-1).reshape(*other, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat = None, is_simplied = False, style_mean = None, style_std = None):
    size = content_feat.size()
    if not is_simplied:
        assert style_feat is not None
        assert (content_feat.size()[:-2] == style_feat.size()[:-2])
        style_mean, style_std = calc_mean_std(style_feat)
    else:
        assert style_mean is not None
        assert style_std is not None
    content_mean, content_std = calc_mean_std(content_feat)
    # return content_feat - content_mean.expand(size) + style_mean.expand(size)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# def block_adaIN(content_feat, style_feat, block = 32):
#     assert (content_feat.size()[:-2] == style_feat.size()[:-2])
#     assert content_feat.size()[-1] % block == 0
#     assert content_feat.size()[-2] % block == 0
#     size = content_feat.size()
#     N, C, H, W = size
#     newC = C * size[-1] * size[-2] / block / block
#     return adaptive_instance_normalization(content_feat.reshape(N,C,H // block, block, W // block, block).transpose(3,4),
#                                         style_feat.reshape(N,C,H // block, block, W // block, block).transpose(3,4)
#                                         ).transpose(3,4).reshape(size)

def block_adaIN(content_feat, style_feat = None, blocknum = 16, is_simplied = False, style_mean = None, style_std = None):
    if not is_simplied:
        assert (content_feat.size()[:-2] == style_feat.size()[:-2])
        content_feat = blockzation(content_feat, blocknum)
        style_feat = blockzation(style_feat, blocknum)
        return  unblockzation(adaptive_instance_normalization(content_feat, style_feat))
    else:
        assert style_mean is not None
        assert style_std is not None
        content_feat = blockzation(content_feat, blocknum)
        return unblockzation(adaptive_instance_normalization(content_feat, is_simplied=True, style_mean=style_mean, style_std=style_std))

def blockzation(feat, blocknum = 16):
    H, W = feat.size()[-2:]
    assert H % blocknum == 0
    assert W % blocknum == 0
    size = feat.size()[:-2]
    feat = feat.reshape(*size,blocknum, H // blocknum, blocknum, W // blocknum).transpose(-2, -3)
    return feat

def unblockzation(feat):
    size = feat.size()
    H = size[-4] * size[-2]
    W = size[-3] * size[-1]
    size = size[:-4]
    return feat.transpose(-2, -3).reshape(*size, H, W)

class StageVGG(nn.Module):
    def __init__(self, path="/home/sunsk/Models/vgg/vgg_normalised.pth", level = 1):
        super(StageVGG, self).__init__()
        encoder = nn.Sequential(
                nn.Conv2d(3, 3, (1, 1)),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(3, 64, (3, 3)),
                nn.ReLU(),  # relu1-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),  # relu1-2
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 128, (3, 3)),
                nn.ReLU(),  # relu2-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),  # relu2-2
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 256, (3, 3)),
                nn.ReLU(),  # relu3-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  # relu3-2
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  # relu3-3
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  # relu3-4
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 512, (3, 3)),
                nn.ReLU(),  # relu4-1, this is the last layer used
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu4-2
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu4-3
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu4-4
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu5-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu5-2
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu5-3
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU()  # relu5-4
        )
        
        encoder.load_state_dict(th.load(path))
        enc_layers = list(encoder.children())[:44]
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        self.level = level
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def encode_with_level(self, input, level):
        for i in range(level):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            input = func(input)
        return input

    def forward(self, x):
        return self.encode_with_level(x, self.level)

def dynamic_adj_add(vec1, vec2):
    # print(vec1.shape)
    assert vec1.shape == vec2.shape
    shape = vec1.shape
    vec1 = vec1.view(shape[0], -1)
    vec2 = vec2.view(shape[0], -1)
    v1v1 = (vec1 * vec1).mean(dim = 1)
    v1v2 = (vec1 * vec2).mean(dim = 1)
    v2v2 = (vec2 * vec2).mean(dim = 1)
    gamma = min_norm_element_from2(v1v1, v1v2, v2v2).view(shape[0], 1)
    coef = ((1 - gamma)/gamma).clamp(0,30)
    coef = th.where(th.isnan(coef), th.full_like(coef, 0), coef)
    # return (gamma * vec1 + (1 - gamma) * vec2).view(shape)
    return (vec1 + coef * vec2).view(shape)

def min_norm_element_from2(v1v1, v1v2, v2v2):
    divide = v1v1+v2v2 - 2*v1v2
    gamma = -1.0 *  (v1v2 - v2v2) / divide
    gamma = th.where(th.isnan(gamma), th.full_like(gamma, 1), gamma)
    return gamma.clamp(0, 1)

# veclist shape :  batch1, batch2, ..., batchn, number, others
# 方案一, 三元优化: 将当前的方案暴力拓展到三个变量的情况(必须做clamp, 是否约束主梯度之外的模长, 约束了模长理论上才sound)
# 方案二, 二元优化: 提取跟diffuison方向垂直的分量, 将这些分量dynamic化(约束或者不约束模长, 理论上都sound)
# 方案三, 二元优化: 仅约束非主梯度的模长(理论上sound)
# 上述方案abandoned, 采用流形来分割梯度, 自由结合即可
def frank_wolfe_solver(veclist, ep = 1e-4, maxnum = 20, ind_dim=1):
    shape = veclist.shape
    veclist = veclist.view(*shape[:ind_dim+1], -1) # shape [B, N, O]
    M = veclist @ veclist.transpose(-1,-2) # shape [B, N, N]
    a = (th.ones(shape[:ind_dim+1]) / shape[ind_dim]).unsqueeze(ind_dim).to(veclist.device) # shape [B, 1, N]
    for _ in range(maxnum):
        minrank = th.argmin(a @ M, dim = ind_dim+1) # shape [B, 1]
        minonehot = th.zeros(shape[:ind_dim+1]).to(veclist.device).scatter_(ind_dim, minrank, 1).unsqueeze(ind_dim) # shape [B, 1, N]
        gamma = min_norm_element_from2(minonehot @ M @ minonehot.transpose(-1,-2),minonehot @ M @ a.transpose(-1,-2), a @ M @ a.transpose(-1,-2)).reshape(*shape[:ind_dim], 1, 1)
        # minvec = th.diagonal(veclist[:,minrank]).transpose(0,1)
        a = (1-gamma)* a + gamma * minonehot
        if th.abs(gamma).mean()< ep:
            return (a @ veclist).view(*shape[:ind_dim], *shape[ind_dim+1:])
    return (a @ veclist).view(*shape[:ind_dim], *shape[ind_dim+1:])



# 给出基准梯度, 仅保留垂直方向的分量 independdims 表示多少分量是并列的不计入计算的
def get_vertical_component(vec, vec_base, independdims = 1):
    assert vec.shape  == vec_base.shape
    assert vec.device == vec_base.device
    shape = vec.shape
    vec = vec.reshape(*shape[:independdims], -1)
    vec_base = vec_base.reshape(*shape[:independdims], -1)
    cos = (vec * vec_base).sum(dim = -1, keepdim = True) / (vec_base ** 2).sum(dim = -1, keepdim = True)
    vec_align = cos * vec_base
    return (vec - vec_align).reshape(shape)

# 实现提取sub manifold分量的算法, 通过做两次垂直分量的提取来实现
# img 当前时间步的图像
# ref 对应的参考的均值和方差
#
# return: 
## 沿着流形的分量, 流形外的分量
def divide_gradient(img, delta, refmean, blocknum = 16):
    assert img.shape == delta.shape
    assert img.device == refmean.device == delta.device
    blockedimg = blockzation(img, blocknum)
    blockeddelta = blockzation(delta, blocknum)
    meanvec = th.ones(blockedimg.shape).to(blockedimg.device)
    middledelta = get_vertical_component(blockeddelta, meanvec, -2)
    finaldelta = get_vertical_component(middledelta, blockedimg - refmean, -2)
    resdelta = blockeddelta - finaldelta
    
    return unblockzation(finaldelta), unblockzation(resdelta)

def retraction(img, refmean, refstd, blocknum = 16):
    assert img.device == refmean.device == refstd.device
    block_img = blockzation(img, blocknum)
    block_restractioned = adaptive_instance_normalization(block_img, is_simplied= True, style_mean=refmean, style_std=refstd)

    return unblockzation(block_restractioned)

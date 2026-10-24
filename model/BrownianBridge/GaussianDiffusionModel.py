"""标准高斯 DDPM + DDIM 采样 (Conditional DDIM 基线, 论文表 3 第 5 行)

与 BrownianBridgeModel 的受控差异(其余全部一致, 逐开关 diff 见 configs/gddpm_joint_240.yaml):
  - 前向过程: x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps, beta linear, T=1000
  - 训练目标: epsilon 预测 + MSE (DDPM 教科书标准)
  - 采样: DDIM 50 步 eta=0, 起点纯噪声 x_T~N(0,I)
主干/条件通道拼接(SpatialRescaler, in_channels=6=x_t3+cond3)/数据口径/EMA/checkpoint
与 BBDM 完全同款, 由 BBDMRunner 直接复用(model_type: GDDPM)。
注: BBDM 像素模式下的 task_embed 分支 (context.shape[1]==in_channels) 恒为 False,
属于死代码, 本类不保留(零行为差异)。
"""
import itertools

import numpy as np
import torch
import torch.nn as nn

from ..utils import extract
from .base.modules.diffusionmodules.openaimodel import UNetModel
from .base.modules.diffusionmodules.util import make_beta_schedule


class GaussianDiffusionModel(nn.Module):
    """Standard Gaussian DDPM with channel-concat condition and DDIM sampling."""

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        model_params = model_config.BB.params

        self.num_timesteps = model_params.num_timesteps
        self.channels = model_params.UNetParams.in_channels
        self.image_size = model_params.UNetParams.image_size
        self.condition_key = model_params.UNetParams.condition_key

        self.beta_schedule = getattr(model_params, "beta_schedule", "linear")
        self.ddim_steps = int(getattr(model_params, "ddim_steps", 50))

        self.num_tasks = 2
        self.task_weights = getattr(model_params, "task_weights", [1.0, 0.8])
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))
        self.register_schedule()

    def register_schedule(self):
        betas = torch.tensor(
            make_beta_schedule(self.beta_schedule, self.num_timesteps),
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        # DDIM 均匀子序列: 999 -> 0 降序 (含两端, 严格递减)
        ts = np.linspace(self.num_timesteps - 1, 0, self.ddim_steps).round().astype(np.int64)
        self.register_buffer("ddim_ts", torch.from_numpy(ts))

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return itertools.chain(self.denoise_fn.parameters())

    # ===================== 训练 =====================

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        sqrt_abar = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_abar * x0 + sqrt_one_minus * noise

    def p_losses(self, x0, context, t, current_task_id=0):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.denoise_fn(x_t, timesteps=t, context=context)
        loss = torch.nn.functional.mse_loss(eps_pred, noise) * self.task_weights[current_task_id]
        return loss, {"recloss": float(loss.detach())}

    def forward(self, x, y, context=None, current_task_id=0, organ_label=None, **kwargs):
        if self.training:
            self.step_counter += 1
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        batch_size = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device).long()
        loss, log = self.p_losses(x, context, t, current_task_id=current_task_id)
        return loss, log

    # ===================== DDIM 采样 =====================

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False, task_id=0, **kwargs):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        return self.ddim_sample_loop(context, clip_denoised)

    @torch.no_grad()
    def ddim_sample_loop(self, context, clip_denoised=True):
        device = context.device
        b = context.shape[0]
        x_t = torch.randn((b, 3, self.image_size, self.image_size), device=device)
        ts = self.ddim_ts.tolist()
        for i in range(len(ts)):
            t = int(ts[i])
            t_prev = int(ts[i + 1]) if i + 1 < len(ts) else -1
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            eps = self.denoise_fn(x_t, timesteps=t_batch, context=context)
            abar_t = self.alphas_cumprod[t]
            x0_pred = (x_t - torch.sqrt(1.0 - abar_t) * eps) / torch.sqrt(abar_t)
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)
            if t_prev < 0:
                return x0_pred
            abar_prev = self.alphas_cumprod[t_prev]
            x_t = torch.sqrt(abar_prev) * x0_pred + torch.sqrt(1.0 - abar_prev) * eps
        return x_t

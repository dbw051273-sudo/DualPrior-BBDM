"""Image-to-Image Schrödinger Bridge (I2SB) Model

Reference:
    Guan-Horng Liu, Arash Vahdat, Lin-Ching Huang, Evangelos Theodorou, Weili Nie, Karsten Kreis.
    "I2SB: Image-to-Image Schrödinger Bridge." ICML 2023.
    arXiv: 2302.05872.

Integrated into BBDM/Swin-UNet virtual staining framework for cross-organ benchmark comparison.
Controlled Variables:
    - Input: 6-channel concatenation (x_t ⊕ Phase PI condition)
    - Target: Standard nominal H&E histology
    - Backbone: Swin-UNet or standard UNet (same depth/heads/channels)
    - Bridge Mechanism: Entropy-regularized optimal transport Schrödinger Bridge
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import extract
from .base.modules.diffusionmodules.openaimodel import UNetModel
from .base.modules.diffusionmodules.util import make_beta_schedule


class I2SBModel(nn.Module):
    """Image-to-Image Schrödinger Bridge Model with direct x_0-prediction parameterization."""

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        model_params = model_config.BB.params

        self.num_timesteps = getattr(model_params, "num_timesteps", 1000)
        self.channels = model_params.UNetParams.in_channels
        self.image_size = model_params.UNetParams.image_size
        self.condition_key = model_params.UNetParams.condition_key

        self.beta_schedule = getattr(model_params, "beta_schedule", "linear")
        self.sample_steps = int(getattr(model_params, "sample_steps", 20))
        self.sigma_max = float(getattr(model_params, "sigma_max", 1.0))
        self.sample_eta = float(getattr(model_params, "sample_eta", 0.0))  # 0.0=deterministic ODE, 1.0=SDE
        self.loss_type = getattr(model_params, "loss_type", "l1")

        self.num_tasks = 2
        self.task_weights = getattr(model_params, "task_weights", [1.0, 0.8])
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        # 骨干去噪网络 (Swin-UNet / OpenAI UNet)
        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))
        self.register_schedule()

    def register_schedule(self):
        """注册 Schrödinger Bridge 方差与边际转移系数"""
        if self.beta_schedule == "linear":
            betas = torch.tensor(
                make_beta_schedule("linear", self.num_timesteps, linear_start=1e-4, linear_end=0.02),
                dtype=torch.float32,
            )
            # 累积方差: sigma_t^2 = sum_{s=1}^t beta_s
            sigmas_sq = torch.cumsum(betas, dim=0)
            sigmas = torch.sqrt(sigmas_sq)
            sigmas = sigmas / sigmas[-1] * self.sigma_max
            sigmas_sq = sigmas ** 2
        elif self.beta_schedule == "geometric":
            sigmas = torch.tensor(
                np.geomspace(1e-4, self.sigma_max, self.num_timesteps),
                dtype=torch.float32,
            )
            sigmas_sq = sigmas ** 2
        else:
            # 默认线性时间插值
            t_ratio = torch.linspace(0.001, 1.0, self.num_timesteps, dtype=torch.float32)
            sigmas = t_ratio * self.sigma_max
            sigmas_sq = sigmas ** 2

        sigma_1_sq = sigmas_sq[-1]  # 终点最大方差
        # gamma_t = sigma_t^2 / sigma_1^2 in [0, 1]
        gamma_t = sigmas_sq / sigma_1_sq
        alpha_t = 1.0 - gamma_t  # 对应 x_0 (H&E 目标) 权重
        std_t = torch.sqrt(gamma_t * alpha_t) * torch.sqrt(sigma_1_sq)

        self.register_buffer("sigmas", sigmas)
        self.register_buffer("sigmas_sq", sigmas_sq)
        self.register_buffer("gamma_t", gamma_t)
        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("std_t", std_t)

        # 采样时间网格: 999 -> 0 严格递减子序列
        ts = np.linspace(self.num_timesteps - 1, 0, self.sample_steps).round().astype(np.int64)
        self.register_buffer("sample_ts", torch.from_numpy(ts))

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return itertools.chain(self.denoise_fn.parameters())

    # ===================== 训练过程 =====================

    def q_sample(self, x0, x1, t, noise=None):
        """Schrödinger Bridge 前向边缘分布采样:
        q(x_t | x_0, x_1) = N( alpha_t * x_0 + (1-alpha_t) * x_1, std_t^2 * I )
        x0: 目标真实 H&E 染色图像 (clean target)
        x1: 条件输入免标记相位图 (Phase PI, condition)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha = extract(self.alpha_t, t, x0.shape)
        std = extract(self.std_t, t, x0.shape)
        mu_t = alpha * x0 + (1.0 - alpha) * x1
        x_t = mu_t + std * noise
        return x_t

    def p_losses(self, x0, context, t, current_task_id=0):
        """I2SB 损失计算:
        网络接收 (x_t ⊕ context), 直接预测终点真实目标 x_0
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0=x0, x1=context, t=t, noise=noise)
        pred_x0 = self.denoise_fn(x_t, timesteps=t, context=context)

        if self.loss_type == "l1":
            loss = F.l1_loss(pred_x0, x0)
        else:
            loss = F.mse_loss(pred_x0, x0)

        loss = loss * self.task_weights[current_task_id]
        return loss, {"recloss": float(loss.detach()), "loss": float(loss.detach())}

    def forward(self, x, y, context=None, current_task_id=0, organ_label=None, **kwargs):
        """
        BBDMRunner 规范调用接口:
        x: 目标真值标称染色图 (H&E, clean target x_0)
        y: 条件输入免标记相位图 (Phase PI, condition x_1)
        """
        if self.training:
            self.step_counter += 1
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        batch_size = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device).long()
        loss, log = self.p_losses(x0=x, context=context, t=t, current_task_id=current_task_id)
        return loss, log

    # ===================== 推理逆向采样 =====================

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False, task_id=0, **kwargs):
        """
        推理接口: 从输入条件 y (Phase PI) 出发逆向转移生成 H&E
        """
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        return self.i2sb_sample_loop(context, clip_denoised=clip_denoised)

    @torch.no_grad()
    def i2sb_sample_loop(self, context, clip_denoised=True):
        """
        I2SB 逆向条件转移采样循环 (Algorithm 1 in Liu et al., ICML 2023)
        起点: x_T = context (测试输入免标记相位图 x_1)
        逐步逆向去噪至 x_0 (生成标称 H&E 染色图)
        """
        device = context.device
        b = context.shape[0]
        # 起点初始化于输入相位图
        x_t = context.clone()

        ts = self.sample_ts.tolist()
        num_steps = len(ts)

        for i in range(num_steps):
            t = int(ts[i])
            t_prev = int(ts[i + 1]) if i + 1 < num_steps else -1
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            # 预测干净终点 x_0
            pred_x0 = self.denoise_fn(x_t, timesteps=t_batch, context=context)
            if clip_denoised:
                pred_x0 = pred_x0.clamp(-1.0, 1.0)

            # 终点退出
            if t_prev < 0:
                return pred_x0

            # 计算转移后验: q(x_{t_prev} | x_t, pred_x0)
            sig_t_sq = self.sigmas_sq[t]
            sig_prev_sq = self.sigmas_sq[t_prev]

            # 权重系数
            c_x = sig_prev_sq / sig_t_sq
            c_0 = (sig_t_sq - sig_prev_sq) / sig_t_sq
            mu_step = c_x * x_t + c_0 * pred_x0

            if self.sample_eta > 0.0 and i < num_steps - 1:
                # 随机性扰动注入
                var_step = (sig_prev_sq * (sig_t_sq - sig_prev_sq)) / sig_t_sq
                std_step = torch.sqrt(torch.clamp(var_step, min=1e-12))
                noise = torch.randn_like(x_t)
                x_t = mu_step + self.sample_eta * std_step * noise
            else:
                # 确定性转移 (ODE 轨迹)
                x_t = mu_step

        return x_t

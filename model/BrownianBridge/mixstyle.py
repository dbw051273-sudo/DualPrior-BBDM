# -*- coding: utf-8 -*-
"""
MixStyle: Domain Generalization with MixStyle (ICLR 2021, Kaiyang Zhou et al., 被引 500+)
专用于 Swin-UNet (B, L, C) 与卷积特征图 (B, C, H, W) 的实例归一化特征解耦模块。
"""
import torch
import torch.nn as nn

class MixStyle(nn.Module):
    """
    MixStyle 官方原理实现：
    在批次内部将解剖内容与染色风格统计量（均值与标准差）解耦并重组。
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x

        if x.dim() == 3:
            # Swin-UNet token 格式: (B, L, C)
            B, L, C = x.shape
            if B <= 1:
                return x
            mu = x.mean(dim=1, keepdim=True)  # (B, 1, C)
            var = x.var(dim=1, keepdim=True)
            sig = (var + self.eps).sqrt()

            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            lmda = self.beta.sample((B, 1, 1)).to(x.device)
            perm = torch.randperm(B)

            mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
            sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)

            return x_normed * sig_mix + mu_mix

        elif x.dim() == 4:
            # 标准 CNN 格式: (B, C, H, W)
            B = x.shape[0]
            if B <= 1:
                return x
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()

            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
            perm = torch.randperm(B)

            mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
            sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)

            return x_normed * sig_mix + mu_mix
        else:
            return x

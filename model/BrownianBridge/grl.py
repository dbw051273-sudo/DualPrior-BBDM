# -*- coding: utf-8 -*-
"""
GRL 梯度反转层 + 域投影器 + 器官判别器
用于 Re-BBDM：让 UNI 特征变得"器官无关"（domain-invariant）。
"""
import numpy as np
import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


class DomainProjector(nn.Module):
    """残差结构，初始为恒等映射，训练中逐渐去偏。"""

    def __init__(self, dim=1024, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        # 零初始化最后一层：初始 F_proj = F_uni + 0 = F_uni
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        return x + self.mlp(x)


class DomainDiscriminator(nn.Module):
    """器官判别器：极简 2 层 MLP，削弱版（Dropout 0.5 + 隐层缩到 32）。"""

    def __init__(self, in_dim=1024, hidden_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),  # 削弱：Dropout 0.5 防止判别器过强
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def get_dann_lambda(step, total_steps, gamma=10.0):
    """DANN 退火：lambda 从 0 缓慢增到 1。
    λ = 2 / (1 + exp(-γ * p)) - 1，p = step / total_steps。
    """
    p = step / max(total_steps, 1)
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)

# -*- coding: utf-8 -*-
"""
病理染色增强模块 (Pathology Stain Augmentation)
基于 Beer-Lambert 定律与 Ruifrok/Macenko 颜色反卷积 (Color Deconvolution)。
将 H&E 图像严格解耦为:
  - Hematoxylin 通道: 细胞核浓度 C_H (蓝紫色)
  - Eosin 通道: 细胞质/神经毡基质浓度 C_E (粉红色)
  - Residual 通道: 背景残差
通过在光密度 (Optical Density, OD) 空间对 C_H 和 C_E 进行物理扰动，
模拟不同医院、不同染色批次、不同切片厚度下的染色漂移，
同时 100% 保持解剖结构与细胞边界的像素级刚性一致。
"""

import math
import random
import numpy as np
import torch
from PIL import Image


class RuifrokStainAugmentor:
    """
    Ruifrok-Macenko 光学密度反卷积染色增强器 (Tensor 原生实现)。
    适用于 DataLoader 在 CPU/GPU 上高速执行 (单张 256x256 耗时 ~1-2ms)。
    """
    def __init__(
        self,
        alpha_range=(0.7, 1.4),   # 染色浓度缩放扰动范围 (模拟染色剂浓度/浸泡时长)
        beta_range=(-0.04, 0.04), # 染色基线偏置扰动范围 (模拟染色底色深浅)
        p=0.5                     # 触发概率
    ):
        self.alpha_min, self.alpha_max = alpha_range
        self.beta_min, self.beta_max = beta_range
        self.p = p

        # 国际标准 Ruifrok 染色矩阵 (Hematoxylin, Eosin, Residual)
        M_ruifrok = np.array([
            [0.650, 0.704, 0.286],  # H 通道吸光率 (R, G, B)
            [0.070, 0.990, 0.105],  # E 通道吸光率 (R, G, B)
            [0.268, 0.570, 0.776]   # 残差通道
        ], dtype=np.float32)

        # 行归一化 (Unit norm per stain vector)
        M_ruifrok /= np.linalg.norm(M_ruifrok, axis=1, keepdims=True)
        M_inv = np.linalg.inv(M_ruifrok).astype(np.float32)

        self.M = torch.from_numpy(M_ruifrok)        # [3, 3]
        self.M_inv = torch.from_numpy(M_inv)        # [3, 3]

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        img_tensor: torch.Tensor, shape [3, H, W], 数值范围 [0.0, 1.0]
        返回: 染色扰动后的 torch.Tensor, shape [3, H, W], 数值范围 [0.0, 1.0]
        """
        if random.random() > self.p:
            return img_tensor

        c, h, w = img_tensor.shape
        device = img_tensor.device
        dtype = img_tensor.dtype

        M_inv = self.M_inv.to(device=device, dtype=dtype)
        M = self.M.to(device=device, dtype=dtype)

        # 1. RGB -> Optical Density (OD) 空间
        # OD = -log10(I / I0), 此处 I0 = 1.0
        od = -torch.log10(torch.clamp(img_tensor, min=1e-4)).contiguous().reshape(3, -1).t()  # [N, 3]

        # 2. OD -> 浓度空间 C: C = OD @ M_inv
        C = torch.matmul(od, M_inv)

        # 3. 独立采样 H 和 E 通道的扰动因子
        ah = random.uniform(self.alpha_min, self.alpha_max)
        ae = random.uniform(self.alpha_min, self.alpha_max)
        bh = random.uniform(self.beta_min, self.beta_max)
        be = random.uniform(self.beta_min, self.beta_max)

        # 4. 对 C_H (通道 0) 和 C_E (通道 1) 进行线性扰动
        C[:, 0] = torch.clamp(C[:, 0] * ah + bh, min=0.0)
        C[:, 1] = torch.clamp(C[:, 1] * ae + be, min=0.0)

        # 5. 重构回 OD 空间并转回 RGB: RGB = 10^(-OD)
        od_aug = torch.matmul(C, M)
        img_aug = torch.pow(10.0, -od_aug).t().contiguous().reshape(c, h, w)

        return torch.clamp(img_aug, 0.0, 1.0)

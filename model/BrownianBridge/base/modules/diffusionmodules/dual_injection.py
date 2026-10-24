# -*- coding: utf-8 -*-
"""
Re-BBDM 双通路注入模块
======================
形色解耦注入：
  - F_spatial（空间结构）→ Zero-Conv 逐像素相加（锁位置）
  - F_style（纹理质感）→ Cross-Attention 动态查询（匹配纹理）

设计依据：
  F_spatial 是 2D 网格，有空间坐标 → 直接相加锁位置
  F_style 是离散 Token，无空间坐标 → attention 动态匹配
"""
import torch
import torch.nn as nn


class ZeroConv2d(nn.Module):
    """零卷积：权重初始化为 0 的 1×1 卷积（ControlNet 式无伤接入）。
    初始输出为 0 → 不改变主干特征，训练中逐渐学习。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class SpatialInjection(nn.Module):
    """空间锚定：F_spatial → Zero-Conv → 逐像素加到 Swin 特征图。

    F_spatial 来自 UNI+GRL 的域不变特征。
    需要从 UNI token [B,197,1024] 重塑为空间特征图 [B,1024,14,14]，
    再插值到 Swin 当前层的分辨率。

    用法：在 SwinTransformerBlock 里，W-MSA 之前调用。
    """
    def __init__(self, uni_dim=1024, swin_dim=256):
        super().__init__()
        self.zero_conv = ZeroConv2d(uni_dim, swin_dim)

    def forward(self, swin_feat, f_spatial_tokens, input_resolution):
        """
        swin_feat: [B, H*W, C] Swin 当前层的 token
        f_spatial_tokens: [B, 197, 1024] UNI+GRL 域不变 token
        input_resolution: (H, W) 当前层的空间分辨率
        返回: [B, H*W, C] 加了空间锚定后的 token
        """
        B, N, C = swin_feat.shape
        H, W = input_resolution

        # UNI token [B,197,1024] → 去掉 CLS → [B,196,1024] → [B,1024,14,14]
        spatial = f_spatial_tokens[:, 1:, :]  # [B, 196, 1024]
        spatial = spatial.permute(0, 2, 1).reshape(B, 1024, 14, 14)

        # 插值到 Swin 当前层分辨率
        if spatial.shape[-1] != H or spatial.shape[-2] != W:
            spatial = torch.nn.functional.interpolate(
                spatial, size=(H, W), mode='bilinear', align_corners=False
            )

        # Zero-Conv → [B, C, H, W]
        delta = self.zero_conv(spatial)  # [B, swin_dim, H, W]

        # 转成 token 形状 → 逐像素加
        delta = delta.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        return swin_feat + delta


class TextureCrossAttention(nn.Module):
    """纹理渲染：F_style 作为 K/V，Swin 特征作为 Q。

    F_style 来自 FAISS 检索的 Top-K 原型（经 Attention Pooling 融合）。
    每个 Swin token 可以动态"查询"最匹配的纹理。

    用法：替代原来的 UNI cross-attn，在 SwinTransformerBlock 里。
    """
    def __init__(self, query_dim, style_dim=512, heads=8):
        super().__init__()
        inner_dim = query_dim
        self.heads = heads
        self.scale = (inner_dim // heads) ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(style_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(style_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.)
        )
        # 零初始化：初始不影响主干
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(self, x, f_style):
        """
        x: [B, N_q, query_dim] Swin token（Q）
        f_style: [B, K, style_dim] 检索的 Top-K 纹理原型（K/V）
        """
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(f_style)
        v = self.to_v(f_style)

        q = q.view(q.shape[0], q.shape[1], h, -1).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], h, -1).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], h, -1).transpose(1, 2)

        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)


# ===== 集成到 SwinTransformerBlock 的说明 =====
"""
在 openaimodel.py 的 SwinTransformerBlock 里，需要这样改：

class SwinTransformerBlock(nn.Module):
    def __init__(self, ..., use_spatial_injection=False, use_texture_attn=False, ...):
        ...
        # 空间锚定（Zero-Conv 相加）
        if use_spatial_injection:
            self.spatial_injection = SpatialInjection(uni_dim=1024, swin_dim=dim)
        # 纹理渲染（Cross-Attention）
        if use_texture_attn:
            self.norm_texture = norm_layer(dim)
            self.texture_attn = TextureCrossAttention(query_dim=dim, style_dim=512)

    def forward(self, x, f_spatial=None, f_style=None):
        # === 空间锚定（W-MSA 之前）===
        if self.use_spatial_injection and f_spatial is not None:
            x = self.spatial_injection(x, f_spatial, self.input_resolution)

        # === 原始 W-MSA ===
        shortcut = x
        x = self.norm1(x)
        ...（原 Swin attention 代码不变）...
        x = shortcut + self.drop_path(x)

        # === 纹理渲染（W-MSA 之后）===
        if self.use_texture_attn and f_style is not None:
            x = x + self.texture_attn(self.norm_texture(x), f_style)

        # === FFN ===
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

注入逻辑：
  - SpatialInjection 加在 W-MSA **之前**（先锁位置，再做窗口注意力）
  - TextureCrossAttention 加在 W-MSA **之后**（先用窗口注意力处理空间关系，再注入纹理）
"""


# ===== 自测 =====
if __name__ == "__main__":
    print("双通路注入模块自测...")

    # SpatialInjection
    si = SpatialInjection(uni_dim=1024, swin_dim=256).cuda()
    swin_feat = torch.randn(2, 64, 256).cuda()  # [B, 8*8, 256]
    f_spatial = torch.randn(2, 197, 1024).cuda()
    out1 = si(swin_feat, f_spatial, (8, 8))
    print(f"  SpatialInjection: {tuple(swin_feat.shape)} + {tuple(f_spatial.shape)} → {tuple(out1.shape)}")
    # 验证零初始化：初始 delta 应该是 0
    delta = (out1 - swin_feat).abs().max().item()
    print(f"  零初始化验证: delta={delta:.6f} (应≈0)")
    assert delta < 1e-5, "Zero-Conv 初始化不是零!"

    # TextureCrossAttention
    tca = TextureCrossAttention(query_dim=256, style_dim=512).cuda()
    swin_q = torch.randn(2, 64, 256).cuda()
    f_style = torch.randn(2, 8, 512).cuda()  # Top-8 风格原型
    out2 = tca(swin_q, f_style)
    print(f"  TextureCrossAttention: Q={tuple(swin_q.shape)} K/V={tuple(f_style.shape)} → {tuple(out2.shape)}")
    # 验证零初始化
    delta2 = out2.abs().max().item()
    print(f"  零初始化验证: output={delta2:.6f} (应≈0)")
    assert delta2 < 1e-5, "Texture attn 输出层零初始化失败!"

    print("\n✅ 双通路注入模块全部通过!")

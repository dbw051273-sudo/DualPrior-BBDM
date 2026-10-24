# -*- coding: utf-8 -*-
"""
UNI 病理基础模型封装器(给 BBDM 用)
======================================
输入: H&E 染色图张量 [B, 3, H, W](范围 [0,1] 或 [-1,1])
输出: patch token [B, 197, 1024](1 CLS + 196 patch,每个 1024 维)

设计要点:
1. UNI 权重完全冻结(只提特征,不训练)
2. 内部用 no_grad,不建计算图(省显存)
3. 自动把输入 resize 到 224×224(UNI 的固定输入尺寸)+ ImageNet 归一化
4. 两种返回模式:patch token(cross-attn 用)/ CLS token(分类用)

用法:
    encoder = UNIEncoder().cuda()
    he_image = ...  # [B,3,H,W], 训练时来自 H&E 真值
    tokens = encoder(he_image)        # [B,197,1024],给 cross-attn 当 K/V
    cls = encoder(he_image, return_cls=True)  # [B,1024],算流形约束用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms


class UNIEncoder(nn.Module):
    """UNI 病理基础模型封装。冻结权重,只做特征提取。"""

    def __init__(self, model_name="hf-hub:MahmoodLab/UNI", freeze=True):
        super().__init__()
        # 用 timm 从 HuggingFace 加载 UNI(和 test_uni.py 验证通过的方式一致)
        self.uni = timm.create_model(
            model_name,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        self.embed_dim = self.uni.embed_dim  # 1024

        if freeze:
            for p in self.uni.parameters():
                p.requires_grad = False
            self.uni.eval()

        # UNI 的预处理: resize 到 224 + ImageNet 归一化
        # 注意: UNI 是在 H&E 上训练的,用标准 ImageNet 归一化(Mahmood lab 推荐)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, x):
        """把任意尺寸/范围的图变成 UNI 期望的 [B,3,224,224] + ImageNet 归一化。
        输入 x: [B,3,H,W], 假设是 [0,1] 或 [-1,1] 范围。"""
        # 1. 统一到 [0,1]
        if x.min() < -0.5:  # 大概是 [-1,1]
            x = (x + 1) / 2
        # 2. resize 到 224
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # 3. ImageNet 归一化
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def forward(self, x, return_cls=False):
        """提取特征。
        x: [B,3,H,W] H&E 图(任何尺寸,任何范围)
        return_cls=False → 返回 [B, 197, 1024] patch token(含 CLS, cross-attn 用)
        return_cls=True  → 返回 [B, 1024] CLS token(流形约束/分类用)
        """
        was_training = self.uni.training
        if was_training:
            self.uni.eval()

        x = self._preprocess(x)
        # forward_features 返回 patch token: [B, 197, 1024](CLS 在第 0 位)
        tokens = self.uni.forward_features(x)  # [B, 197, 1024]

        if return_cls:
            out = tokens[:, 0]  # [B, 1024] CLS token
        else:
            out = tokens  # [B, 197, 1024] 全部 patch token

        if was_training:
            self.uni.train()
        return out

    def extract_and_cache(self, x):
        """给训练循环用的便捷接口:返回 patch token 给 cross-attn。
        等价于 forward(x, return_cls=False),但语义更明确。"""
        return self.forward(x, return_cls=False)

    def extract_cls_with_grad(self, x):
        """梯度版 CLS 提取（Foundation Perceptual Loss 专用）。
        权重仍冻结（requires_grad=False），但保留计算图，
        使 L_manifold 的梯度能穿过冻结的 UNI 回传到生成图像 x0_recon。
        用梯度检查点（重算换显存）：ViT-L 全图激活 ~2GB → 只存层边界，
        16GB 显存上必须如此，否则 step 1000 manifold 激活即 OOM。
        注意：调用方需保证 x.requires_grad 且未 detach。
        """
        was_training = self.uni.training
        if was_training:
            self.uni.eval()
        x = self._preprocess(x)
        tokens = torch.utils.checkpoint.checkpoint(
            self.uni.forward_features, x, use_reentrant=False)
        if was_training:
            self.uni.train()
        return tokens[:, 0]  # [B,1024] CLS

    @torch.no_grad()
    def _extract_raw_tokens(self, x):
        """提原始 UNI token（冻结，无梯度）。供 forward_domain_invariant 内部用。"""
        was_training = self.uni.training
        if was_training:
            self.uni.eval()
        x = self._preprocess(x)
        tokens = self.uni.forward_features(x)  # [B, 197, 1024]
        if was_training:
            self.uni.train()
        return tokens.detach()

    def forward_domain_invariant(self, x, domain_projector, grl_layer, discriminator, grl_lambda):
        """域不变前向：UNI 冻结提特征 → DomainProjector（去偏）→ 返回域不变 token + 域 logits。

        参数:
            x: [B,3,H,W] H&E 图
            domain_projector: DomainProjector 实例（残差投影器）
            grl_layer: GradientReversalLayer 实例
            discriminator: DomainDiscriminator 实例
            grl_lambda: 当前 DANN 退火 λ 值

        返回:
            f_proj: [B,197,1024] 域不变 token（喂 cross-attn，不过 GRL）
            domain_logits: [B,num_organs] 域分类 logits（过 GRL，用于对抗 loss）
        """
        # 1. UNI 冻结提原始 token（无梯度）
        raw_tokens = self._extract_raw_tokens(x)  # [B,197,1024], detached

        # 2. DomainProjector（残差，有梯度 → 训练这个）
        f_proj = domain_projector(raw_tokens)  # [B,197,1024]

        # 3. f_proj 直接返回（喂 cross-attn，不过 GRL）
        # 4. mean-pool → GRL → Discriminator → 域 logits
        global_feat = f_proj.mean(dim=1)  # [B,1024]
        feat_reversed = grl_layer(global_feat, grl_lambda)  # GRL：前向不变，反向乘 -λ
        domain_logits = discriminator(feat_reversed)  # [B,num_organs]

        return f_proj, domain_logits


# ===== 自测 =====
if __name__ == "__main__":
    print("UNIEncoder 自测...")
    enc = UNIEncoder().cuda()

    # 测试 1: 256×256 输入(BBDM 的尺寸)
    x256 = torch.rand(2, 3, 256, 256).cuda()
    tok = enc(x256)
    print(f"  256×256 输入 → {tuple(tok.shape)} (期望 (2, 197, 1024))")
    assert tok.shape == (2, 197, 1024), "shape 错!"

    # 测试 2: CLS token
    cls = enc(x256, return_cls=True)
    print(f"  CLS token → {tuple(cls.shape)} (期望 (2, 1024))")
    assert cls.shape == (2, 1024)

    # 测试 3: [-1,1] 范围输入(BBDM 内部格式)
    x_norm = torch.rand(2, 3, 128, 128).cuda() * 2 - 1  # [-1,1]
    tok2 = enc(x_norm)
    print(f"  [-1,1] 128×128 输入 → {tuple(tok2.shape)} (期望 (2, 197, 1024))")
    assert tok2.shape == (2, 197, 1024)

    # 测试 4: 权重冻结确认
    n_trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"  可训练参数: {n_trainable} (期望 0,全冻结)")

    print("\n✅ UNIEncoder 全部通过!")

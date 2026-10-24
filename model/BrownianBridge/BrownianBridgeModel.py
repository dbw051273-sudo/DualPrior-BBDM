import os
import itertools
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from ..utils import extract, default
from .grl import GradientReversalLayer, DomainProjector, DomainDiscriminator, get_dann_lambda
from ..BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from .uni_encoder import UNIEncoder


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class PatchDiscriminator(nn.Module):
    """E4: 谱归一化 PatchGAN (2.77M)。realism-only —— 不接收器官标签/PI 输入,
    只判别真实 H&E vs 生成 H&E 的纹理真实性。fp32 前向(SN 幂迭代与 hinge 在 bf16 下不精确)。"""

    def __init__(self, in_ch=3, base=64):
        super().__init__()

        def sn_conv(c_in, c_out, k=4, s=2, p=1):
            return nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, k, s, p))

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),                # 输入有界, 首层无 SN
            nn.LeakyReLU(0.2, True),
            sn_conv(base, base * 2), nn.LeakyReLU(0.2, True),    # 128x64x64
            sn_conv(base * 2, base * 4), nn.LeakyReLU(0.2, True),  # 256x32x32
            sn_conv(base * 4, base * 8), nn.LeakyReLU(0.2, True),  # 512x16x16
            sn_conv(base * 8, 1, 4, 1, 1),                      # 1x15x15 patch logits
        )

    def forward(self, x):
        if x.is_cuda:  # fp32 孤岛: 外层 bf16 autocast 会强制 conv 降精度, 需显式关闭
            with torch.autocast(device_type='cuda', enabled=False):
                return self.net(x.float())
        return self.net(x.float())


class BrownianBridgeModel(nn.Module):
    """Brownian Bridge model using reconstruction loss only."""

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        model_params = model_config.BB.params

        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = getattr(model_params, "max_var", 1)
        self.eta = getattr(model_params, "eta", 1)
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        self.num_tasks = 2
        self.task_embed_dim = model_params.UNetParams.model_channels
        self.task_embed = nn.Embedding(self.num_tasks, self.task_embed_dim)
        self.task_weights = getattr(model_params, "task_weights", [1.0, 0.8])
        self.noise_level = getattr(model_params, "noise_level", 0)

        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        # ===== UNI cross-attention 配置(可选,默认关,向后兼容) =====
        self.use_uni = getattr(model_params, "use_uni", False)
        self.uni_input_mode = getattr(model_params, "uni_input_mode", "pi")  # 'pi' or 'he'
        uni_kwargs = {}
        # ===== 解码器 FFN MoE(可选,默认关,向后兼容;与 UNI 无关独立生效) =====
        self.use_moe = getattr(model_params, "use_moe", False)
        self.moe_aux_weight = getattr(model_params, "moe_aux_weight", 0.01)
        if self.use_moe:
            uni_kwargs["use_moe"] = True
            uni_kwargs["moe_layers"] = getattr(model_params, "moe_layers", [1, 2])
            uni_kwargs["num_experts"] = getattr(model_params, "num_experts", 3)
        # style_only: 无 UNI 仅建纹理检索通路（CONCH 单独消融用）
        self.style_only = getattr(model_params, "style_only", False)
        if self.use_uni or self.style_only:
            uni_kwargs["use_uni_cross_attn"] = self.use_uni
            uni_kwargs["uni_context_dim"] = getattr(model_params, "uni_context_dim", 1024)
            uni_kwargs["uni_cross_attn_layers"] = getattr(model_params, "uni_cross_attn_layers", [0, 1])
            # Re-BBDM 双通路注入：SpatialInjection(Zero-Conv) + TextureCrossAttention
            uni_kwargs["use_dual_injection"] = getattr(model_params, "use_dual_injection", True)
            uni_kwargs["dual_injection_layers"] = getattr(model_params, "dual_injection_layers", [1, 2])
            uni_kwargs["style_dim"] = getattr(model_params, "style_dim", 512)
            # 实例化 UNI 编码器(冻结) —— 仅真 UNI 模式；style_only 无编码器
            if self.use_uni:
                self.uni_encoder = UNIEncoder(freeze=True)
            else:
                self.uni_encoder = None

            # ===== GRL 域适应组件（Re-BBDM）=====
            self.use_grl = getattr(model_params, "use_grl", False)
            if self.use_grl:
                self.domain_projector = DomainProjector(dim=1024, hidden_dim=512)
                self.grl = GradientReversalLayer()
                self.domain_discriminator = DomainDiscriminator(
                    in_dim=1024, hidden_dim=getattr(model_params, "disc_hidden_dim", 256),
                    num_classes=getattr(model_params, "num_organs", 3)
                )
                self.adv_weight = getattr(model_params, "adv_weight", 0.05)
                self.grl_gamma = getattr(model_params, "grl_gamma", 10.0)
                self.adv_start_step = getattr(model_params, "adv_start_step", self.warmup_steps if hasattr(self, 'warmup_steps') else 1000)
                self.grl_total_steps = getattr(model_params, "grl_total_steps", 50000)
            else:
                self.domain_projector = None
                self.grl = None
                self.domain_discriminator = None
        else:
            self.uni_encoder = None
            self.use_grl = False

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams), **uni_kwargs)

        # ===== Re-BBDM 双通路：FAISS 记忆库 + CONCH 加载（可选）=====
        self.use_faiss = getattr(model_params, "use_faiss", False)
        self.faiss_index = None
        self.faiss_values = None
        self.conch_model = None
        self.conch_preprocess = None
        if self.use_faiss:
            import faiss as _faiss
            import shutil
            bank_dir = getattr(model_params, "faiss_bank_dir", ".")
            # FAISS 不支持中文路径，用 TEMP 中转读
            idx_path = os.path.join(bank_dir, "faiss_memory_bank.index")
            tmp_idx = os.path.join(os.environ.get("TEMP", "/tmp"), "faiss_read_tmp.index")
            if os.path.exists(idx_path):
                shutil.copy(idx_path, tmp_idx)
                self.faiss_index = _faiss.read_index(tmp_idx)
                self.faiss_values = torch.load(os.path.join(bank_dir, "faiss_values.pt")).numpy()
                print(f"[FAISS] 已加载: {self.faiss_index.ntotal} 向量")
            else:
                print(f"[FAISS] ⚠️ 索引文件不存在: {idx_path}")
            # CONCH 加载
            conch_ckpt = getattr(model_params, "conch_checkpoint",
                r"C:\Users\felix\.cache\huggingface\hub\models--MahmoodLab--CONCH\snapshots\f9ca9f877171a28ade80228fb195ac5d79003357\pytorch_model.bin")
            if os.path.exists(conch_ckpt):
                from conch.open_clip_custom import create_model_from_pretrained
                self.conch_model, self.conch_preprocess = create_model_from_pretrained(
                    'conch_ViT-B-16', checkpoint_path=conch_ckpt, device='cuda')
                self.conch_model.eval()
                for p in self.conch_model.parameters():
                    p.requires_grad = False
                print("[CONCH] 已加载（冻结）")

        self.task_emb_proj = None
        self._proj_params_registered = False

        # ===== E2 风格条件 CFG 化: 训练期 null-style dropout + 推理期风格引导 =====
        self.style_dropout = getattr(model_params, "style_dropout", 0.0)
        self.style_cfg_scale = getattr(model_params, "style_cfg_scale", 1.0)

        # ===== E3: t 门控 LPIPS(Alex v0.1) 感知损失 on x0_hat (与评估指标同骨干) =====
        self.lpips_weight = getattr(model_params, "lpips_weight", 0.0)
        self.lpips_start_step = getattr(model_params, "lpips_start_step", 2000)
        self.lpips_ramp_steps = getattr(model_params, "lpips_ramp_steps", 1000)
        self.lpips_t_max = getattr(model_params, "lpips_t_max", 400)
        self._lpips_holder = {}  # 普通字典: 不入 state_dict/optimizer/checkpoint

        # ===== 特征缓存查表(可选, 默认关): 预计算 UNI token + CONCH 查询向量, 训练任务0时查表 =====
        # 约束: 仅适用于输入侧无增广(geo_aug/photometric_aug 关)的配置; 任务1(输入=H&E自身)不走缓存
        self.use_feat_cache = getattr(model_params, "use_feat_cache", False)
        self.feat_cache_dir = getattr(model_params, "feat_cache_dir", None)
        self._feat_cache = None   # 惰性加载: {'idx': {(label,name): row}, 'uni': fp16, 'q': fp16}
        self._feat_keys = None    # runner 每 batch 注入 [(organ_label, cond_name), ...]; None=实时编码

        # ===== E4: capped PatchGAN (realism-only 分布匹配, 攻 FID) =====
        # 5 项审计修复: ① opt_idx 分派由 runner 做 ② D 相位先 zero_grad 清脏梯度(runner)
        # ③ s.detach() ④ 仅新鲜训练(不 resume 旧 optimizer ckpt) ⑤ 占位调度器 LambdaLR(e:1.0)
        self.use_patch_d = getattr(model_params, "use_patch_d", False)
        self.d_start = getattr(model_params, "d_start", 2500)
        self.d_tmax = getattr(model_params, "d_tmax", 400)
        self.d_r = getattr(model_params, "d_r", 0.3)
        self.d_ramp = getattr(model_params, "d_ramp", 500)
        if self.use_patch_d:
            self.patch_d = PatchDiscriminator()
            self._d_cache = None  # (x0_hat.detach(), x0) 供 D 相位(opt_idx=1)复用

        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        # ===== 三维无监督 Loss + Kendall 动态加权 配置 =====
        # 控制开关: 只有 use_uni=True 时才启用(因为三维 loss 依赖 UNI 特征)
        self.use_aux_loss = self.use_uni
        # warm-up: 前 warmup_steps 步只用 L1(学长原版), 稳定训练
        self.warmup_steps = getattr(model_params, "warmup_steps", 1000)
        # HED 染色浓度损失（Beer-Lambert 分离 H/E 后 L1，可微，攻 FID 根因）
        self.stain_weight = getattr(model_params, "stain_weight", 0.0)
        self.stain_start_step = getattr(model_params, "stain_start_step", 500)
        # 检索鲁棒训练: 训练期检索丢弃 top-1(自身), 模拟推理时目标不在库内的近似检索;
        # 仅 self.training 时生效, 推理行为不变。默认 False = 行为零变化
        self.self_exclude = getattr(model_params, "self_exclude", False)
        # 随机检索诊断(eval 用): retrieval_mode='random' 时检索结果替换为库内随机 8 向量,
        # 查询仍照常计算——唯一变量是"检索到什么"。默认 '' = 行为零变化
        self.retrieval_mode = getattr(model_params, "retrieval_mode", "")
        self._retrieval_rng = np.random.default_rng(1234) if self.retrieval_mode == "random" else None
        # Ruifrok 染色矩阵（行 = H/E/D 在 OD 空间的 RGB 向量），与 skimage rgb2hed 同源
        self.register_buffer(
            "stain_matrix_inv",
            torch.inverse(torch.tensor([[0.65, 0.70, 0.29],
                                        [0.07, 0.99, 0.11],
                                        [0.27, 0.57, 0.78]])), persistent=False)
        # staged loss 引入(数据驱动: FID 发现染色是主因, manifold 先引入)
        # 三 loss 的"开始生效步数": manifold(主因)最早, selfconsist 次之
        # struct 默认关闭(如无必要勿增实体): 数据显示结构只掉 9%, L1+manifold 已足够保结构
        # 想做消融"+struct"实验时, config 里设 aux_loss_start_steps.struct 为具体步数即可
        self.aux_loss_start_steps = getattr(
            model_params, "aux_loss_start_steps",
            {"manifold": self.warmup_steps,         # 主因(染色), warm-up 结束即引入
             "selfconsist": self.warmup_steps + 1000,  # 次要, 延 1000 步
             "struct": 10**9}                           # 默认关闭(如无必要勿增实体); 消融时改小
        )
        # config 里的 dict 会被 dict2namespace 转成 Namespace，统一转回 dict
        if not isinstance(self.aux_loss_start_steps, dict):
            self.aux_loss_start_steps = vars(self.aux_loss_start_steps)
        # EMA 归一化: 每个 loss 除以自己的滑动平均, 让量级对齐到 ~1
        self.loss_ema_decay = 0.99
        for name in ["manifold", "struct", "selfconsist"]:
            self.register_buffer(f"ema_{name}", torch.tensor(1.0))
        # 零阈值自门控(manifold_gate='self'): L_manifold 权重 g=clamp(cos_ema,0,1),
        # cos_ema = 输出/真值在 UNI 空间余弦的 EMA —— 就绪度由模型自判, 跨训练长度免调参。
        # 默认 '' = 不门控, 行为零变化
        self.manifold_gate = getattr(model_params, "manifold_gate", "")
        self.cos_ema = None  # 运行态标量(不入 checkpoint, 断点重启后数百步内重爬)
        # Kendall 不确定性加权: log σ² 是可学习参数(初始 0 → σ²=1 → λ=0.5)
        if self.use_aux_loss:
            self.log_vars = nn.Parameter(torch.zeros(3))  # [manifold, selfconsist, struct]
        # L_manifold 的训练集 UNI 特征统计(训练前预算, 缓冲区)
        self.register_buffer("uni_feat_mean", None)  # 训练前用 compute_uni_feat_stats 填充
        self.register_buffer("uni_feat_cov_inv", None)

    def extract_uni_feat(self, he_image):
        """用 UNI 提取 H&E 图的 patch token。UNI 冻结,无梯度。
        he_image: [B,3,H,W] H&E 图(任何范围,内部预处理)
        返回: [B,197,1024] patch token, 或 None(未启用 UNI)
        """
        if not self.use_uni or self.uni_encoder is None:
            return None
        return self.uni_encoder(he_image)

    def extract_uni_cls(self, he_image):
        """提取 CLS token(用于 L_manifold 流形约束)。"""
        if not self.use_uni or self.uni_encoder is None:
            return None
        return self.uni_encoder(he_image, return_cls=True)  # [B,1024]

    def extract_f_spatial(self, pi_image):
        """结构通路：UNI(PI) → DomainProjector → f_spatial。
        训练(无GRL分支)和推理共用此方法，保证特征空间一致。
        """
        if not self.use_uni or self.uni_encoder is None:
            return None
        raw = self.uni_encoder._extract_raw_tokens(pi_image)  # [B,197,1024]
        if self.use_grl and self.domain_projector is not None:
            return self.domain_projector(raw)
        return raw

    @torch.no_grad()
    def compute_uni_feat_stats(self, he_images):
        """训练前预算训练集 H&E 的 UNI CLS 特征均值。
        he_images: DataLoader 或 tensor 列表
        调用后, self.uni_feat_mean 被填充, L_manifold 生效。
        """
        if not self.use_aux_loss:
            return
        feats = []
        device = next(self.parameters()).device
        for img in he_images:
            if isinstance(img, (list, tuple)): img = img[0]
            if img.dim() == 3: img = img.unsqueeze(0)
            feats.append(self.extract_uni_cls(img.to(device)).cpu())
        all_feats = torch.cat(feats, dim=0)  # [N, 1024]
        self.uni_feat_mean = all_feats.mean(0).to(device)
        print(f"[L_manifold] UNI 特征均值已计算: 基于 {len(all_feats)} 张 H&E, shape={tuple(self.uni_feat_mean.shape)}")

    def _l_stain(self, x0_recon, x0_gt):
        """HED 染色浓度损失：Beer-Lambert 光学密度分离 H/E 通道后 L1。
        可微（pred 分支梯度完整），直接约束染色分布 —— FID 的根因。
        pred/gt 均为 [-1,1] RGB。返回 H、E 两通道的平均 L1。
        """
        def to_stain(img):
            i01 = ((img.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(min=1e-3)  # [B,3,H,W]
            od = -torch.log(i01)                                        # 光学密度
            # 浓度 c = od @ inv(S)：o_r = Σ_s c_s S[s,r] → c = od @ S^{-1}
            return torch.einsum("brhw,rs->bshw", od, self.stain_matrix_inv.to(od.dtype))

        c_pred = to_stain(x0_recon)
        with torch.no_grad():
            c_gt = to_stain(x0_gt)
        # 通道 0=H(核), 1=E(质)：各算 L1 后平均。DAB 通道舍弃。
        loss_h = (c_pred[:, 0] - c_gt[:, 0]).abs().mean()
        loss_e = (c_pred[:, 1] - c_gt[:, 1]).abs().mean()
        return loss_h + loss_e

    def _l_manifold(self, x0_recon, x0_gt=None):
        """实例级流形对齐（Foundation Perceptual Loss）：
        L = 1 - cos( UNI(x0_recon), UNI(x0_gt) )
        旧版均值对齐已废弃：uni_feat_mean 从未预计算(恒0)且 no_grad 无梯度——双重死亡。
        新版：pred 分支走 extract_cls_with_grad(梯度穿过冻结 UNI 回传生成器)，
        gt 分支 no_grad + detach 只当靶子。x0_recon 传入前勿 detach。
        manifold_gate='self' 时零阈值自门控：g=clamp(cos_ema,0,1) 由调用方乘入权重,
        cos_ema<0.1 时直接早退(省 UNI 带梯度前向, 纯算力优化不改数值语义)。
        返回 (loss, active)：early-exit 时 active=False, 调用方跳过 EMA/加权。
        """
        if not self.use_uni or self.uni_encoder is None or x0_gt is None:
            return torch.tensor(0.0, device=x0_recon.device), False
        with torch.no_grad():
            z_gt = self.uni_encoder(x0_gt, return_cls=True)
            if self.manifold_gate == "self":
                z_probe = self.uni_encoder(x0_recon.clamp(-1.0, 1.0).detach(), return_cls=True)
                cos_probe = torch.nn.functional.cosine_similarity(z_probe, z_gt, dim=-1).mean().detach()
                self.cos_ema = cos_probe if self.cos_ema is None else \
                    self.loss_ema_decay * self.cos_ema + (1 - self.loss_ema_decay) * cos_probe
                if float(self.cos_ema.clamp(0.0, 1.0)) < 0.1:  # 未就绪: 门≈关, 跳过带梯度前向
                    return torch.tensor(0.0, device=x0_recon.device), False
        z_pred = self.uni_encoder.extract_cls_with_grad(x0_recon.clamp(-1.0, 1.0))
        cos = torch.nn.functional.cosine_similarity(z_pred, z_gt, dim=-1)
        return (1.0 - cos).mean(), True

    def _l_struct(self, x0_recon, x0):
        """L_struct: Sobel 边缘 L1(管细胞核结构, 数据显示结构本不崩,权重会自动小)。
        x0_recon: 生成图 [B,3,H,W], x0: 真值/输入(同尺寸)
        """
        sobel = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],
                             device=x0_recon.device).view(1,1,3,3)
        sobel_xy = torch.cat([sobel, sobel.transpose(2,3)], dim=0)  # [2,1,3,3]
        def edge(img):
            gray = img.mean(1, keepdim=True)  # [B,1,H,W] 转灰度
            gx = torch.nn.functional.conv2d(gray, sobel, padding=1)
            gy = torch.nn.functional.conv2d(gray, sobel.transpose(2,3), padding=1)
            return torch.sqrt(gx**2 + gy**2 + 1e-8)
        return (edge(x0_recon) - edge(x0)).abs().mean()

    def _l_self_consist(self, x0_recon):
        """L_self_consist: 自一致性(0,因为这里简化为只算 x0_recon 的 TV 正则)。
        真正的"两次采样一致性"成本高(多一次完整 forward), 这里用 TV 正则近似:
        鼓励生成图平滑、避免跨域时的高频伪影。
        """
        # Total Variation 正则(近似自一致性)
        tv_h = (x0_recon[:, :, 1:, :] - x0_recon[:, :, :-1, :]).abs().mean()
        tv_w = (x0_recon[:, :, :, 1:] - x0_recon[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    def forward(self, x, y, context=None, current_task_id=0, organ_label=None, **kwargs):
        if self.training:
            self.step_counter += 1

        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        batch_size, _, height, width = x.shape

        if context is not None and context.shape[1] == self.channels:
            task_ids = torch.full(
                (batch_size,),
                current_task_id,
                device=x.device,
                dtype=torch.long,
            )
            task_emb = self.task_embed(task_ids)
            task_emb = task_emb.unsqueeze(2).unsqueeze(3).expand(
                -1, -1, height, width
            )

            if self.task_emb_proj is None:
                self.create_task_emb_proj(
                    context_channel=context.shape[1],
                    device=x.device,
                )

            context = context + self.task_emb_proj(task_emb)

        t = torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=x.device,
        ).long()

        # ===== UNI 特征提取 + GRL 域适应 =====
        # uni_input_mode='pi': UNI 吃 PI(y) → 训练+推理一致
        # uni_input_mode='he': UNI 吃 H&E(x) → 旧设计，推理时不注入
        _uni_img = y if self.uni_input_mode == 'pi' else x
        # 特征缓存: 仅训练期任务0(输入=PI原图,与缓存口径一致); GRL 相位/任务1 走实时编码
        cached = None
        if (self.use_feat_cache and self.training and current_task_id == 0
                and self._feat_keys is not None
                and not (self.use_grl and self.step_counter.item() >= self.adv_start_step)):
            cached = self.lookup_feat(self._feat_keys)
        domain_logits = None
        if self.use_uni:
            if self.training and self.use_grl and organ_label is not None and self.step_counter.item() >= self.adv_start_step:
                grl_lambda = get_dann_lambda(
                    self.step_counter.item() - self.adv_start_step,
                    self.grl_total_steps, self.grl_gamma
                )
                uni_feat, domain_logits = self.uni_encoder.forward_domain_invariant(
                    _uni_img, self.domain_projector, self.grl, self.domain_discriminator, grl_lambda
                )
            elif cached is not None:
                uni_feat = cached[0]                       # 查表: 跳过 UNI 前向
            else:
                uni_feat = self.extract_f_spatial(_uni_img)
        else:
            uni_feat = None

        # ===== FAISS 检索 F_style (可选 null-style dropout, 仅 task 0; 'none' 模式整体跳过=免CONCH前向) =====
        f_style = None
        if ((self.use_uni or self.style_only) and self.faiss_index is not None
                and self.retrieval_mode != "none"):
            f_style = (self.retrieve_f_style_from_query(cached[1]) if cached is not None
                       else self.retrieve_f_style(y))  # CONCH 始终吃 PI; 缓存路径跳过前向只留检索
            # task 1 的检索查询是 H&E 对 PI 键(跨域), 不参与 dropout
            if (self.training and current_task_id == 0
                    and self.style_dropout > 0
                    and torch.rand(()).item() < self.style_dropout):
                f_style = None  # 无风格分支: 让权重学会"有/无风格"两种模式, 推理期可外推

        return self.p_losses(
            x0=x,
            y=y,
            context=context,
            t=t,
            current_task_id=current_task_id,
            uni_feat=uni_feat,
            domain_logits=domain_logits,
            organ_label=organ_label,
            f_spatial=uni_feat,  # 域不变特征同时充当 F_spatial
            f_style=f_style,
        )

    @torch.no_grad()
    def retrieve_f_style(self, pi_image):
        """用 CONCH 编码 PI 图 → FAISS 检索 → 返回 F_style [B, K, 512]。
        pi_image: [B,3,H,W] [-1,1] 范围
        """
        import faiss
        from PIL import Image as PILImage

        B = pi_image.shape[0]
        styles = []
        for b in range(B):
            # [-1,1] → [0,255] → PIL
            img = pi_image[b].cpu()
            img = ((img * 0.5 + 0.5) * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            pil = PILImage.fromarray(img)
            inp = self.conch_preprocess(pil).unsqueeze(0).cuda()

            feat = self.conch_model.encode_image(inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat_np = feat.cpu().numpy().astype(np.float32)

            # FAISS 检索 Top-K(自查询排除: 训练期搜 9 丢 top-1=自身; 随机诊断: 库内随机 8)
            if self._retrieval_rng is not None:
                I = self._retrieval_rng.integers(0, len(self.faiss_values), size=(1, 8))
            else:
                se = self.training and self.self_exclude
                _, I = self.faiss_index.search(feat_np, k=9 if se else 8)
                if se:
                    I = I[:, 1:]
            topk_values = self.faiss_values[I[0]]  # [8, 512]
            styles.append(torch.from_numpy(topk_values))

        return torch.stack(styles).cuda()  # [B, 8, 512]

    # ===== 特征缓存查表路径 =====
    def _ensure_feat_cache(self):
        """惰性加载 train+val 缓存并合并索引。不可用则关开关返回 None。"""
        if self._feat_cache is not None:
            return self._feat_cache
        if not (self.use_feat_cache and self.feat_cache_dir):
            return None
        uni_all, q_all, idx = [], [], {}
        cache_map = None
        for stage in ('train', 'val'):
            fp = os.path.join(self.feat_cache_dir, f'feat_cache_{stage}.pt')
            if not os.path.exists(fp):
                continue
            d = torch.load(fp, map_location='cpu')
            if cache_map is None:
                cache_map = d.get('organ_map')
            base = sum(u.shape[0] for u in uni_all)
            uni_all.append(d['uni'])
            q_all.append(d['conch_q'])
            for i, k in enumerate(d['keys']):
                idx[(int(k[0]), str(k[1]))] = base + i
        if not uni_all:
            print(f'[feat_cache] 警告: {self.feat_cache_dir} 无缓存文件, 回退实时编码', flush=True)
            self.use_feat_cache = False
            return None
        # 出处防护: 缓存的器官标签空间必须与当前训练配置一致(按路径基名比对, 跨机器路径无关)。
        # 器官表由 runner 注入(_train_organ_map); 防止单器官配置(肾=0)误用三器官缓存(肝=0)静默查错器官。
        cur = []
        for l, p in (getattr(self, '_train_organ_map', None) or []):
            cur.append((int(l), os.path.basename(str(p)).lower()))
        cand = sorted((int(l), os.path.basename(str(p)).lower()) for l, p in cache_map) if cache_map else None
        if not cand or cand != sorted(cur):
            print(f'[feat_cache] 警告: 缓存出处 {cand} 与当前配置器官 {sorted(cur)} 不一致, 回退实时编码', flush=True)
            self.use_feat_cache = False
            return None
        self._feat_cache = {'idx': idx, 'uni': torch.cat(uni_all), 'q': torch.cat(q_all)}
        print(f'[feat_cache] 已加载 {self._feat_cache["uni"].shape[0]} 条 (train+val), '
              f'查表可跳过 UNI/CONCH 前向', flush=True)
        return self._feat_cache

    def lookup_feat(self, keys):
        """全部命中返回 (uni fp32 cuda [B,197,1024], q fp32 cuda [B,512]); 任一未命中/无缓存返回 None(回退实时)。"""
        c = self._ensure_feat_cache()
        if c is None or not keys:
            return None
        try:
            rows = [c['idx'][(int(k[0]), str(k[1]))] for k in keys]
        except KeyError:
            return None
        r = torch.tensor(rows, dtype=torch.long)
        dev = next(self.parameters()).device
        return c['uni'][r].to(dev).float(), c['q'][r].to(dev).float()

    def retrieve_f_style_from_query(self, q):
        """缓存路径: 跳过 CONCH 前向, 直接用查询向量做 FAISS 检索。输出与 retrieve_f_style 同构 [B,8,512]。"""
        import faiss
        styles = []
        for b in range(q.shape[0]):
            feat = q[b:b + 1] / q[b:b + 1].norm(dim=-1, keepdim=True)
            feat_np = feat.detach().cpu().numpy().astype(np.float32)
            if self._retrieval_rng is not None:
                I = self._retrieval_rng.integers(0, len(self.faiss_values), size=(1, 8))
            else:
                se = self.training and self.self_exclude
                _, I = self.faiss_index.search(feat_np, k=9 if se else 8)
                if se:
                    I = I[:, 1:]
            styles.append(torch.from_numpy(self.faiss_values[I[0]]))
        return torch.stack(styles).cuda()  # [B, 8, 512]

    def _get_lpips(self):
        """一次性构造冻结 LPIPS(Alex v0.1) — 与评估指标同骨干, 普通字典持有不入 checkpoint。"""
        if "fn" not in self._lpips_holder:
            import lpips as _lpips
            fn = _lpips.LPIPS(net="alex", version="0.1")
            for p in fn.parameters():
                p.requires_grad_(False)
            fn.eval()
            self._lpips_holder["fn"] = fn.to(next(self.parameters()).device)
        return self._lpips_holder["fn"]

    def p_losses(
        self,
        x0,
        y,
        context,
        t,
        noise=None,
        current_task_id=0,
        uni_feat=None,
        domain_logits=None,
        organ_label=None,
        f_spatial=None,
        f_style=None,
    ):
        """训练损失: warm-up + staged loss 引入 + EMA 归一化 + Kendall 加权。

        数据驱动设计(基于 FID 发现):
          - 染色是跨域主因(FID +285%) → L_manifold 最先引入(主导)
          - 结构基本不崩(SSIM -9%) → L_struct 最后引入(次要)
          - warm-up 前 warmup_steps 步只用 L1, 稳定训练
        """
        noise = noise if noise is not None else torch.randn_like(x0)

        x_t, objective = self.q_sample(
            x0,
            y,
            t,
            noise,
            current_task_id=current_task_id,
        )
        objective_recon = self.denoise_fn(
            x_t,
            timesteps=t,
            context=context,
            uni_feat=uni_feat,
            f_spatial=f_spatial,
            f_style=f_style,
        )

        recloss = (objective - objective_recon).abs().mean()

        x0_recon = self.predict_x0_from_objective(
            x_t,
            y,
            t,
            objective_recon,
        )

        # ===== 主 loss: L1(学长原版, 始终启用) =====
        total_loss = recloss * self.task_weights[current_task_id]
        step = self.step_counter.item()
        log_aux = {}

        # ===== MoE 专家负载均衡辅助损失(解码器 FFN MoE 启用时) =====
        if self.use_moe:
            l_moe = self.denoise_fn.get_aux_loss()
            if self.training and torch.is_grad_enabled():
                total_loss = total_loss + self.moe_aux_weight * l_moe
            moe_p = self.denoise_fn.get_moe_stats()
            log_aux["moe_aux"] = float(l_moe.detach()) if l_moe.requires_grad else float(l_moe)
            log_aux["moe_p0"], log_aux["moe_p1"], log_aux["moe_p2"] = moe_p

        # ===== HED 染色浓度损失（攻 FID 根因，可微，独立于 aux 体系）=====
        if self.stain_weight > 0 and step >= self.stain_start_step and x0_recon.dim() == 4:
            l_stain = self._l_stain(x0_recon, x0_gt=x0)
            total_loss = total_loss + self.stain_weight * l_stain
            log_aux["l_stain_raw"] = float(l_stain.detach())

        # ===== E3: LPIPS(Alex) 感知损失 on x0_hat (t 门控 + task0 + fp32 孤岛) =====
        if (self.lpips_weight > 0 and current_task_id == 0
                and step >= self.lpips_start_step and x0_recon.dim() == 4):
            t_mask = t < self.lpips_t_max  # [B] bool, 命中概率≈0.40 (sin 调度下 t<400 时 σ_t≤0.15)
            if t_mask.any():  # batch=1 时约 60% 步为空掩码, 必须守卫(空张量 mean 会产 NaN)
                ramp = min(1.0, (step - self.lpips_start_step) / max(1, self.lpips_ramp_steps))
                with torch.autocast(device_type=x0_recon.device.type, enabled=False):
                    x0_hat = x0_recon.float()[t_mask].clamp(-1.2, 1.2)  # 梯度保留(勿 detach)
                    gt = x0.float()[t_mask]                               # loader 张量无梯度
                    l_lpips = self._get_lpips()(x0_hat, gt).mean()
                total_loss = total_loss + self.lpips_weight * ramp * l_lpips
                log_aux["l_lpips_raw"] = float(l_lpips.detach())
                log_aux["lpips_frac"] = float(t_mask.float().mean())

        # ===== E4 G 相位: capped PatchGAN 对抗项 (realism-only, 梯度上限防 GRL 惨案) =====
        # 守卫 torch.is_grad_enabled(): val 每 50 步在 no_grad 下调用 loss_fn, autograd.grad 会崩
        if (self.use_patch_d and self.training and torch.is_grad_enabled()
                and step >= self.d_start and current_task_id == 0
                and t.max().item() < self.d_tmax):
            x0_hat_live = x0_recon  # 带梯度(勿 detach)
            d_fake = self.patch_d(x0_hat_live)
            l_g = -d_fake.mean()
            # 闭式上限: 对抗梯度在共享入口 x0_hat 处 ≤ d_r·ramp·w_task/√N,
            # 而 L1 梯度范数在该入口恰为 w_task/√N (L1-mean 的解析梯度) —— 比例被代数钉死
            g_img = torch.autograd.grad(l_g, x0_hat_live, retain_graph=True)[0]
            n_pix = x0_hat_live.numel()
            w_task = self.task_weights[current_task_id]
            ramp = min(1.0, (step - self.d_start) / max(1, self.d_ramp))
            s = (self.d_r * ramp * w_task / (math.sqrt(n_pix) * (g_img.norm() + 1e-12))).clamp(max=1.0)
            s = s.detach()  # 断二阶图: 缩放系数不参与 backward
            total_loss = total_loss + s * l_g
            # 供 D 相位(opt_idx=1)复用 —— detach, 无路径回 G
            self._d_cache = (x0_recon.detach().float(), x0.float())
            log_aux["d_scale"] = float(s)
            log_aux["d_fake"] = float(d_fake.mean().detach())

        # ===== 三维无监督 loss(warm-up + staged 引入 + EMA 归一化 + Kendall) =====
        if self.use_aux_loss and step >= self.warmup_steps and x0_recon.dim() == 4:
            # detach x0_recon 给 loss 用(不让三维 loss 的梯度回传污染 x0_recon 的计算图,
            # 避免 L1 和三维 loss 的梯度冲突; 三维 loss 主要通过各自约束独立作用)
            x0_recon_for_aux = x0_recon.detach().clamp(-1, 1)

            aux_losses = {}  # {name: (live_tensor, is_active)} — 只计算已激活项
            # (未激活的不再白白多跑一次 UNI 梯度检查点前向, 仅省算力, 数值行为不变)
            if step >= self.aux_loss_start_steps["manifold"]:
                lm, m_active = self._l_manifold(x0_recon, x0_gt=x0)
                aux_losses["manifold"] = (lm.to(recloss.device), m_active)

            # 2. L_self_consist(自一致性, 次要)
            if step >= self.aux_loss_start_steps["selfconsist"]:
                lc = self._l_self_consist(x0_recon_for_aux)
                aux_losses["selfconsist"] = (lc, True)

            # 3. L_struct(结构, 数据显示不重要, 最后引入)
            if step >= self.aux_loss_start_steps["struct"]:
                ls = self._l_struct(x0_recon_for_aux, x0.detach())
                aux_losses["struct"] = (ls, True)

            # EMA 归一化 + Kendall 加权（梯度经 live tensor 分子回传，EMA 分母 detach）
            for i, name in enumerate(["manifold", "selfconsist", "struct"]):
                if name not in aux_losses:
                    continue
                live_val, active = aux_losses[name]
                raw_val = float(live_val.detach())
                if not active or not np.isfinite(raw_val):
                    log_aux[f"l_{name[:5]}_raw"] = raw_val
                    continue

                # EMA 归一化(除以滑动平均,让量级对齐到~1)——分母 detach，分子保梯度
                ema_buf = getattr(self, f"ema_{name}")
                ema_new = self.loss_ema_decay * ema_buf + (1 - self.loss_ema_decay) * raw_val
                getattr(self, f"ema_{name}").copy_(ema_new)
                norm_val = live_val / (ema_buf.detach() + 1e-8)

                # Kendall 不确定性加权: λ_i = exp(-log σ²_i) / 2
                precision = torch.exp(-self.log_vars[i]) * 0.5  # = 1/(2σ²)
                # 零阈值自门控(manifold 专用): g=clamp(cos_ema,0,1), detach——只当调度员
                gate = 1.0
                if i == 0 and self.manifold_gate == "self" and self.cos_ema is not None:
                    gate = float(self.cos_ema.clamp(0.0, 1.0).detach())
                # 归一化后的 loss 项 + log σ² 正则(防止 σ² 跑大)
                total_loss = total_loss + precision * gate * norm_val + 0.5 * self.log_vars[i] * self.task_weights[current_task_id]

                log_aux[f"l_{name[:5]}_raw"] = raw_val
                log_aux[f"l_{name[:5]}_w"] = float(precision.detach())
                if i == 0 and self.cos_ema is not None:
                    log_aux["manifold_cos_ema"] = float(self.cos_ema)
                    log_aux["manifold_gate"] = gate

        # ===== GRL 对抗 loss（Re-BBDM 域适应）=====
        if domain_logits is not None and organ_label is not None:
            import torch.nn.functional as F
            # organ_label: [B] 器官类别索引（0=liver, 1=kidney, 2=brain）
            organ_label = organ_label.to(domain_logits.device).long()
            # label_smoothing=0.2: 防止 softmax 饱和,确保梯度永远不为 0
            loss_adv = F.cross_entropy(domain_logits, organ_label, label_smoothing=0.2)
            # λ 不乘在 loss 上（GRL backward 内部已处理）
            # adv_weight 控制对抗强度（避免 CE ~1.0 冲垮 L1 ~0.1）
            total_loss = total_loss + self.adv_weight * loss_adv

            # 梯度裁剪（防对抗 loss 暴涨导致梯度爆炸）
            # 在 backward 之后的 optimizer.step 之前由训练循环执行
            # 这里记录是否需要裁剪（flag 给训练循环）
            log_aux["needs_grad_clip"] = True

            # 监控判别器准确率
            with torch.no_grad():
                pred = domain_logits.argmax(dim=1)
                acc = (pred == organ_label).float().mean().item()
            log_aux["adv_loss"] = float(loss_adv.detach())
            log_aux["disc_acc"] = acc

        log_dict = {
            "loss": total_loss.detach(),
            "recloss": recloss.detach(),
            "x0_recon": x0_recon.detach().cpu(),
            **log_aux,
        }

        if self.training and step % 50 == 0:
            aux_str = " ".join(f"{k}={v:.3f}" for k, v in log_aux.items())
            print(
                f"[Step {step}] recloss={recloss.item():.4f} | total={total_loss.item():.4f} | {aux_str}"
            )

        return total_loss, log_dict

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError

        m_tminus = np.append(0, m_t[:-1])
        variance_t = 2.0 * (m_t - m_t**2) * self.max_var
        variance_tminus = np.append(0.0, variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * (
            (1.0 - m_t) / (1.0 - m_tminus)
        ) ** 2
        posterior_variance_t = (
            variance_t_tminus * variance_tminus / variance_t
        )

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("m_t", to_torch(m_t))
        self.register_buffer("m_tminus", to_torch(m_tminus))
        self.register_buffer("variance_t", to_torch(variance_t))
        self.register_buffer("variance_tminus", to_torch(variance_tminus))
        self.register_buffer("variance_t_tminus", to_torch(variance_t_tminus))
        self.register_buffer("posterior_variance_t", to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == "linear":
                midsteps = torch.arange(
                    self.num_timesteps - 1,
                    1,
                    step=-((self.num_timesteps - 1) / (self.sample_step - 2)),
                ).long()
                self.steps = torch.cat(
                    (midsteps, torch.tensor([1, 0]).long()),
                    dim=0,
                )
            elif self.sample_type == "cosine":
                steps = np.linspace(
                    start=0,
                    stop=self.num_timesteps,
                    num=self.sample_step + 1,
                )
                steps = (
                    (np.cos(steps / self.num_timesteps * np.pi) + 1.0)
                    / 2.0
                    * self.num_timesteps
                )
                self.steps = torch.from_numpy(steps).long()
            else:
                raise NotImplementedError
        else:
            self.steps = torch.arange(self.num_timesteps - 1, -1, -1)

    def create_task_emb_proj(self, context_channel, device):
        if self.task_emb_proj is None:
            self.task_emb_proj = nn.Conv2d(
                in_channels=self.task_embed_dim,
                out_channels=context_channel,
                kernel_size=1,
            ).to(device)

            if not self._proj_params_registered:
                self.register_parameter(
                    "task_emb_proj_weight",
                    self.task_emb_proj.weight,
                )
                if self.task_emb_proj.bias is not None:
                    self.register_parameter(
                        "task_emb_proj_bias",
                        self.task_emb_proj.bias,
                    )
                self._proj_params_registered = True

            self.task_emb_proj.apply(weights_init)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        self.task_embed.apply(weight_init)
        if self.task_emb_proj is not None:
            self.task_emb_proj.apply(weight_init)
        return self

    def get_parameters(self):
        params = itertools.chain(
            self.denoise_fn.parameters(),
            self.task_embed.parameters(),
        )
        if self.task_emb_proj is not None:
            params = itertools.chain(params, self.task_emb_proj.parameters())
        # 【2026-08-14 根因修复】GRL 组件 + Kendall log_vars 必须进优化器。
        # 历史 bug：此前从未加入 → 投影器(残差零初始化=恒等映射)与判别器
        # 冻结在随机初始化，对抗梯度在 _extract_raw_tokens().detach() 处断路，
        # GRL 在所有主训练中从未生效（证据：两次不同 adv_weight 的 checkpoint
        # 投影器权重 bit 级相同；adv_loss 恒约 ln(3)=1.099 = 随机分类器签名）。
        extra = []
        if self.use_grl and self.domain_projector is not None:
            extra += list(self.domain_projector.parameters())
            extra += list(self.domain_discriminator.parameters())
        if self.use_aux_loss and hasattr(self, "log_vars"):
            extra += [self.log_vars]
        if extra:
            params = itertools.chain(params, extra)
        return params

    def q_sample(self, x0, y, t, noise=None, current_task_id=0):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if current_task_id == 1:
            x0 = x0 + self.noise_level * noise

        x_t = (1.0 - m_t) * x0 + m_t * y + sigma_t * noise

        if self.objective == "grad":
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == "noise":
            objective = noise
        elif self.objective == "ysubx":
            objective = y - x0
        else:
            raise NotImplementedError

        return x_t, objective

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == "grad":
            x0_recon = x_t - objective_recon
        elif self.objective == "noise":
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (
                x_t - m_t * y - sigma_t * objective_recon
            ) / (1.0 - m_t)
        elif self.objective == "ysubx":
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError

        return x0_recon

    @torch.no_grad()
    def sample(
        self,
        y,
        context=None,
        clip_denoised=True,
        sample_mid_step=False,
        task_id=0,
    ):
        if context is not None and task_id is not None and self.task_emb_proj is not None:
            batch_size = y.shape[0]
            task_ids = torch.full(
                (batch_size,),
                task_id,
                device=y.device,
                dtype=torch.long,
            )
            task_emb = self.task_embed(task_ids)
            task_emb = task_emb.unsqueeze(2).unsqueeze(3).expand(
                -1, -1, y.shape[2], y.shape[3]
            )
            context = context + self.task_emb_proj(task_emb)

        # ===== 推理时 UNI 特征:用 y(布朗桥端点)初始化,后续 p_sample_loop 用 x0_recon 更新 =====
        # 注意: 推理时 y 通常是 PI 条件(无 H&E 真值)。UNI 对 PI 是 OOD,
        # 所以初始 uni_feat 只是粗略估计,后续逐步用模型预测的 x0_recon(H&E)自引导修正。
        # 若不想推理时注入,设 use_uni=False 或 sample 里 uni_update_interval=0。
        return self.p_sample_loop(
            y,
            context,
            clip_denoised,
            sample_mid_step,
        )

    def _denoise_cfg(self, x_t, t, context, uni_feat, f_spatial, f_style):
        """带风格 CFG 的去噪: obj = obj_u + s·(obj_c − obj_u)。
        仅推理期且 s≠1 且有风格条件时生效。objective='grad' 下 x0 = x_t − obj,
        对 objective 外推严格等价于对 x0 外推(线性), 与末端 clamp 不冲突。"""
        obj = self.denoise_fn(
            x_t,
            timesteps=t,
            context=context,
            uni_feat=uni_feat,
            f_spatial=f_spatial,
            f_style=f_style,
        )
        if (not self.training) and self.style_cfg_scale != 1.0 and f_style is not None:
            obj_u = self.denoise_fn(
                x_t,
                timesteps=t,
                context=context,
                uni_feat=uni_feat,
                f_spatial=f_spatial,
                f_style=None,
            )
            obj = obj_u + self.style_cfg_scale * (obj - obj_u)
        return obj

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False, uni_feat=None, f_spatial=None, f_style=None):
        batch_size = x_t.shape[0]
        device = x_t.device

        if self.steps[i] == 0:
            t = torch.full(
                (batch_size,),
                self.steps[i],
                device=device,
                dtype=torch.long,
            )
            objective_recon = self._denoise_cfg(
                x_t, t, context, uni_feat, f_spatial, f_style,
            )
            x0_recon = self.predict_x0_from_objective(
                x_t,
                y,
                t,
                objective_recon,
            )
            if clip_denoised:
                x0_recon.clamp_(-1.0, 1.0)
            return x0_recon, x0_recon

        t = torch.full(
            (batch_size,),
            self.steps[i],
            device=device,
            dtype=torch.long,
        )
        n_t = torch.full(
            (batch_size,),
            self.steps[i + 1],
            device=device,
            dtype=torch.long,
        )

        objective_recon = self._denoise_cfg(
            x_t, t, context, uni_feat, f_spatial, f_style,
        )
        x0_recon = self.predict_x0_from_objective(
            x_t,
            y,
            t,
            objective_recon,
        )

        if clip_denoised:
            x0_recon.clamp_(-1.0, 1.0)

        m_t = extract(self.m_t, t, x_t.shape)
        m_nt = extract(self.m_t, n_t, x_t.shape)
        var_t = extract(self.variance_t, t, x_t.shape)
        var_nt = extract(self.variance_t, n_t, x_t.shape)

        sigma2_t = (
            var_t
            - var_nt * (1.0 - m_t) ** 2 / (1.0 - m_nt) ** 2
        ) * var_nt / var_t
        # eta 支持: 浮点(全程恒定) 或 与 steps 对齐的数组(逐采样步动态, i=0 为最大t)
        _eta_i = self.eta[i] if isinstance(self.eta, (list, tuple)) else self.eta
        sigma_t = torch.sqrt(sigma2_t) * _eta_i
        noise = torch.randn_like(x_t)

        x_tminus_mean = (
            (1.0 - m_nt) * x0_recon
            + m_nt * y
            + torch.sqrt((var_nt - sigma2_t) / var_t)
            * (x_t - (1.0 - m_t) * x0_recon - m_t * y)
        )

        return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(
        self,
        y,
        context=None,
        clip_denoised=True,
        sample_mid_step=False,
    ):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        # ===== 特征预提取（固定，不随扩散步更新）=====
        # PI 模式：UNI(PI) → DomainProjector → f_spatial + FAISS f_style
        # H&E 模式：推理时不注入（H&E 不存在）
        if self.use_uni and self.uni_input_mode == 'pi':
            f_spatial = self.extract_f_spatial(y)
            f_style = (self.retrieve_f_style(y)
                       if (self.faiss_index is not None and self.retrieval_mode != "none") else None)
            uni_feat = f_spatial
        elif self.style_only and self.faiss_index is not None:
            f_spatial = None       # 无 UNI: 空间注入自动跳过(block 级 None 门控)
            f_style = self.retrieve_f_style(y)
            uni_feat = None
        else:
            f_spatial = None
            f_style = None
            uni_feat = None

        if sample_mid_step:
            imgs = [y]
            one_step_imgs = []

            for i in tqdm(
                range(len(self.steps)),
                desc="reverse sampling",
                total=len(self.steps),
            ):
                img, x0_recon = self.p_sample(
                    x_t=imgs[-1],
                    y=y,
                    context=context,
                    i=i,
                    clip_denoised=clip_denoised,
                    uni_feat=uni_feat,
                    f_spatial=f_spatial,
                    f_style=f_style,
                )
                imgs.append(img)
                one_step_imgs.append(x0_recon)

            return imgs, one_step_imgs

        img = y
        for i in tqdm(
            range(len(self.steps)),
            desc="reverse sampling",
            total=len(self.steps),
        ):
            img, x0_recon = self.p_sample(
                x_t=img,
                y=y,
                context=context,
                i=i,
                clip_denoised=clip_denoised,
                uni_feat=uni_feat,
                f_spatial=f_spatial,
                f_style=f_style,
            )

        return img

    def decode(self, x_latent, cond=True, normalize=None):
        return x_latent


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

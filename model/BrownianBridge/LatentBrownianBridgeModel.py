import itertools
import torch
import torch.nn as nn
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel

def disabled_train(self, mode=True):
    return self

class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        if self.condition_key == 'first_stage':
            self.context_channel = self._get_vqgan_encoder_out_channel()
        elif self.condition_key == 'SpatialRescaler':
            self.context_channel = model_config.CondStageParams.out_channels
        elif self.condition_key == 'nocond':
            self.context_channel = 0
        else:
            raise NotImplementedError

        if self.condition_key != 'nocond':
            self.create_task_emb_proj(context_channel=self.context_channel, device=torch.device("cpu"))

        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

        self.ori_latent_mean = None
        self.ori_latent_std = None
        self.cond_latent_mean = None
        self.cond_latent_std = None
        print("✅ LatentBrownianBridgeModel with CTSG Initialized.\n")

    def _get_vqgan_encoder_out_channel(self):
        module = self.vqgan.encoder
        while isinstance(module, nn.Sequential) or (hasattr(module, 'children') and len(list(module.children())) > 0):
            children = list(module.children())
            if not children: break
            module = children[-1]
        if isinstance(module, nn.Conv2d):
            return module.out_channels
        return 128

    def get_ema_net(self):
        return self

    def get_parameters(self):
        parent_params = super().get_parameters()
        if self.condition_key == 'SpatialRescaler':
            return itertools.chain(parent_params, self.cond_stage_model.parameters())
        return parent_params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None and self.condition_key == 'SpatialRescaler':
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x_img, x_cond_img, context=None, current_task_id=0, **kwargs):
        # 1. 基础编码
        with torch.no_grad():
            x_latent = self.encode(x_img, cond=False) # IHC
            x_cond_latent = self.encode(x_cond_img, cond=(current_task_id == 0)) # HE

        context = self.get_cond_stage_context(x_cond_img, current_task_id=current_task_id)

        # ==========================================================
        # ==========================================================
        teacher_context = None
        teacher_y = None

        if self.training and current_task_id == 0:
            # Teacher: Task 1 (IHC->IHC)
            teacher_context = self.get_cond_stage_context(x_img, current_task_id=1)
            with torch.no_grad():
                teacher_y = self.encode(x_img, cond=True)
        kwargs['teacher_y'] = teacher_y
        kwargs['teacher_context'] = teacher_context
        # 传递 teacher 数据和 kwargs 给父类
        return super().forward(
            x=x_latent,
            y=x_cond_latent,
            context=context,
            current_task_id=current_task_id,
            **kwargs # 允许通过 kwargs 传递 Loss 权重
        )

    def get_cond_stage_context(self, x_cond_img, current_task_id=0):
        if self.cond_stage_model is None: return None

        if self.condition_key == 'first_stage':
            raw_context = self.vqgan.encoder(x_cond_img)
        else:
            context_output = self.cond_stage_model(x_cond_img)
            raw_context = context_output[0] if isinstance(context_output, tuple) else context_output
            while isinstance(raw_context, tuple): raw_context = raw_context[0]

        model_device = self.task_embed.weight.device
        context = raw_context.to(model_device)

        target_h, target_w = 64, 64
        if (context.shape[2], context.shape[3]) != (target_h, target_w):
            context = torch.nn.functional.interpolate(context, size=(target_h, target_w), mode='bilinear', align_corners=False)

        b = context.shape[0]
        task_id_tensor = torch.tensor(current_task_id, device=model_device).repeat(b)
        task_emb = self.task_embed(task_id_tensor)
        task_emb = task_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, target_h, target_w)
        task_emb = self.task_emb_proj(task_emb)
        context = context + task_emb

        if self.condition_key == 'first_stage': context = context.detach()
        return context

    @torch.no_grad()
    def encode(self, x_img, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model_device = self.task_embed.weight.device
        x_img = x_img.to(model_device)
        x_latent = self.vqgan.encoder(x_img)
        if not self.model_config.latent_before_quant_conv:
            x_latent = self.vqgan.quant_conv(x_latent)

        if normalize:
            if cond:
                mean = self.cond_latent_mean.to(model_device)
                std = self.cond_latent_std.to(model_device)
            else:
                mean = self.ori_latent_mean.to(model_device)
                std = self.ori_latent_std.to(model_device)
            x_latent = (x_latent - mean) / std
        return x_latent

    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model_device = self.task_embed.weight.device
        if normalize:
            if cond:
                mean = self.cond_latent_mean.to(model_device)
                std = self.cond_latent_std.to(model_device)
            else:
                mean = self.ori_latent_mean.to(model_device)
                std = self.ori_latent_std.to(model_device)
            x_latent = x_latent * std + mean
        if self.model_config.latent_before_quant_conv:
            x_latent = self.vqgan.quant_conv(x_latent)
        x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
        img = self.vqgan.decode(x_latent_quant.to(model_device))
        return img.clamp(-1., 1.)

    @torch.no_grad()
    def sample(self, x_cond_img, clip_denoised=False, sample_mid_step=False, task_id=0):
        x_cond_latent = self.encode(x_cond_img, cond=(task_id == 0))
        context = self.get_cond_stage_context(x_cond_img, current_task_id=task_id)
        latent_result = super().sample(y=x_cond_latent, context=context, clip_denoised=clip_denoised, sample_mid_step=sample_mid_step, task_id=task_id)
        if sample_mid_step:
            out_imgs = [self.decode(latent.detach(), cond=False) for latent in latent_result[0]]
            one_step_imgs = [self.decode(latent.detach(), cond=False) for latent in latent_result[1]]
            return out_imgs, one_step_imgs
        else:
            return self.decode(latent_result, cond=False)

    @torch.no_grad()
    def sample_vqgan(self, x_img):
        model_device = self.task_embed.weight.device
        x_rec, _ = self.vqgan(x_img.to(model_device))
        return x_rec.clamp(-1., 1.)
# import itertools
# import torch
# import torch.nn as nn
# from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
# from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
# from model.VQGAN.vqgan import VQModel

# def disabled_train(self, mode=True):
#     return self

# class LatentBrownianBridgeModel(BrownianBridgeModel):
#     def __init__(self, model_config):
#         super().__init__(model_config)

#         self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
#         self.vqgan.train = disabled_train
#         for param in self.vqgan.parameters():
#             param.requires_grad = False
#         print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

#         if self.condition_key == 'first_stage':
#             self.context_channel = self._get_vqgan_encoder_out_channel()
#         elif self.condition_key == 'SpatialRescaler':
#             self.context_channel = model_config.CondStageParams.out_channels
#         elif self.condition_key == 'nocond':
#             self.context_channel = 0
#         else:
#             raise NotImplementedError

#         if self.condition_key != 'nocond':
#             self.create_task_emb_proj(context_channel=self.context_channel, device=torch.device("cpu"))

#         if self.condition_key == 'nocond':
#             self.cond_stage_model = None
#         elif self.condition_key == 'first_stage':
#             self.cond_stage_model = self.vqgan
#         elif self.condition_key == 'SpatialRescaler':
#             self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
#         else:
#             raise NotImplementedError

#         self.ori_latent_mean = None
#         self.ori_latent_std = None
#         self.cond_latent_mean = None
#         self.cond_latent_std = None
#         print("✅ LatentBrownianBridgeModel with CTSG Initialized.\n")

#     def _get_vqgan_encoder_out_channel(self):
#         module = self.vqgan.encoder
#         while isinstance(module, nn.Sequential) or (hasattr(module, 'children') and len(list(module.children())) > 0):
#             children = list(module.children())
#             if not children: break
#             module = children[-1]
#         if isinstance(module, nn.Conv2d):
#             return module.out_channels
#         return 128

#     def get_ema_net(self):
#         return self

#     def get_parameters(self):
#         parent_params = super().get_parameters()
#         if self.condition_key == 'SpatialRescaler':
#             return itertools.chain(parent_params, self.cond_stage_model.parameters())
#         return parent_params

#     def apply(self, weights_init):
#         super().apply(weights_init)
#         if self.cond_stage_model is not None and self.condition_key == 'SpatialRescaler':
#             self.cond_stage_model.apply(weights_init)
#         return self

#     def forward(self, x_img, x_cond_img, context=None, current_task_id=0, **kwargs):
#         # 1. 基础编码
#         with torch.no_grad():
#             x_latent = self.encode(x_img, cond=False) # IHC
#             x_cond_latent = self.encode(x_cond_img, cond=(current_task_id == 0)) # HE

#         context = self.get_cond_stage_context(x_cond_img, current_task_id=current_task_id)

#         # ==========================================================
#         # 🌟 创新点三：准备 Teacher 数据
#         # ==========================================================
#         teacher_context = None
#         teacher_y = None

#         if self.training and current_task_id == 0:
#             # Teacher: Task 1 (IHC->IHC)
#             teacher_context = self.get_cond_stage_context(x_img, current_task_id=1)
#             with torch.no_grad():
#                 teacher_y = self.encode(x_img, cond=True)

#         kwargs['teacher_context'] = teacher_context
#         kwargs['teacher_y'] = teacher_y

#         # 传递 teacher 数据和 kwargs 给父类
#         return super().forward(
#             x=x_latent,
#             y=x_cond_latent,
#             context=context,
#             current_task_id=current_task_id,
#             **kwargs  # ✅ 现在 kwargs 里装的是全新的潜空间 teacher 了
#         )

#     def get_cond_stage_context(self, x_cond_img, current_task_id=0):
#         if self.cond_stage_model is None: return None

#         if self.condition_key == 'first_stage':
#             raw_context = self.vqgan.encoder(x_cond_img)
#             # raw_context = self.encode(x_cond_img, cond=(current_task_id==0))
#         else:
#             context_output = self.cond_stage_model(x_cond_img)
#             raw_context = context_output[0] if isinstance(context_output, tuple) else context_output
#             while isinstance(raw_context, tuple): raw_context = raw_context[0]

#         model_device = self.task_embed.weight.device
#         context = raw_context.to(model_device)

#         target_h, target_w = 64, 64
#         if (context.shape[2], context.shape[3]) != (target_h, target_w):
#             context = torch.nn.functional.interpolate(context, size=(target_h, target_w), mode='bilinear', align_corners=False)

#         b = context.shape[0]
#         task_id_tensor = torch.tensor(current_task_id, device=model_device).repeat(b)
#         task_emb = self.task_embed(task_id_tensor)
#         task_emb = task_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, target_h, target_w)
#         task_emb = self.task_emb_proj(task_emb)
#         context = context + task_emb

#         if self.condition_key == 'first_stage': context = context.detach()
#         return context

#     @torch.no_grad()
#     def encode(self, x_img, cond=True, normalize=None):
#         normalize = self.model_config.normalize_latent if normalize is None else normalize
#         model_device = self.task_embed.weight.device
#         x_img = x_img.to(model_device)
#         x_latent = self.vqgan.encoder(x_img)
#         if not self.model_config.latent_before_quant_conv:
#             x_latent = self.vqgan.quant_conv(x_latent)

#         if normalize:
#             if cond:
#                 mean = self.cond_latent_mean.to(model_device)
#                 std = self.cond_latent_std.to(model_device)
#             else:
#                 mean = self.ori_latent_mean.to(model_device)
#                 std = self.ori_latent_std.to(model_device)
#             x_latent = (x_latent - mean) / std
#         return x_latent

#     def decode(self, x_latent, cond=True, normalize=None):
#         normalize = self.model_config.normalize_latent if normalize is None else normalize
#         model_device = self.task_embed.weight.device
#         if normalize:
#             if cond:
#                 mean = self.cond_latent_mean.to(model_device)
#                 std = self.cond_latent_std.to(model_device)
#             else:
#                 mean = self.ori_latent_mean.to(model_device)
#                 std = self.ori_latent_std.to(model_device)
#             x_latent = x_latent * std + mean
#         if self.model_config.latent_before_quant_conv:
#             x_latent = self.vqgan.quant_conv(x_latent)
#         # x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
#         # img = self.vqgan.decode(x_latent_quant.to(model_device))
#         img = self.vqgan.decode(x_latent.to(model_device))
#         return img.clamp(-1., 1.)

#     @torch.no_grad()
#     def sample(self, x_cond_img, clip_denoised=False, sample_mid_step=False, task_id=0):
#         x_cond_latent = self.encode(x_cond_img, cond=(task_id == 0))
#         context = self.get_cond_stage_context(x_cond_img, current_task_id=task_id)
#         latent_result = super().sample(y=x_cond_latent, context=context, clip_denoised=clip_denoised, sample_mid_step=sample_mid_step, task_id=task_id)
#         if sample_mid_step:
#             out_imgs = [self.decode(latent.detach(), cond=False) for latent in latent_result[0]]
#             one_step_imgs = [self.decode(latent.detach(), cond=False) for latent in latent_result[1]]
#             return out_imgs, one_step_imgs
#         else:
#             return self.decode(latent_result, cond=False)

#     @torch.no_grad()
#     def sample_vqgan(self, x_img):
#         model_device = self.task_embed.weight.device
#         x_rec, _ = self.vqgan(x_img.to(model_device))
#         return x_rec.clamp(-1., 1.)
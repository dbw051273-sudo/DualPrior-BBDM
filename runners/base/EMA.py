# # import torch.nn as nn
# #
# #
# # class EMA:
# #     def __init__(self, ema_decay):
# #         super().__init__()
# #         self.ema_decay = ema_decay
# #         self.backup = {}
# #         self.shadow = {}
# #
# #     def register(self, current_model: nn.Module):
# #         for name, param in current_model.named_parameters():
# #             if param.requires_grad:
# #                 self.shadow[name] = param.data.clone()
# #
# #     def reset_device(self, current_model: nn.Module):
# #         for name, param in current_model.named_parameters():
# #             if param.requires_grad:
# #                 self.shadow[name] = self.shadow[name].to(param.data.device)
# #
# #     def update(self, current_model: nn.Module, with_decay=True):
# #         for name, param in current_model.named_parameters():
# #             if param.requires_grad:
# #                 assert name in self.shadow
# #                 if with_decay:
# #                     new_average = (1.0 - self.ema_decay) * param.data + self.ema_decay * self.shadow[name]
# #                 else:
# #                     new_average = param.data
# #                 self.shadow[name] = new_average.clone()
# #
# #     def apply_shadow(self, current_model: nn.Module):
# #         for name, param in current_model.named_parameters():
# #             if param.requires_grad:
# #                 assert name in self.shadow
# #                 self.backup[name] = param.data
# #                 param.data = self.shadow[name]
# #
# #     def restore(self, current_model: nn.Module):
# #         for name, param in current_model.named_parameters():
# #             if param.requires_grad:
# #                 assert name in self.backup
# #                 param.data = self.backup[name]
# #         self.backup = {}
# import torch.nn as nn
#
#
# class EMA:
#     def __init__(self, ema_decay):
#         super().__init__()
#         self.ema_decay = ema_decay
#         self.backup = {}
#         self.shadow = {}
#
#     def register(self, current_model: nn.Module):
#         for name, param in current_model.named_parameters():
#             if param.requires_grad:
#                 # 最小化修改1：强制覆盖已有参数（确保新增参数被注册）
#                 self.shadow[name] = param.data.clone()  # 即使name已存在也更新
#
#     def reset_device(self, current_model: nn.Module):
#         for name, param in current_model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = self.shadow[name].to(param.data.device)
#
#     def update(self, current_model: nn.Module, with_decay=True):
#         for name, param in current_model.named_parameters():
#             if param.requires_grad:
#                 # 最小化修改2：增强错误提示，明确缺失的参数名
#                 assert name in self.shadow, f"EMA未注册参数: {name}（请检查register是否调用）"
#                 if with_decay:
#                     new_average = (1.0 - self.ema_decay) * param.data + self.ema_decay * self.shadow[name]
#                 else:
#                     new_average = param.data
#                 self.shadow[name] = new_average.clone()
#
#     def apply_shadow(self, current_model: nn.Module):
#         for name, param in current_model.named_parameters():
#             if param.requires_grad:
#                 # 同步增强错误提示
#                 assert name in self.shadow, f"EMA未注册参数: {name}"
#                 self.backup[name] = param.data
#                 param.data = self.shadow[name]
#
#     def restore(self, current_model: nn.Module):
#         for name, param in current_model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}
#
import torch.nn as nn


class EMA:
    """影子权重 EMA。shadow_device='cpu' 时影子驻留 CPU(16GB 显卡上 GPU 影子 +3GB 会 OOM):
    update 每 interval 步做一次 GPU→CPU 传输混合(~3GB, <1% 步时); apply_shadow/restore
    仅在显存空闲的评估场景使用; 训练期默认不换装(runner 按 EMA.apply_during_training 门控)。"""

    def __init__(self, ema_decay, shadow_device='cpu'):
        super().__init__()
        self.ema_decay = ema_decay
        self.shadow_device = shadow_device
        self.backup = {}
        self.shadow = {}

    def _to_shadow(self, tensor):
        if self.shadow_device == 'cpu':
            return tensor.detach().to('cpu', copy=True)
        return tensor.clone()

    def register(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self._to_shadow(param.data)

    def reset_device(self, current_model: nn.Module):
        if self.shadow_device == 'cpu':
            return  # CPU 影子无需迁移
        for name, param in current_model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.shadow[name].to(param.data.device)

    def update(self, current_model: nn.Module, with_decay=True):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                # 动态注册延迟创建的参数(如投影层), 首次只建影子不更新
                if name not in self.shadow:
                    self.shadow[name] = self._to_shadow(param.data)
                    continue

                src = param.data.cpu() if self.shadow_device == 'cpu' else param.data
                sh = self.shadow[name]
                if sh.device != src.device:  # 断点续训防御: 历史存档的影子可能混入异设备张量
                    sh = sh.to(src.device)
                if with_decay:
                    new_average = (1.0 - self.ema_decay) * src + self.ema_decay * sh
                else:
                    new_average = src
                self.shadow[name] = new_average if self.shadow_device == 'cpu' else new_average.clone()

    def apply_shadow(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"EMA未注册参数: {name}"
                if self.shadow_device == 'cpu':
                    self.backup[name] = param.data.detach().to('cpu', copy=True)
                else:
                    self.backup[name] = param.data
                param.data = self.shadow[name].to(param.device)

    def restore(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                if self.shadow_device == 'cpu':
                    param.data = self.backup[name].to(param.device)
                else:
                    param.data = self.backup[name]
        self.backup = {}
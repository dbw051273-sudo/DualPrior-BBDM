import copy
import torch
import torch.nn as nn


class SWAD:
    """
    SWAD: Stochastic Weight Averaging Densely (Cha et al., NeurIPS 2021)
    在损失平坦区（Loss Valley）进行密集均匀算术平均，寻找参数平坦极小值，提升未见目标域的泛化能力。
    - shadow_device='cpu'：平均权重常驻 CPU，零额外 GPU 显存开销；
    - 递归在线增量均值更新：\bar{\theta}_{n} = \bar{\theta}_{n-1} + (\theta - \bar{\theta}_{n-1}) / n
    - 能够同时平滑 floating-point parameters 与 buffers，并在非浮点 buffer（如 num_batches_tracked）上保持最新值；
    - 严格与 EMA 并存：EMA 捕获局部平滑，SWAD 捕获平坦质心，互相独立且在 checkpoint 中分别落盘。
    """

    def __init__(self, start_step=30000, update_interval=100, loss_threshold=0.045, shadow_device='cpu'):
        self.start_step = start_step
        self.update_interval = update_interval
        self.loss_threshold = loss_threshold
        self.shadow_device = shadow_device

        self.shadow = {}
        self.n_averaged = 0
        self.active = False
        self.actual_start_step = None

    def update(self, current_model: nn.Module, step: int, loss: float = 0.0):
        """
        更新 SWAD 权重
        :param current_model: 当前训练中的 PyTorch 模型
        :param step: 当前全局训练步数 (global_step)
        :param loss: 当前步骤的 loss（用于检测是否已收敛进入平坦谷底）
        :return: bool，是否成功触发了平均
        """
        if not self.active:
            # 当步数达到 start_step 且 loss 已经收敛进入 loss_threshold 平坦区时激活
            # 如果 loss_threshold 为 None 或 <=0，则纯步数门控
            loss_in_valley = (loss <= self.loss_threshold) if (self.loss_threshold is not None and self.loss_threshold > 0) else True
            if step >= self.start_step and loss_in_valley:
                self.active = True
                self.actual_start_step = step
            else:
                return False

        self.n_averaged += 1

        # 遍历 state_dict 可以同时平滑 parameters 和 buffers（如 SWA 规范）
        state_dict = current_model.state_dict()
        for name, val in state_dict.items():
            val_cpu = val.detach().to(self.shadow_device)
            if name not in self.shadow:
                self.shadow[name] = val_cpu.clone()
            else:
                if val.is_floating_point():
                    self.shadow[name].add_((val_cpu - self.shadow[name]) / float(self.n_averaged))
                else:
                    self.shadow[name].copy_(val_cpu)
        return True

    def state_dict(self):
        return {
            'shadow': self.shadow,
            'n_averaged': self.n_averaged,
            'active': self.active,
            'actual_start_step': self.actual_start_step,
            'start_step': self.start_step,
            'update_interval': self.update_interval,
            'loss_threshold': self.loss_threshold
        }

    def load_state_dict(self, state_dict):
        if state_dict is None:
            return
        self.shadow = state_dict.get('shadow', {})
        self.n_averaged = state_dict.get('n_averaged', 0)
        self.active = state_dict.get('active', False)
        self.actual_start_step = state_dict.get('actual_start_step', None)

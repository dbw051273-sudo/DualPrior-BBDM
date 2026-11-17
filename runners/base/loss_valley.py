"""SWAD loss-valley selection over dense post-update weight segments."""
from collections import deque
from math import isfinite

import torch


class LossValley:
    def __init__(self, config):
        self.start_step = config.start_step
        self.eval_freq = config.eval_freq
        self.update_interval = config.update_interval
        self.n_converge = config.n_converge
        self.n_tolerance = config.n_tolerance
        self.tolerance_ratio = config.tolerance_ratio
        self.window = deque(maxlen=self.n_converge)
        self.pending = deque()
        self.segment = {}
        self.segment_n = 0
        self.segment_first = None
        self.segment_last = None
        self.shadow = {}
        self.n_averaged = 0
        self.actual_start_step = None
        self.actual_end_step = None
        self.val_history = []
        self.threshold = None
        self.stopped = False
        self.finalized = False
        self.fallback = False

    @torch.no_grad()
    def collect(self, model, step):
        if self.stopped or step <= self.start_step or (step - self.start_step) % self.update_interval:
            return
        self.segment_n += 1
        if self.segment_first is None:
            self.segment_first = step
        self.segment_last = step
        for name, tensor in model.state_dict().items():
            if name not in self.segment:
                self.segment[name] = tensor.detach().clone()
            elif tensor.is_floating_point():
                self.segment[name].lerp_(tensor, 1 / self.segment_n)
            else:
                self.segment[name].copy_(tensor)

    def _accept(self, item):
        first, last, count, state, _ = item
        if self.n_averaged == 0:
            self.actual_start_step = first
            self.shadow = state
        else:
            fraction = count / (self.n_averaged + count)
            for name, tensor in state.items():
                if tensor.is_floating_point():
                    self.shadow[name].lerp_(tensor, fraction)
                else:
                    self.shadow[name].copy_(tensor)
        self.n_averaged += count
        self.actual_end_step = last

    def _admit_pending(self):
        if min(item[-1] for item in self.pending) > self.threshold:
            self.pending.clear()
            self.stopped = True
        else:
            self._accept(self.pending.popleft())

    def on_validation(self, step, loss):
        if not isfinite(loss) or not self.segment_n:
            raise ValueError(f'Invalid SWAD segment at {step}: validation loss={loss}')
        state = {key: value.to('cpu', copy=True) for key, value in self.segment.items()}
        item = (self.segment_first, self.segment_last, self.segment_n, state, loss)
        self.segment = {}
        self.segment_n = 0
        self.segment_first = self.segment_last = None
        self.val_history.append((step, loss))
        if self.threshold is None:
            self.window.append(item)
            if len(self.window) < self.n_converge:
                return
            losses = [segment[-1] for segment in self.window]
            if losses.index(min(losses)) == 0:
                self.threshold = (1 + self.tolerance_ratio) * sum(losses) / len(losses)
                self._accept(self.window.popleft())
                self.pending.extend(self.window)
                self.window.clear()
        else:
            self.pending.append(item)
        while not self.stopped and len(self.pending) >= self.n_tolerance:
            self._admit_pending()

    def finalize(self):
        if self.finalized:
            return
        if self.threshold is None:
            if not self.window:
                raise RuntimeError('SWAD needs at least one completed source-validation segment')
            self.fallback = True
            self._accept(self.window[-1])
            self.window.clear()
        while self.pending and not self.stopped:
            self._admit_pending()
        self.finalized = True

    def state_dict(self):
        return dict(vars(self), window=list(self.window), pending=list(self.pending))

    def load_state_dict(self, state, device):
        for key in ('start_step', 'eval_freq', 'update_interval', 'n_converge', 'n_tolerance', 'tolerance_ratio'):
            if state[key] != getattr(self, key):
                raise ValueError(f'SWAD resume configuration differs: {key}')
        self.__dict__.update(state)
        self.window = deque(state['window'], maxlen=self.n_converge)
        self.pending = deque(state['pending'])
        self.segment = {key: tensor.to(device) for key, tensor in state['segment'].items()}

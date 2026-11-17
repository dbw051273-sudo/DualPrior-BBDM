"""First-order MLDG update and source-domain batch sampler."""
import random
from collections import defaultdict
from contextlib import nullcontext

import torch


class OrganBatchSampler:
    """Keep batch size four and supply two examples of the held source."""

    def __init__(self, samples, offset):
        self.groups = defaultdict(list)
        for index, sample in enumerate(samples):
            self.groups[sample[2]].append(index)
        if sorted(self.groups) != [0, 1, 2] or any(len(g) < 2 for g in self.groups.values()):
            raise ValueError('FOMLDG requires labels 0, 1, 2 and at least two samples per source')
        self.offset = offset
        self.batches = len(samples) // 4

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            self.offset += 1
            held = self.offset % 3
            yield [i for label, group in self.groups.items()
                   for i in random.sample(group, 2 if label == held else 1)]


def first_order_step(model, train_loss, test_loss, inner_lr, beta, backup_device):
    """Populate outer gradients; the existing runner owns the optimizer step."""
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if any(p.grad is not None for p in params):
        raise RuntimeError('FOMLDG requires cleared gradients and no accumulation')
    # Cached autocast casts would otherwise reuse pre-update parameters.
    context = (torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False)
               if torch.is_autocast_enabled() else nullcontext())
    with context:
        loss_tr = train_loss()
        train_grad = torch.autograd.grad(loss_tr, params, allow_unused=True)
        original = [p.detach().to(backup_device, copy=True) if g is not None else None
                    for p, g in zip(params, train_grad)]
        counter = getattr(model, 'step_counter', None)
        counter_after_train = None if counter is None else counter.detach().clone()
        try:
            with torch.no_grad():
                for param, grad in zip(params, train_grad):
                    if grad is not None:
                        param.add_(grad, alpha=-inner_lr)
                if counter is not None:
                    counter.sub_(1)
            loss_te = test_loss()
            test_grad = torch.autograd.grad(loss_te, params, allow_unused=True)
        finally:
            with torch.no_grad():
                for param, saved in zip(params, original):
                    if saved is not None:
                        param.copy_(saved)
                if counter is not None:
                    counter.copy_(counter_after_train)
        for param, g_tr, g_te in zip(params, train_grad, test_grad):
            if g_te is not None:
                g_te.mul_(beta)
                param.grad = g_te if g_tr is None else g_tr.add_(g_te)
            else:
                param.grad = g_tr
    return loss_tr.detach(), loss_te.detach()

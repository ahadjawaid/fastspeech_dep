import torch.nn.functional as F
import torch
from utils import clip_mel

class TransformerScheduler():
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul, self.d_model = lr_mul, d_model
        self.n_warmup_steps, self.n_steps = n_warmup_steps, 0

    def step(self):
        self._update_learning_rate()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
    
    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def mae_loss(pred, target): return torch.abs(target - pred).mean()
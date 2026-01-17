import torch
import torch.optim
from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .utils import *

def euclidean_update(p, d_p, lr):
    p.data = p.data - lr * d_p
    return p.data

def poincare_grad(p, d_p):
    p_sqnorm = torch.clamp(torch.sum(p.data ** 2, dim=-1, keepdim=True), 0, 1-1e-5)
    d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p

def poincare_update(p, d_p, lr):
    v = -lr * d_p
    p.data = full_p_exp_map(p.data, v)
    return p.data


class RiemannianSGD(Optimizer):
    
    def __init__(self, params, lr=required, param_names=None, scheduler_cfg=None):
        defaults = dict(lr=lr)
        super(RiemannianSGD, self).__init__(params, defaults)
        self.param_names = param_names or []
        self.scheduler_cfg = scheduler_cfg or {}
        self._euclid_scheduler = None
        # Changed to epoch-based scheduler, no longer called each step
        if self.scheduler_cfg.get("enable", False):
            dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self._euclid_dummy_opt = torch.optim.SGD([dummy_param], lr=lr)
            self._euclid_scheduler = CosineAnnealingWarmRestarts(
                self._euclid_dummy_opt,
                T_0=int(self.scheduler_cfg.get("T0", 50)),  # Now represents epoch count
                T_mult=int(self.scheduler_cfg.get("Tmult", 2)),
                eta_min=float(self.scheduler_cfg.get("eta_min", 0.0)),
            )
        

    def step(self, lr=None):
        loss = None
        euclid_lr = None
        if self._euclid_scheduler is not None:
            euclid_lr = self._euclid_scheduler.get_last_lr()[0]
        name_idx = 0
        for group in self.param_groups:
            base_lr = lr if lr is not None else group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_name = self.param_names[name_idx] if name_idx < len(self.param_names) else None
                name_idx += 1
                if param_name in ["Eh.weight", "rvh.weight"]:
                    d_p = poincare_grad(p, d_p)
                    p.data = poincare_update(p, d_p, base_lr)
                else:
                    lr_to_use = euclid_lr if euclid_lr is not None else base_lr
                    p.data = euclidean_update(p, d_p, lr_to_use)
        # No longer call scheduler.step() each step
        # Changed to call scheduler_epoch_step() at end of epoch
        return loss
    
    def scheduler_epoch_step(self):
        """Called at end of each epoch to update learning rate"""
        if self._euclid_scheduler is not None:
            self._euclid_scheduler.step()
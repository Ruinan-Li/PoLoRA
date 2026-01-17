import torch

from .utils import p_log_map, p_exp_map


def clamp_to_ball(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    max_norm = 1.0 - eps
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    scale = torch.where(norm > max_norm, max_norm / (norm + eps), torch.ones_like(norm))
    return x * scale


def poincare_to_euclid(x: torch.Tensor) -> torch.Tensor:
    x = clamp_to_ball(x)
    return p_log_map(x)


def euclid_to_poincare(x: torch.Tensor) -> torch.Tensor:
    y = p_exp_map(x)
    return clamp_to_ball(y)


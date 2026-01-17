import torch

def artanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def p_exp_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    return torch.tanh(normv)*v/normv

def p_log_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10, 1-1e-5)
    return artanh(normv)*v/normv

def full_p_exp_map(x, v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    y = torch.tanh(normv/(1-sqxnorm)) * v/normv
    return p_sum(x, y)

def p_sum(x, y):
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5)
    dotxy = torch.sum(x*y, dim=-1, keepdim=True)
    numerator = (1+2*dotxy+sqynorm)*x + (1-sqxnorm)*y
    denominator = 1 + 2*dotxy + sqxnorm*sqynorm
    return numerator/denominator

def clamp_to_ball(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    max_norm = 1.0 - eps
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    scale = torch.where(norm > max_norm, max_norm / (norm + eps), torch.ones_like(norm))
    return x * scale

def p_lora_matmul(lora_A: torch.Tensor, lora_B: torch.Tensor, scaling: float) -> torch.Tensor:
    delta_tangent = torch.matmul(lora_B, lora_A)
    scaled_delta = scaling * delta_tangent
    delta_hyp = p_exp_map(scaled_delta)
    delta_hyp = clamp_to_ball(delta_hyp)
    
    return delta_hyp

import torch
import torch.nn as nn
import numpy as np
from .utils import (
    p_log_map, p_exp_map, p_sum, clamp_to_ball, p_lora_matmul
)


class RiemannianLoRAEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, r, scaling=0.1,
                 device=None, dtype=torch.double):
        super(RiemannianLoRAEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.r = r
        self.scaling = scaling
        
        self.register_buffer('base_weight', None)
        self._base_initialized = False
        
        self.lora_A = nn.Parameter(
            torch.zeros(r, embedding_dim, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(num_embeddings, r, device=device, dtype=dtype)
        )
        
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
    
    def set_base_weight(self, base_weight):
        if base_weight.size(0) != self.num_embeddings:
            raise ValueError(
                f"base_weight size mismatch: expect {self.num_embeddings}, got {base_weight.size(0)}"
            )
        base_weight = clamp_to_ball(base_weight).detach()
        if self.base_weight is None:
            self.base_weight = base_weight.clone()
        else:
            self.base_weight.copy_(base_weight)
        self._base_initialized = True
    
    def forward(self, indices):
        if self.base_weight is None or self.base_weight.numel() == 0 or not self._base_initialized:
            raise RuntimeError("Base weight must be set before forward pass")
        
        if indices.dim() == 0:
            x_hyp = self.base_weight[indices.item()].unsqueeze(0)
            squeeze_output = True
        else:
            x_hyp = self.base_weight[indices]
            squeeze_output = False
        
        x_hyp = clamp_to_ball(x_hyp)
        
        if indices.dim() == 0:
            lora_B_selected = self.lora_B[indices.item()].unsqueeze(0)
        else:
            lora_B_selected = self.lora_B[indices]
        
        delta_hyp = p_lora_matmul(self.lora_A, lora_B_selected, self.scaling)
        result = p_sum(x_hyp, delta_hyp)
        result = clamp_to_ball(result)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result

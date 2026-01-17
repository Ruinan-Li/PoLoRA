import numpy as np
import torch
import torch.nn as nn
from .utils import *


class MuRP(torch.nn.Module):
    def __init__(self, d, dim):
        super(MuRP, self).__init__()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device="cuda"))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device="cuda"))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device="cuda"))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device="cuda"))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        Ru = self.Wu[r_idx]
        rvh = self.rvh.weight[r_idx]


        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1, 
                        u/(torch.norm(u, 2, dim=-1, keepdim=True)-1e-5), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= 1, 
                        v/(torch.norm(v, 2, dim=-1, keepdim=True)-1e-5), v)
        rvh = torch.where(torch.norm(rvh, 2, dim=-1, keepdim=True) >= 1, 
                          rvh/(torch.norm(rvh, 2, dim=-1, keepdim=True)-1e-5), rvh)   
        u_e = p_log_map(u)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        v_m = p_sum(v, rvh)
        u_m = torch.where(torch.norm(u_m, 2, dim=-1, keepdim=True) >= 1, 
                          u_m/(torch.norm(u_m, 2, dim=-1, keepdim=True)-1e-5), u_m)
        v_m = torch.where(torch.norm(v_m, 2, dim=-1, keepdim=True) >= 1, 
                          v_m/(torch.norm(v_m, 2, dim=-1, keepdim=True)-1e-5), v_m)
        
        sqdist = (2.*artanh(torch.clamp(torch.norm(p_sum(-u_m, v_m), 2, dim=-1), 1e-10, 1-1e-5)))**2

        return -sqdist + self.bs[u_idx] + self.bo[v_idx]


class MuRPScorer(nn.Module):
    def __init__(self, num_relations, dim, device, dtype=torch.double, use_rvh=True):
        super(MuRPScorer, self).__init__()
        self.num_relations = num_relations
        self.dim = dim
        self.device = device
        self.dtype = dtype
        
        self.Wu = nn.Parameter(
            torch.tensor(
                np.random.uniform(-1, 1, (num_relations, dim)),
                dtype=dtype,
                device=device
            )
        )
        if use_rvh:
            self.rvh = nn.Parameter(
                torch.zeros(num_relations, dim, dtype=dtype, device=device)
            )
        else:
            self.rvh = None
        
        self.bs = None
        self.bo = None
    
    def set_bias(self, num_entities, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        
        self.bs = nn.Parameter(
            torch.zeros(num_entities, dtype=dtype, device=device)
        )
        self.bo = nn.Parameter(
            torch.zeros(num_entities, dtype=dtype, device=device)
        )
    
    def forward(self, u_emb, r_idx, v_emb, relation_emb=None, u_idx=None, v_idx=None):
        if u_emb.dim() == 3:
            batch_size, nneg_plus_one, dim = u_emb.shape
            u_emb = u_emb.view(-1, dim)
            v_emb = v_emb.view(-1, dim)
            r_idx_flat = r_idx.view(-1)
            reshape_output = True
        else:
            reshape_output = False
            r_idx_flat = r_idx
        
        u_norm = torch.norm(u_emb, 2, dim=-1, keepdim=True)
        u_emb = torch.where(
            u_norm >= 1,
            u_emb / (u_norm - 1e-5),
            u_emb
        )
        
        v_norm = torch.norm(v_emb, 2, dim=-1, keepdim=True)
        v_emb = torch.where(
            v_norm >= 1,
            v_emb / (v_norm - 1e-5),
            v_emb
        )
        
        Ru = self.Wu[r_idx_flat]
        if self.rvh is not None:
            rvh = self.rvh[r_idx_flat]
        else:
            if relation_emb is None:
                raise ValueError("relation_emb must be provided when rvh is None")
            rvh = None
        
        if relation_emb is not None:
            if relation_emb.dim() == 3:
                relation_emb = relation_emb.view(-1, relation_emb.size(-1))
            rvh = relation_emb
        
        rvh_norm = torch.norm(rvh, 2, dim=-1, keepdim=True)
        rvh = torch.where(
            rvh_norm >= 1,
            rvh / (rvh_norm - 1e-5),
            rvh
        )
        
        u_e = p_log_map(u_emb)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        v_m = p_sum(v_emb, rvh)
        
        u_m_norm = torch.norm(u_m, 2, dim=-1, keepdim=True)
        u_m = torch.where(
            u_m_norm >= 1,
            u_m / (u_m_norm - 1e-5),
            u_m
        )
        
        v_m_norm = torch.norm(v_m, 2, dim=-1, keepdim=True)
        v_m = torch.where(
            v_m_norm >= 1,
            v_m / (v_m_norm - 1e-5),
            v_m
        )
        
        diff = p_sum(-u_m, v_m)
        diff_norm = torch.clamp(
            torch.norm(diff, 2, dim=-1),
            1e-10,
            1 - 1e-5
        )
        sqdist = (2.0 * artanh(diff_norm)) ** 2
        
        score = -sqdist
        if self.bs is not None and u_idx is not None:
            if reshape_output:
                u_idx_flat = u_idx.view(-1)
                score = score + self.bs[u_idx_flat]
            else:
                score = score + self.bs[u_idx]
        
        if self.bo is not None and v_idx is not None:
            if reshape_output:
                v_idx_flat = v_idx.view(-1)
                score = score + self.bo[v_idx_flat]
            else:
                score = score + self.bo[v_idx]
        
        if reshape_output:
            score = score.view(batch_size, nneg_plus_one)
        
        return score
    
    def parameters(self):
        params = [self.Wu]
        if self.rvh is not None:
            params.append(self.rvh)
        if self.bs is not None:
            params.append(self.bs)
        if self.bo is not None:
            params.append(self.bo)
        return params


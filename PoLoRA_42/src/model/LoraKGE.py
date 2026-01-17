from collections import defaultdict
import os

import torch

from .BaseModel import BaseModel
from ..poincare.mapping import clamp_to_ball
from ..poincare.utils import p_sum, artanh, p_log_map
from ..poincare.riemann_lora import RiemannianLoRAEmbedding
from ..poincare.model import MuRPScorer

class LoraKGE(BaseModel):
    def __init__(self, args, kg) -> None:
        super(LoraKGE, self).__init__(args, kg)
        self.entity_lora = None
        self.relation_lora = None
        self.poincare_ent_embeddings = None
        self.poincare_rel_embeddings = None
        self.murp_scorer = None
        self.prev_poincare_ent_embeddings = None
        self.prev_poincare_rel_embeddings = None
        self.prev_relation_scaling = None
        self.prev_bs = None
        self.prev_bo = None
        self.entity_neighbors = {}
        self.relation_samples = {}
        self.prev_contains_new = False

    def store_old_parameters(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def switch_snapshot(self):
        curr_snapshot = int(self.args.snapshot)
        
        next_snapshot = curr_snapshot + 1
        if next_snapshot >= int(self.args.snapshot_num):
            return

        should_skip_init = (curr_snapshot == 0 and getattr(self, 'prev_contains_new', False))
        
        if not should_skip_init:
            if curr_snapshot == 0:
                self.initialize_from_poincare()
            else:
                self.update_poincare_embeddings()

            self.store_old_parameters()
            self.prev_poincare_ent_embeddings = clamp_to_ball(
                self.poincare_ent_embeddings.detach().clone()
            )
            self.prev_poincare_rel_embeddings = clamp_to_ball(
                self.poincare_rel_embeddings.detach().clone()
            )
            self.prev_contains_new = False
        else:
            self.store_old_parameters()

        self._prepare_graph_context(next_snapshot)

        entity_base = self._build_entity_base_tensor(next_snapshot)
        self._initialize_entity_lora(entity_base)
        num_ent_curr = self.kg.snapshots[curr_snapshot].num_ent
        num_ent_next = entity_base.size(0)
        num_new_entities = max(0, num_ent_next - num_ent_curr)

        current_rel = (
            self.prev_poincare_rel_embeddings.size(0)
            if self.prev_poincare_rel_embeddings is not None
            else 0
        )
        next_rel = self.kg.snapshots[next_snapshot].num_rel
        num_new_relations = max(0, next_rel - current_rel)
        poincare_dim = int(self.args.poincare_dim) if hasattr(self.args, "poincare_dim") else self.args.emb_dim
        relation_base = self._build_relation_base_tensor(next_snapshot, poincare_dim)
        self._initialize_relation_lora(relation_base)
        self.prev_poincare_rel_embeddings = clamp_to_ball(relation_base.detach().clone())
        if self.prev_relation_scaling is None:
            self.prev_relation_scaling = torch.ones(0, poincare_dim, device=self.args.device, dtype=torch.double)
        if self.prev_relation_scaling.size(0) < next_rel:
            extra = torch.ones(num_new_relations, poincare_dim, device=self.args.device, dtype=torch.double)
            self.prev_relation_scaling = torch.cat([self.prev_relation_scaling, extra], dim=0)

        num_relations = next_rel
        num_entities = num_ent_next
        self.murp_scorer = MuRPScorer(
            num_relations=num_relations,
            dim=poincare_dim,
            device=self.args.device,
            dtype=torch.double,
            use_rvh=False,
        )
        self.murp_scorer.set_bias(num_entities, device=self.args.device, dtype=torch.double)
        if self.prev_relation_scaling is not None:
            old_relations = min(self.prev_relation_scaling.size(0), self.murp_scorer.Wu.size(0))
            self.murp_scorer.Wu.data[:old_relations] = self.prev_relation_scaling[:old_relations].to(self.murp_scorer.Wu.data.device)
        if self.prev_bs is not None and self.murp_scorer.bs is not None:
            old_entities = min(self.prev_bs.size(0), self.murp_scorer.bs.size(0))
            self.murp_scorer.bs.data[:old_entities] = self.prev_bs[:old_entities].to(self.murp_scorer.bs.data.device)
        if self.prev_bo is not None and self.murp_scorer.bo is not None:
            old_entities = min(self.prev_bo.size(0), self.murp_scorer.bo.size(0))
            self.murp_scorer.bo.data[:old_entities] = self.prev_bo[:old_entities].to(self.murp_scorer.bo.data.device)
        self.murp_scorer.to(self.args.device)
        self.add_module('murp_scorer', self.murp_scorer)
    
    def snapshot_post_processing(self):
        if self.args.snapshot == 0:
            trainer = getattr(self, "poincare_trainer", None)
            if trainer is not None:
                ent_poincare, rel_poincare = trainer.get_embeddings()
                poincare_entities = trainer.entities
                poincare_relations = trainer.relations
                
                num_ent = self.kg.snapshots[0].num_ent
                num_rel = self.kg.snapshots[0].num_rel
                poincare_dim = ent_poincare.size(1)
                
                ent_remapped = torch.zeros(num_ent, poincare_dim, dtype=torch.double, device=self.args.device)
                for kg_ent_name, kg_ent_id in self.kg.entity2id.items():
                    if kg_ent_id < num_ent and kg_ent_name in trainer.entity_idxs:
                        poincare_ent_id = trainer.entity_idxs[kg_ent_name]
                        ent_remapped[kg_ent_id] = ent_poincare[poincare_ent_id]
                
                rel_remapped = torch.zeros(num_rel, poincare_dim, dtype=torch.double, device=self.args.device)
                Wu_remapped = torch.zeros(num_rel, poincare_dim, dtype=torch.double, device=self.args.device)
                
                for kg_rel_name, kg_rel_id in self.kg.relation2id.items():
                    if kg_rel_id < num_rel:
                        poincare_rel_name = kg_rel_name.replace('_inv', '_reverse')
                        if poincare_rel_name in trainer.relation_idxs:
                            poincare_rel_id = trainer.relation_idxs[poincare_rel_name]
                            rel_remapped[kg_rel_id] = rel_poincare[poincare_rel_id]
                            if hasattr(trainer.model, "Wu"):
                                Wu_remapped[kg_rel_id] = trainer.model.Wu[poincare_rel_id]
                
                bs_remapped = torch.zeros(num_ent, dtype=torch.double, device=self.args.device)
                bo_remapped = torch.zeros(num_ent, dtype=torch.double, device=self.args.device)
                
                if hasattr(trainer.model, "bs") and hasattr(trainer.model, "bo"):
                    for kg_ent_name, kg_ent_id in self.kg.entity2id.items():
                        if kg_ent_id < num_ent and kg_ent_name in trainer.entity_idxs:
                            poincare_ent_id = trainer.entity_idxs[kg_ent_name]
                            bs_remapped[kg_ent_id] = trainer.model.bs[poincare_ent_id]
                            bo_remapped[kg_ent_id] = trainer.model.bo[poincare_ent_id]
                
                self.poincare_ent_embeddings = clamp_to_ball(ent_remapped)
                self.poincare_rel_embeddings = clamp_to_ball(rel_remapped)
                self.prev_poincare_ent_embeddings = self.poincare_ent_embeddings.detach().clone()
                self.prev_poincare_rel_embeddings = self.poincare_rel_embeddings.detach().clone()
                self.prev_contains_new = True
                self.prev_relation_scaling = Wu_remapped.detach().clone()
                self.prev_bs = bs_remapped.detach().clone()
                self.prev_bo = bo_remapped.detach().clone()
                
                if hasattr(self.args, "logger") and self.args.logger is not None:
                    self.args.logger.info("Snapshot 0 embeddings correctly converted to KG index space via index mapping")
        else:
            if self.murp_scorer is not None:
                self.prev_relation_scaling = self.murp_scorer.Wu.detach().clone()
                if self.murp_scorer.bs is not None:
                    self.prev_bs = self.murp_scorer.bs.detach().clone()
                if self.murp_scorer.bo is not None:
                    self.prev_bo = self.murp_scorer.bo.detach().clone()
            
            self.prev_poincare_ent_embeddings = self.poincare_ent_embeddings.detach().clone()
            self.prev_poincare_rel_embeddings = self.poincare_rel_embeddings.detach().clone()
            self.prev_contains_new = True
            
            if hasattr(self.args, "logger") and self.args.logger is not None:
                self.args.logger.info(
                    "Snapshot %d: using loaded trained embeddings (%d entities), not recomputing",
                    self.args.snapshot,
                    self.poincare_ent_embeddings.size(0)
                )

    def has_poincare_embeddings(self):
        return self.poincare_ent_embeddings is not None and self.poincare_rel_embeddings is not None

    def initialize_from_poincare(self):
        trainer = getattr(self, "poincare_trainer", None)
        if trainer is None:
            return
        ent_poincare, rel_poincare = trainer.get_embeddings()
        ent_poincare = clamp_to_ball(ent_poincare.to(self.args.device).double())
        rel_poincare = clamp_to_ball(rel_poincare.to(self.args.device).double())
        self.poincare_ent_embeddings = ent_poincare.detach().clone()
        self.poincare_rel_embeddings = rel_poincare.detach().clone()
        self.prev_poincare_ent_embeddings = self.poincare_ent_embeddings.detach().clone()
        self.prev_poincare_rel_embeddings = self.poincare_rel_embeddings.detach().clone()
        self._initialize_entity_lora(self.poincare_ent_embeddings)
        self._initialize_relation_lora(self.poincare_rel_embeddings)

    def get_full_poincare_embeddings(self):
        if self.args.snapshot == 0:
            return (
                clamp_to_ball(self.poincare_ent_embeddings),
                clamp_to_ball(self.poincare_rel_embeddings),
            )

        if self.entity_lora is None:
            raise RuntimeError("Entity LoRA not initialized for snapshot > 0.")
        num_entities = self.entity_lora.num_embeddings
        ent_idx = torch.arange(num_entities, device=self.args.device)
        entity_embeddings = clamp_to_ball(self.entity_lora.forward(ent_idx))
        relation_embeddings = self.get_current_relation_embeddings()
        return entity_embeddings, relation_embeddings

    def update_poincare_embeddings(self):
        ent_embeddings, rel_embeddings = self.get_full_poincare_embeddings()
        self.poincare_ent_embeddings = ent_embeddings.detach().clone()
        self.poincare_rel_embeddings = rel_embeddings.detach().clone()

    def get_current_relation_embeddings(self):
        if self.relation_lora is not None:
            rel_idx = torch.arange(self.relation_lora.num_embeddings, device=self.args.device)
            return clamp_to_ball(self.relation_lora.forward(rel_idx))
        if self.murp_scorer is not None and self.murp_scorer.rvh is not None:
            return clamp_to_ball(self.murp_scorer.rvh)
        if self.poincare_rel_embeddings is not None:
            return clamp_to_ball(self.poincare_rel_embeddings)
        raise RuntimeError("Relation embeddings are not initialized.")

    def predict_poincare(self, head, relation, stage='Valid', chunk_size=1024):
        if stage != 'Test':
            num_ent = self.kg.snapshots[self.args.snapshot_valid].num_ent
        else:
            num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent

        ent_embed, rel_embed = self.get_full_poincare_embeddings()
        ent_embed = ent_embed[:num_ent]
        h = torch.index_select(ent_embed, 0, head)
        r = torch.index_select(rel_embed, 0, relation)
        pred_center = clamp_to_ball(p_sum(h, r))
        scores = []
        total_entities = ent_embed.size(0)
        batch_size = pred_center.size(0)
        for start in range(0, total_entities, chunk_size):
            end = min(total_entities, start + chunk_size)
            candidates = ent_embed[start:end]
            cand_size = candidates.size(0)
            pred_expand = pred_center.unsqueeze(1).expand(batch_size, cand_size, pred_center.size(-1))
            cand_expand = candidates.unsqueeze(0).expand(batch_size, cand_size, candidates.size(-1))
            diff = p_sum(-pred_expand, cand_expand)
            norm = torch.clamp(torch.norm(diff, dim=-1), 1e-10, 1 - 1e-5)
            sqdist = (2.0 * artanh(norm)) ** 2
            scores.append(-sqdist)
        return torch.cat(scores, dim=1)

    def _prepare_graph_context(self, snapshot_id):
        neighbors = defaultdict(list)
        relation_samples = defaultdict(list)
        triples = self.kg.snapshots[snapshot_id].train
        for h, r, t in triples:
            neighbors[h].append(t)
            neighbors[t].append(h)
            relation_samples[r].append((h, t))
        self.entity_neighbors = neighbors
        self.relation_samples = relation_samples

    def _build_entity_base_tensor(self, next_snapshot):
        num_entities = self.kg.snapshots[next_snapshot].num_ent
        poincare_dim = int(self.args.poincare_dim) if hasattr(self.args, "poincare_dim") else self.args.emb_dim
        base_tensor = torch.zeros(num_entities, poincare_dim, device=self.args.device, dtype=torch.double)
        
        source_embeddings = (
            self.prev_poincare_ent_embeddings 
            if self.prev_poincare_ent_embeddings is not None 
            else self.poincare_ent_embeddings
        )
        
        if source_embeddings is not None:
            prev_num = min(source_embeddings.size(0), num_entities)
            base_tensor[:prev_num] = clamp_to_ball(source_embeddings[:prev_num])
        neighbors = getattr(self, "entity_neighbors", {})
        old_limit = self.poincare_ent_embeddings.size(0) if self.poincare_ent_embeddings is not None else 0
        for ent_id in range(self.kg.snapshots[self.args.snapshot].num_ent, num_entities):
            idx = ent_id
            neighbor_ids = [
                n_id for n_id in neighbors.get(ent_id, []) if n_id < old_limit
            ]
            if neighbor_ids:
                init_vec = clamp_to_ball(base_tensor[neighbor_ids].mean(dim=0, keepdim=False))
            else:
                init_vec = clamp_to_ball(
                    1e-3 * torch.randn(poincare_dim, device=self.args.device, dtype=torch.double)
                )
            base_tensor[idx] = init_vec
        return base_tensor

    def _initialize_entity_lora(self, base_tensor):
        prev_lora = self.entity_lora
        num_entities = base_tensor.size(0)
        poincare_dim = base_tensor.size(1)
        ent_r = int(getattr(self.args, "ent_r", 64))
        entity_lora = RiemannianLoRAEmbedding(
            num_embeddings=num_entities,
            embedding_dim=poincare_dim,
            r=ent_r,
            scaling=1.0,
            device=self.args.device,
            dtype=torch.double,
        )
        entity_lora.set_base_weight(base_tensor)
        with torch.no_grad():
            if prev_lora is not None:
                if prev_lora.lora_A.shape == entity_lora.lora_A.shape:
                    entity_lora.lora_A.copy_(prev_lora.lora_A.to(entity_lora.lora_A.device))
                old_num = min(prev_lora.lora_B.size(0), entity_lora.lora_B.size(0))
                if old_num > 0:
                    entity_lora.lora_B[:old_num].copy_(prev_lora.lora_B[:old_num].to(entity_lora.lora_B.device))
            curr_snapshot = int(getattr(self.args, "snapshot", 0))
            total_snapshots = len(self.kg.snapshots)
            old_ent_count = (
                self.kg.snapshots[curr_snapshot].num_ent
                if curr_snapshot < total_snapshots
                else num_entities
            )
            old_ent_count = min(old_ent_count, num_entities)
            if old_ent_count < num_entities:
                neighbors = getattr(self, "entity_neighbors", {})
                for ent_id in range(old_ent_count, num_entities):
                    neighbor_ids = [
                        n_id
                        for n_id in neighbors.get(ent_id, [])
                        if n_id < old_ent_count
                    ]
                    if neighbor_ids:
                        neighbor_idx = torch.tensor(
                            neighbor_ids, device=self.args.device, dtype=torch.long
                        )
                        neighbor_mean = entity_lora.lora_B[neighbor_idx].mean(dim=0)
                        entity_lora.lora_B[ent_id].copy_(neighbor_mean)
                    else:
                        entity_lora.lora_B[ent_id].normal_(mean=0.0, std=1e-3)
        entity_lora = entity_lora.to(self.args.device)
        self.entity_lora = entity_lora

    def _build_relation_base_tensor(self, next_snapshot, poincare_dim):
        total_rel = self.kg.snapshots[next_snapshot].num_rel
        base_tensor = torch.zeros(total_rel, poincare_dim, device=self.args.device, dtype=torch.double)
        
        source_rel_embeddings = (
            self.prev_poincare_rel_embeddings
            if self.prev_poincare_rel_embeddings is not None
            else self.poincare_rel_embeddings
        )
        
        if source_rel_embeddings is not None:
            prev_rel = min(source_rel_embeddings.size(0), total_rel)
            base_tensor[:prev_rel] = clamp_to_ball(source_rel_embeddings[:prev_rel])
        relation_samples = getattr(self, "relation_samples", {})
        entity_base = clamp_to_ball(self.prev_poincare_ent_embeddings)
        base_rel = self.kg.snapshots[self.args.snapshot].num_rel
        for rel_id in range(base_rel, total_rel):
            samples = relation_samples.get(rel_id, [])
            translations = []
            for h, t in samples:
                if h < entity_base.size(0) and t < entity_base.size(0):
                    h_vec = entity_base[h].unsqueeze(0)
                    t_vec = entity_base[t].unsqueeze(0)
                    translations.append(clamp_to_ball(p_sum(-h_vec, t_vec)).squeeze(0))
            if translations:
                base_tensor[rel_id] = clamp_to_ball(torch.stack(translations).mean(dim=0))
            else:
                base_tensor[rel_id] = clamp_to_ball(
                    1e-3 * torch.randn(poincare_dim, device=self.args.device, dtype=torch.double)
                )
        return base_tensor

    def _initialize_relation_lora(self, base_tensor):
        prev_lora = self.relation_lora
        num_relations = base_tensor.size(0)
        poincare_dim = base_tensor.size(1)
        rel_r = int(getattr(self.args, "rel_r", 32))
        relation_lora = RiemannianLoRAEmbedding(
            num_embeddings=num_relations,
            embedding_dim=poincare_dim,
            r=rel_r,
            scaling=1.0,
            device=self.args.device,
            dtype=torch.double,
        )
        relation_lora.set_base_weight(base_tensor)
        if prev_lora is not None:
            with torch.no_grad():
                if prev_lora.lora_A.shape == relation_lora.lora_A.shape:
                    relation_lora.lora_A.copy_(prev_lora.lora_A.to(relation_lora.lora_A.device))
                old_rel = min(prev_lora.lora_B.size(0), relation_lora.lora_B.size(0))
                if old_rel > 0:
                    relation_lora.lora_B[:old_rel].copy_(prev_lora.lora_B[:old_rel].to(relation_lora.lora_B.device))
        relation_lora = relation_lora.to(self.args.device)
        self.relation_lora = relation_lora

    def ensure_lora_from_embeddings(self):
        if self.entity_lora is None and self.poincare_ent_embeddings is not None:
            self._initialize_entity_lora(clamp_to_ball(self.poincare_ent_embeddings))
        if self.relation_lora is None and self.poincare_rel_embeddings is not None:
            self._initialize_relation_lora(clamp_to_ball(self.poincare_rel_embeddings))

    def get_lora_state(self):
        state = {}
        if self.entity_lora is not None:
            state["entity_lora"] = self.entity_lora.state_dict()
        if self.relation_lora is not None:
            state["relation_lora"] = self.relation_lora.state_dict()
        return state

    def load_lora_state(self, state):
        if state is None:
            self.ensure_lora_from_embeddings()
            return
        ent_state = state.get("entity_lora")
        if ent_state is not None:
            if self.entity_lora is None:
                self.ensure_lora_from_embeddings()
            self.entity_lora.load_state_dict(ent_state, strict=False)
        rel_state = state.get("relation_lora")
        if rel_state is not None:
            if self.relation_lora is None:
                self.ensure_lora_from_embeddings()
            self.relation_lora.load_state_dict(rel_state, strict=False)

class TransE(LoraKGE):
    def __init__(self, args, kg) -> None:
        super(TransE, self).__init__(args, kg)
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    def predict(self, head, relation, stage='Valid'):
        if stage != 'Test':
            num_ent = self.kg.snapshots[self.args.snapshot_valid].num_ent
        else:
            num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
        
        if self.args.snapshot == 0:
            raise RuntimeError(
                "Snapshot 0 prediction should be handled by PoincareSnapshotTrainer, check calling flow."
            )

        if self.murp_scorer is None:
            raise RuntimeError("MuRP scorer not initialized for snapshot > 0")
        
        poincare_dim = int(self.args.poincare_dim) if hasattr(self.args, "poincare_dim") else self.args.emb_dim
        all_ent_emb, all_rel_emb = self.get_full_poincare_embeddings()
        h_emb = all_ent_emb[head]
        r_emb = all_rel_emb[relation]
        
        t_all_emb = all_ent_emb[:num_ent]
        
        batch_size = h_emb.size(0)
        chunk_size = 256
        scores = []
        
        for start in range(0, num_ent, chunk_size):
            end = min(num_ent, start + chunk_size)
            t_chunk = t_all_emb[start:end]
            
            h_expanded = h_emb.unsqueeze(1).expand(batch_size, end - start, -1)
            r_expanded = r_emb.unsqueeze(1).expand(batch_size, end - start, -1)
            t_expanded = t_chunk.unsqueeze(0).expand(batch_size, -1, -1)
            
            h_flat = h_expanded.contiguous().view(-1, poincare_dim)
            r_flat = r_expanded.contiguous().view(-1, poincare_dim)
            t_flat = t_expanded.contiguous().view(-1, poincare_dim)
            
            rel_indices = relation.unsqueeze(1).expand(batch_size, end - start).contiguous().view(-1)
            rel_chunk = r_emb.unsqueeze(1).expand(batch_size, end - start, -1)
            rel_chunk = rel_chunk.contiguous().view(-1, poincare_dim)
            head_indices = head.unsqueeze(1).expand(batch_size, end - start).contiguous().view(-1)
            tail_indices = (
                torch.arange(start, end, device=head.device, dtype=head.dtype)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .contiguous()
                .view(-1)
            )
            
            chunk_scores = self.murp_scorer.forward(
                h_flat,
                rel_indices,
                t_flat,
                relation_emb=rel_chunk,
                u_idx=head_indices,
                v_idx=tail_indices,
            )
            chunk_scores = chunk_scores.view(batch_size, end - start)
            scores.append(chunk_scores)
        
        score = torch.cat(scores, dim=1)
        
        return score

    def bce_loss(self, head, rel, tail, label=None, sample_weight=None, buffer_idx=None):
        if self.args.snapshot == 0:
            raise RuntimeError(
                "Snapshot 0 training should be handled by PoincareSnapshotTrainer, Euclidean Margin logic is disabled."
            )
        else:
            if self.murp_scorer is None:
                raise RuntimeError("MuRP scorer not initialized for snapshot > 0")
            
            all_ent_emb, all_rel_emb = self.get_full_poincare_embeddings()
            
            batch_size_total = head.size(0)
            neg_ratio = int(self.args.neg_ratio)
            batch_size = batch_size_total // (neg_ratio + 1)
            
            h_emb = all_ent_emb[head]
            r_emb = all_rel_emb[rel]
            t_emb = all_ent_emb[tail]
            
            scores = self.murp_scorer.forward(
                h_emb,
                rel,
                t_emb,
                relation_emb=r_emb,
                u_idx=head,
                v_idx=tail,
            )
            
            raw_label = label
            targets = (raw_label > 0).to(scores.dtype)
            
            loss_vec = self.bce_loss_fn(scores, targets)
            per_sample_loss = loss_vec

            if sample_weight is not None:
                weight = sample_weight.to(loss_vec.device).to(loss_vec.dtype)
                per_sample_loss = loss_vec * weight
                loss = per_sample_loss.sum() / loss_vec.size(0)
            else:
                loss = per_sample_loss.mean()

            pos_losses = None
            pos_buffer_idx = None
            if raw_label is not None:
                pos_mask = raw_label > 0
                if pos_mask.any().item():
                    pos_losses = per_sample_loss[pos_mask]
                    if buffer_idx is not None:
                        pos_buffer_idx = buffer_idx[pos_mask]

            return loss, pos_losses, pos_buffer_idx

    def loss(self, head, relation, tail=None, label=None, sample_weight=None, buffer_idx=None):
        return self.bce_loss(head, relation, tail, label, sample_weight=sample_weight, buffer_idx=buffer_idx)
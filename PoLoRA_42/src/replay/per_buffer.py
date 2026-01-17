import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass
class Experience:
    fact: Tuple[int, int, int]
    snapshot: int
    struct_importance: float
    loss_abs: float
    priority: float
    prob: float
    weight: float
    buffer_idx: int


class PERBuffer:
    """
    Mixed priority experience replay buffer: manages old samples, computes priorities and IS weights.
    """

    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.alpha = float(getattr(args, "per_alpha", 0.5))
        self.gamma = float(getattr(args, "per_gamma", 0.5))
        self.beta = float(getattr(args, "per_beta", 0.5))
        self.eps = float(getattr(args, "per_eps", 1e-3))
        self.buffer: List[Experience] = []
        self.struct_cache: Dict[int, Dict[Tuple[int, int], float]] = {}
        self.struct_default: Dict[int, float] = {}
        self.logger = getattr(args, "logger", None)

    def size(self) -> int:
        return len(self.buffer)

    def add_snapshot(self, snapshot_id: int, facts: Sequence[Tuple[int, int, int]]) -> None:
        if not facts:
            return
        struct_map = self._get_struct_importance(snapshot_id)
        default_val = self.struct_default.get(snapshot_id, 1.0)
        start_idx = len(self.buffer)
        for offset, fact in enumerate(facts):
            h, _, t = fact
            struct_key = (h, t)
            struct_val = struct_map.get(struct_key)
            if struct_val is None:
                struct_val = struct_map.get((t, h), default_val)
            exp = Experience(
                fact=fact,
                snapshot=snapshot_id,
                struct_importance=max(struct_val, self.eps),
                loss_abs=0.0,
                priority=0.0,
                prob=0.0,
                weight=1.0,
                buffer_idx=start_idx + offset,
            )
            self.buffer.append(exp)
        self._recompute_probabilities()

    def sample(self, num_samples: int) -> List[Dict[str, float]]:
        if num_samples <= 0 or not self.buffer:
            return []
        probs = [max(exp.prob, self.eps) for exp in self.buffer]
        total_prob = sum(probs)
        if total_prob <= 0:
            probs = [1.0 for _ in self.buffer]
        indices = random.choices(range(len(self.buffer)), weights=probs, k=num_samples)
        samples = []
        for idx in indices:
            exp = self.buffer[idx]
            samples.append(
                {
                    "fact": exp.fact,
                    "weight": float(exp.weight),
                    "buffer_idx": exp.buffer_idx,
                }
            )
        return samples

    def update(self, indices: Sequence[int], losses: Sequence[float]) -> None:
        if not indices or not losses:
            return
        changed = False
        for idx, loss in zip(indices, losses):
            if idx is None or idx < 0 or idx >= len(self.buffer):
                continue
            loss_value = float(abs(loss))
            if not math.isfinite(loss_value):
                continue
            exp = self.buffer[idx]
            if math.isclose(exp.loss_abs, loss_value, rel_tol=1e-6, abs_tol=1e-8):
                continue
            exp.loss_abs = loss_value
            changed = True
        if changed:
            self._recompute_probabilities()

    def recompute_all_losses(self, model, batch_size: int = 512) -> None:
        """
        Recompute losses for all samples in buffer using current model.
        Avoids loss_abs=0 at new snapshot startup relying only on structural weights.
        """
        if model is None or not self.buffer:
            return

        # Snapshot 0 model has no bce_loss/MuRP logic, force switch to >=1 branch
        orig_snapshot = int(getattr(self.args, "snapshot", 0))
        if orig_snapshot == 0:
            setattr(self.args, "snapshot", 1)
        device = getattr(self.args, "device", torch.device("cpu"))
        neg_ratio = int(getattr(self.args, "neg_ratio", 10))

        model.eval()
        with torch.no_grad():
            for start in range(0, len(self.buffer), batch_size):
                end = min(len(self.buffer), start + batch_size)
                heads, rels, tails = [], [], []
                labels, weights, buffer_idx_list = [], [], []
                for exp in self.buffer[start:end]:
                    ss_id = int(exp.snapshot)
                    num_ent = self.kg.snapshots[ss_id].num_ent
                    h, r, t = exp.fact
                    base_weight = float(exp.weight)

                    # Generate negative samples following "random replace head or tail" strategy
                    neg_h = torch.randint(0, num_ent, (neg_ratio,), device=device)
                    neg_t = torch.randint(0, num_ent, (neg_ratio,), device=device)
                    pos_h = torch.full((neg_ratio,), h, device=device, dtype=torch.long)
                    pos_t = torch.full((neg_ratio,), t, device=device, dtype=torch.long)
                    rand = torch.rand(neg_ratio, device=device)
                    head_ids = torch.where(rand > 0.5, pos_h, neg_h)
                    tail_ids = torch.where(rand > 0.5, neg_t, pos_t)

                    h_all = torch.cat([torch.tensor([h], device=device, dtype=torch.long), head_ids])
                    r_all = torch.full((neg_ratio + 1,), r, device=device, dtype=torch.long)
                    t_all = torch.cat([torch.tensor([t], device=device, dtype=torch.long), tail_ids])
                    lbl_all = torch.cat(
                        [torch.ones(1, device=device, dtype=torch.float), -torch.ones(neg_ratio, device=device)]
                    )
                    w_all = torch.full((neg_ratio + 1,), base_weight, device=device, dtype=torch.float)
                    bidx_all = torch.full((neg_ratio + 1,), int(exp.buffer_idx), device=device, dtype=torch.long)

                    heads.append(h_all)
                    rels.append(r_all)
                    tails.append(t_all)
                    labels.append(lbl_all)
                    weights.append(w_all)
                    buffer_idx_list.append(bidx_all)

                head = torch.cat(heads, dim=0)
                rel = torch.cat(rels, dim=0)
                tail = torch.cat(tails, dim=0)
                label = torch.cat(labels, dim=0)
                weight = torch.cat(weights, dim=0)
                buffer_idx = torch.cat(buffer_idx_list, dim=0)

                loss, pos_losses, pos_buffer_idx = model.loss(
                    head, rel, tail, label, sample_weight=weight, buffer_idx=buffer_idx
                )
                if (
                    pos_losses is not None
                    and pos_buffer_idx is not None
                    and pos_losses.numel() > 0
                ):
                    self.update(
                        pos_buffer_idx.detach().cpu().tolist(),
                        pos_losses.detach().cpu().tolist(),
                    )

        setattr(self.args, "snapshot", orig_snapshot)

    # ------------------------------------------------------------------------- #
    # Internal helpers
    # ------------------------------------------------------------------------- #
    def _compute_priority(self, loss_abs: float, struct_importance: float) -> float:
        loss_term = max(loss_abs, 0.0) + self.eps
        struct_term = max(struct_importance, self.eps)
        score = math.pow(loss_term, self.alpha) * math.pow(struct_term, self.gamma)
        return max(score, self.eps)

    def _recompute_probabilities(self) -> None:
        if not self.buffer:
            return
        scores = []
        for exp in self.buffer:
            exp.priority = self._compute_priority(exp.loss_abs, exp.struct_importance)
            scores.append(exp.priority)
        total_score = sum(scores)
        if total_score <= 0:
            total_score = self.eps * len(scores)
        for exp, score in zip(self.buffer, scores):
            exp.prob = score / total_score
        pool_size = len(self.buffer)
        for exp in self.buffer:
            prob = max(exp.prob, self.eps)
            weight = math.pow(1.0 / (pool_size * prob), self.beta)
            exp.weight = weight

    def _get_struct_importance(self, snapshot_id: int) -> Dict[Tuple[int, int], float]:
        if snapshot_id in self.struct_cache:
            return self.struct_cache[snapshot_id]
        file_path = os.path.join(self.args.data_path, str(snapshot_id), "train_edges_betweenness.txt")
        struct_map = self._parse_betweenness_file(file_path)
        if not struct_map:
            self.struct_default[snapshot_id] = 1.0
            self.struct_cache[snapshot_id] = {}
            if self.logger:
                self.logger.warning(
                    "Failed to load train_edges_betweenness.txt for snapshot %d, using default structural weight 1.0.",
                    snapshot_id,
                )
            return self.struct_cache[snapshot_id]
        min_val = min(struct_map.values())
        self.struct_default[snapshot_id] = max(min_val, self.eps)
        self.struct_cache[snapshot_id] = struct_map
        return struct_map

    def _parse_betweenness_file(self, file_path: str) -> Dict[Tuple[int, int], float]:
        if not os.path.isfile(file_path):
            return {}
        entries: List[Tuple[int, int, float]] = []
        with open(file_path, "r") as rf:
            for line in rf:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                try:
                    h = int(parts[0])
                    t = int(parts[1])
                    score = float(parts[2])
                except ValueError:
                    continue
                entries.append((h, t, score))
        if not entries:
            return {}
        entries.sort(key=lambda x: x[2], reverse=True)
        inv_rank_list: List[Tuple[Tuple[int, int], float]] = []
        for rank, (h, t, _) in enumerate(entries, start=1):
            inv_rank = 1.0 / rank
            inv_rank_list.append(((h, t), inv_rank))
        denom = sum(val for _, val in inv_rank_list)
        if denom <= 0:
            denom = 1.0
        struct_map: Dict[Tuple[int, int], float] = {}
        for (h, t), inv_rank in inv_rank_list:
            normalized = inv_rank / denom
            struct_map[(h, t)] = normalized
            struct_map[(t, h)] = normalized
        return struct_map


def build_training_facts_for_snapshot(kg, snapshot_id: int, train_new: bool) -> List[Tuple[int, int, int]]:
    """
    Build positive sample set for experience buffer, including original and inverse relations.
    """
    if snapshot_id not in kg.snapshots:
        return []
    snapshot = kg.snapshots[snapshot_id]
    base = snapshot.train if train_new else snapshot.train_all
    facts: List[Tuple[int, int, int]] = []
    for h, r, t in base:
        facts.append((h, r, t))
        facts.append((t, r + 1, h))
    return facts


import os
import numpy as np
import torch

from .load_data import Data
from .model import MuRP
from .rsgd import RiemannianSGD


class PoincareSnapshotTrainer:
    """
    Wrapper around MuRP training for snapshot-0.
    """

    def __init__(self, args, snapshot_id=None):
        self.args = args
        self.snapshot_id = int(self.args.snapshot) if snapshot_id is None else int(snapshot_id)
        snapshot_dir = os.path.join(self.args.data_path, str(self.snapshot_id)) + "/"
        self.data = Data(data_dir=snapshot_dir)
        self.entities = self.data.entities
        self.relations = self.data.relations
        self.entity_idxs = {ent: idx for idx, ent in enumerate(self.entities)}
        self.relation_idxs = {rel: idx for idx, rel in enumerate(self.relations)}
        self.cuda = args.device.type == "cuda"
        self.device = args.device

        self.model = MuRP(self.data, int(self.args.poincare_dim))
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.optimizer = RiemannianSGD(
            self.model.parameters(),
            lr=float(self.args.poincare_lr),
            param_names=self.param_names,
        )

        if self.cuda:
            self.model.cuda()

        self.er_vocab = self.get_er_vocab(self.get_data_idxs(self.data.data))
        self.all_entity_indices = torch.arange(
            len(self.entities), device=self.device, dtype=torch.long
        )

    def get_data_idxs(self, data):
        return [
            (
                self.entity_idxs[data[i][0]],
                self.relation_idxs[data[i][1]],
                self.entity_idxs[data[i][2]],
            )
            for i in range(len(data))
        ]

    @staticmethod
    def get_er_vocab(data_idxs):
        vocab = {}
        for triple in data_idxs:
            key = (triple[0], triple[1])
            vocab.setdefault(key, []).append(triple[2])
        return vocab

    def train_epoch(self):
        train_data_idxs = self.get_data_idxs(self.data.train_data)
        np.random.shuffle(train_data_idxs)
        losses = []
        batch_size = int(self.args.poincare_batch_size)
        nneg = int(self.args.poincare_nneg)
        entity_pool = list(self.entity_idxs.values())

        for j in range(0, len(train_data_idxs), batch_size):
            data_batch = np.array(train_data_idxs[j : j + batch_size])
            if data_batch.size == 0:
                continue
            negsamples = np.random.choice(entity_pool, size=(data_batch.shape[0], nneg))

            e1_idx = torch.LongTensor(
                np.repeat(data_batch[:, 0][:, np.newaxis], nneg + 1, axis=1)
            )
            r_idx = torch.LongTensor(
                np.repeat(data_batch[:, 1][:, np.newaxis], nneg + 1, axis=1)
            )
            e2_idx = torch.LongTensor(
                np.concatenate(
                    (data_batch[:, 2][:, np.newaxis], negsamples),
                    axis=1,
                )
            )
            targets = torch.zeros(e1_idx.shape, dtype=torch.double)
            targets[:, 0] = 1.0

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
                targets = targets.cuda()

            self.optimizer.zero_grad()
            predictions = self.model.forward(e1_idx, r_idx, e2_idx)
            loss = self.model.loss(predictions, targets)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return float(np.mean(losses)) if losses else 0.0

    def evaluate(self, split="valid"):
        if split == "valid":
            data = self.data.valid_data
        else:
            data = self.data.test_data
        test_data_idxs = self.get_data_idxs(data)
        sr_vocab = self.er_vocab

        hits = [[] for _ in range(10)]
        ranks = []

        self.model.eval()
        with torch.no_grad():
            for data_point in test_data_idxs:
                e1_idx = torch.tensor(data_point[0], device=self.device)
                r_idx = torch.tensor(data_point[1], device=self.device)
                e2_idx = torch.tensor(data_point[2], device=self.device)

                head_repeat = e1_idx.repeat(len(self.entities))
                rel_repeat = r_idx.repeat(len(self.entities))
                predictions_s = self.model.forward(
                    head_repeat,
                    rel_repeat,
                    self.all_entity_indices,
                )

                filt = sr_vocab[(data_point[0], data_point[1])]
                target_value = predictions_s[e2_idx].item()
                filt_idx = torch.tensor(filt, device=self.device, dtype=torch.long)
                predictions_s[filt_idx] = -np.Inf
                predictions_s[e1_idx] = -np.Inf
                predictions_s[e2_idx] = target_value

                sort_values, sort_idxs = torch.sort(predictions_s, descending=True)
                sort_idxs = sort_idxs.detach().cpu().numpy()
                rank = np.where(sort_idxs == e2_idx.item())[0][0]
                ranks.append(rank + 1)
                for hits_level in range(10):
                    hits[hits_level].append(1.0 if rank <= hits_level else 0.0)

        self.model.train()
        result = {
            "mr": float(np.mean(ranks)),
            "mrr": float(np.mean(1.0 / np.array(ranks))),
            "hits1": float(np.mean(hits[0])),
            "hits3": float(np.mean(hits[2])),
            "hits5": float(np.mean(hits[4])),
            "hits10": float(np.mean(hits[9])),
        }
        return result

    def get_embeddings(self):
        """
        Return the learned entity/relation embeddings after training.
        """
        ent = self.model.Eh.weight.data
        rel = self.model.rvh.weight.data
        return ent.detach().clone(), rel.detach().clone()


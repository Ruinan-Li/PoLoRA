import math
import random
from torch.utils.data import Dataset
from src.utils import *
from src.replay.per_buffer import build_training_facts_for_snapshot


class TrainDatasetMarginLoss(Dataset):
    def __init__(self, args, kg):
        super(TrainDatasetMarginLoss, self).__init__()
        self.args = args
        self.kg = kg
        self.facts, self.facts_new = self.build_facts()
        self.neg_ratio = int(self.args.neg_ratio)

    def __len__(self):
        if self.args.train_new:
            return len(self.facts_new[self.args.snapshot])
        else:
            return len(self.facts[self.args.snapshot])

    def __getitem__(self, index):
        if self.args.train_new:
            ele = self.facts_new[self.args.snapshot][index]
        else:
            ele = self.facts[self.args.snapshot][index]
        fact, label = ele['fact'], ele['label']

        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        weights = torch.ones(len(label), dtype=torch.float)
        if len(weights) > 0:
            weights[0] = float(self.neg_ratio)
        buffer_idx = torch.full((len(label),), -1, dtype=torch.long)
        return fact, label, weights, buffer_idx

    @staticmethod
    def collate_fn(data):
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        weights = torch.cat([_[2] for _ in data], dim=0)
        buffer_idx = torch.cat([_[3] for _ in data], dim=0)
        return fact[:, 0], fact[:, 1], fact[:, 2], label, weights, buffer_idx

    def corrupt(self, fact):
        ss_id = self.args.snapshot
        h, r, t = fact
        prob = 0.5

        neg_h = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label

    def build_facts(self):
        facts, facts_new = [], []
        for ss_id in range(int(self.args.snapshot_num)):
            facts_, facts_new_ = [], []
            for h, r, t in self.kg.snapshots[ss_id].train:
                facts_new_.append({'fact': (h, r, t), 'label': 1})
                facts_new_.append({'fact': (t, r + 1, h), 'label': 1})
            for h, r, t in self.kg.snapshots[ss_id].train_all:
                facts_.append({'fact': (h, r, t), 'label': 1})
                facts_.append({'fact': (h, r + 1, t), 'label': 1})
            facts.append(facts_)
            facts_new.append(facts_new_)
        return facts, facts_new


class PERTrainDataset(Dataset):
    """
    PER dataset for snapshots >= 1: resample old samples by P(i) each epoch, add all new samples.
    """

    def __init__(self, args, kg, per_buffer) -> None:
        super().__init__()
        self.args = args
        self.kg = kg
        self.per_buffer = per_buffer
        self.snapshot = int(self.args.snapshot)
        self.neg_ratio = int(self.args.neg_ratio)
        self.base_new = build_training_facts_for_snapshot(kg, self.snapshot, self.args.train_new)
        self.curr_data = []
        self.refresh_for_epoch()

    def __len__(self):
        return len(self.curr_data)

    def refresh_for_epoch(self):
        self.curr_data = []
        new_cnt = len(self.base_new)
        if new_cnt > 0:
            for fact in self.base_new:
                self.curr_data.append({"fact": fact, "weight": 1.0, "buffer_idx": -1})
        # Read replay ratio from parameters
        replay_ratio = float(getattr(self.args, "per_replay_ratio", 1.0))
        old_needed = int(math.ceil(new_cnt * replay_ratio))
        old_samples = self.per_buffer.sample(old_needed) if self.per_buffer is not None else []
        for sample in old_samples:
            self.curr_data.append(
                {
                    "fact": sample["fact"],
                    "weight": sample["weight"],
                    "buffer_idx": sample["buffer_idx"],
                }
            )
        random.shuffle(self.curr_data)

    def __getitem__(self, index):
        ele = self.curr_data[index]
        fact, label = self.corrupt(ele["fact"])
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        base_weight = float(ele["weight"])
        weight = torch.full((len(label),), base_weight, dtype=torch.float)
        buffer_idx = torch.full((len(label),), int(ele["buffer_idx"]), dtype=torch.long)
        return fact, label, weight, buffer_idx

    @staticmethod
    def collate_fn(data):
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        weights = torch.cat([_[2] for _ in data], dim=0)
        buffer_idx = torch.cat([_[3] for _ in data], dim=0)
        return fact[:, 0], fact[:, 1], fact[:, 2], label, weights, buffer_idx

    def corrupt(self, fact):
        ss_id = self.snapshot
        h, r, t = fact
        prob = 0.5

        neg_h = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.neg_ratio)
        neg_t = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label


class TestDataset(Dataset):
    def __init__(self, args, kg):
        super(TestDataset, self).__init__()
        self.args = args
        self.kg = kg

        self.valid, self.test = self.build_facts()

    def __len__(self):
        if self.args.valid:
            return len(self.valid[self.args.snapshot_valid])
        else:
            return len(self.test[self.args.snapshot_test])

    def __getitem__(self, index):
        if self.args.valid:
            element = self.valid[self.args.snapshot_valid][index]
        else:
            element = self.test[self.args.snapshot_test][index]
        fact, label = torch.LongTensor(element['fact']), element['label']
        label = self.get_label(label)
        return fact[0], fact[1], fact[2], label
    
    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        return h, r, t, label
    
    def get_label(self, label):
        """ for valid and test, a label is all entities labels: [0, ..., 0, 1, 0, ..., 0]"""
        if self.args.valid:
            y = np.zeros([self.kg.snapshots[self.args.snapshot_valid].num_ent], dtype=np.float32)
        else:
            y = np.zeros([self.kg.snapshots[self.args.snapshot_test].num_ent], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


    def build_facts(self):
        """ build positive data """
        valid, test = [], []
        for ss_id in range(int(self.args.snapshot_num)):
            valid_, test_ = [], []
            if self.args.train_new:
                for (h, r, t) in self.kg.snapshots[ss_id].valid:
                    valid_.append({'fact': (h, r, t), 'label': self.kg.snapshots[ss_id].hr2t_all[(h, r)]})
            else:
                for (h, r, t) in self.kg.snapshots[ss_id].valid_all:
                    valid_.append({'fact': (h, r, t), 'label': self.kg.snapshots[ss_id].hr2t_all[(h, r)]})
            if self.args.train_new:
                for (h, r, t) in self.kg.snapshots[ss_id].valid:
                    valid_.append({'fact': (t, r + 1, h), 'label': self.kg.snapshots[ss_id].hr2t_all[(t, r + 1)]})
            else:
                for (h, r, t) in self.kg.snapshots[ss_id].valid_all:
                    valid_.append({'fact': (t, r + 1, h), 'label': self.kg.snapshots[ss_id].hr2t_all[(t, r + 1)]})
            
            for (h, r, t) in self.kg.snapshots[ss_id].test:
                test_.append({'fact': (h, r, t), 'label': self.kg.snapshots[ss_id].hr2t_all[(h, r)]})
            for (h, r, t) in self.kg.snapshots[ss_id].test:
                test_.append({'fact': (t, r + 1, h), 'label': self.kg.snapshots[ss_id].hr2t_all[(t, r + 1)]})
            valid.append(valid_)
            test.append(test_)
        return valid, test

from ..utils import *
from ..data_load.data_loader import TrainDatasetMarginLoss, PERTrainDataset, TestDataset
from torch.utils.data import DataLoader


class TrainBatchProcessor():
    def __init__(self, args, kg, per_buffer=None) -> None:
        self.args = args
        self.kg = kg
        self.per_buffer = per_buffer
        use_per = bool(getattr(args, "per_enable", False)) and int(args.snapshot) > 0 and per_buffer is not None
        if use_per:
            self.dataset = PERTrainDataset(args, kg, per_buffer)
        else:
            self.dataset = TrainDatasetMarginLoss(args, kg)
        self.shuffle_mode = True
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=self.shuffle_mode,
            batch_size=int(self.args.batch_size),
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.random_seed)),
            pin_memory=True,
        )

    def process_epoch(self, model, optimizer):
        model.train()
        if hasattr(self.dataset, "refresh_for_epoch"):
            self.dataset.refresh_for_epoch()
        total_loss = 0.0
        if self.args.record:
            loss_save_path = "/data/my_cl_kge/save/" + str(self.args.snapshot) + ".txt"
            with open(loss_save_path, "a", encoding="utf-8") as wf:
                wf.write(str(self.args.epoch))
                wf.write("\t")
        for b_id, batch in enumerate(self.data_loader):
            bh, br, bt, by, bw, bidx = batch
            optimizer.zero_grad()
            loss, pos_losses, pos_buffer_idx = model.loss(
                bh.to(self.args.device),
                br.to(self.args.device),
                bt.to(self.args.device),
                by.to(self.args.device) if by is not None else by,
                sample_weight=bw.to(self.args.device) if bw is not None else None,
                buffer_idx=bidx.to(self.args.device) if bidx is not None else None,
            )
            loss = loss.float()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (
                self.per_buffer is not None
                and pos_buffer_idx is not None
                and pos_losses is not None
                and pos_losses.numel() > 0
            ):
                mask = pos_buffer_idx >= 0
                if mask.any():
                    upd_idx = pos_buffer_idx[mask].detach().cpu()
                    upd_loss = pos_losses[mask].detach().cpu()
                    self.per_buffer.update(upd_idx.tolist(), upd_loss.tolist())

            model.epoch_post_processing(bh.size(0))
            if self.args.record:
                with open(loss_save_path, "a", encoding="utf-8") as wf:
                    wf.write(str(loss.item()))
                    wf.write("\t")
        if self.args.record:
            with open(loss_save_path, "a", encoding="utf-8") as wf:
                wf.write("\n")
        return total_loss


class DevBatchProcessor():
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = 25
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
            generator=torch.Generator().manual_seed(int(args.random_seed)),
            pin_memory=True,
        )

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = {}
        snapshot_idx = int(self.args.snapshot_valid) if self.args.valid else int(self.args.snapshot_test)
        hr2t = self.kg.snapshots[snapshot_idx].hr2t_all
        for batch in self.data_loader:
            head, relation, tail, label = batch
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            label = label.to(self.args.device)
            num += len(head)
            stage = "Valid" if self.args.valid else "Test"
            use_poincare_eval = (
                getattr(self.args, "use_poincare_eval", False)
                and hasattr(model, "predict_poincare")
                and model.has_poincare_embeddings()
                and int(self.args.snapshot) == 0
                and int(self.args.snapshot_test) == 0
            )
            if use_poincare_eval:
                pred = model.predict_poincare(head, relation, stage=stage)
            else:
                pred = model.predict(head, relation, stage=stage)
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail]
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[batch_size_range, tail] = target_pred
            if self.args.predict_result and stage == "Test":
                logits_sorted, indices_sorted = torch.sort(pred, dim=-1, descending=True)
                predict_result_path = (
                    "/data2/jun/lora_clkge/save/predict_result/" + "lora_kge/" + str(self.args.snapshot) + "_" + str(self.args.snapshot_test) + ".txt"
                )
                with open(predict_result_path, "a", encoding="utf-8") as af:
                    batch_num = len(head)
                    for i in range(batch_num):
                        top1 = indices_sorted[i][0]
                        top2 = indices_sorted[i][1]
                        top3 = indices_sorted[i][2]
                        af.write(self.kg.id2entity[head[i].detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2relation[relation[i].detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[tail[i].detach().cpu().item()])
                        af.write("\n")
                        af.write(self.kg.id2entity[top1.detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[top2.detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[top3.detach().cpu().item()])
                        af.write("\n")
                        af.write("----------------------------------------------------------")
                        af.write("\n")
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[batch_size_range, tail]
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results[f'hits{k + 1}'] = torch.numel(
                    ranks[ranks <= (k + 1)]
                ) + results.get(f'hits{k + 1}', 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results

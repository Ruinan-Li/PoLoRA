from .utils import *
from .model.model_process import *
from .poincare.snapshot0 import PoincareSnapshotTrainer


class Trainer():
    def __init__(self, args, kg, model, optimizer, per_buffer=None) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.optimizer = optimizer
        self.logger = args.logger
        self.poincare_trainer = None
        if int(self.args.snapshot) == 0:
            self.poincare_trainer = PoincareSnapshotTrainer(args, snapshot_id=int(self.args.snapshot))
            setattr(self.model, "poincare_trainer", self.poincare_trainer)
            self.train_processor = None
            self.valid_processor = None
        else:
            self.train_processor = TrainBatchProcessor(args, kg, per_buffer=per_buffer)
            self.valid_processor = DevBatchProcessor(args, kg)

    def run_epoch(self):
        if int(self.args.snapshot) == 0:
            if self.poincare_trainer is None:
                raise RuntimeError("Poincare trainer not initialized for snapshot 0.")
            loss = self.poincare_trainer.train_epoch()
            res = self.poincare_trainer.evaluate(split="valid")
            return loss, res
        else:
            self.args.valid = True
            loss = self.train_processor.process_epoch(self.model, self.optimizer)
            res = self.valid_processor.process_epoch(self.model)
            self.args.valid = False
            
            # Call scheduler epoch step at end of each epoch
            if hasattr(self.optimizer, 'scheduler_epoch_step'):
                self.optimizer.scheduler_epoch_step()
            
            return loss, res

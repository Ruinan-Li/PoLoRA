from .utils import *
from .model.model_process import *
from .poincare.snapshot0 import PoincareSnapshotTrainer


class Tester():
    def __init__(self, args, kg, model) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.poincare_trainer = getattr(self.model, "poincare_trainer", None)
        # Only use MuRP path when current training snapshot is 0 and evaluation target is 0
        self.use_snapshot0_trainer = (
            int(self.args.snapshot) == 0 and int(self.args.snapshot_test) == 0
        )
        if self.use_snapshot0_trainer:
            if self.poincare_trainer is None:
                self.poincare_trainer = PoincareSnapshotTrainer(self.args, snapshot_id=0)
                setattr(self.model, "poincare_trainer", self.poincare_trainer)
            self.test_processor = None
        else:
            self.test_processor = DevBatchProcessor(args, kg)

    def test(self):
        if self.use_snapshot0_trainer and self.poincare_trainer is not None:
            split = "valid" if getattr(self.args, "valid", False) else "test"
            return self.poincare_trainer.evaluate(split=split)
        self.args.valid = False
        return self.test_processor.process_epoch(self.model)
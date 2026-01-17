import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, args, kg) -> None:
        super(BaseModel, self).__init__()
        self.args = args
        self.kg = kg

    def switch_snapshot(self):
        """ After the training process of a snapshot, prepare for next snapshot """
        pass

    def pre_snapshot(self):
        """ Process before training on a snapshot """
        pass

    def epoch_post_processing(self, size=None):
        """ Post process after a training iteration """
        pass

    def snapshot_post_processing(self):
        """ Post after training on a snapshot """
        pass

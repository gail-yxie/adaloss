import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from parse_config import ConfigParser


def cross_entropy(output, target):
    return F.cross_entropy(output, target)


class AbstractLoss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.0):
        super().__init__()
        self.num_examp = num_examp
        self.USE_CUDA = torch.cuda.is_available()
        self.num_classes = num_classes
        self.beta = beta
        self.rate = -1
        self.config = ConfigParser.get_instance()
        self.b0 = self.config['train_loss']['args']['b0']
        self.acc_loss_array = "NA"

    def update_epoch_acc_rate(self):
        pass


class CrossEntropy(AbstractLoss):
    def __init__(self, num_examp, num_classes=10, beta=0.0):
        super().__init__(num_examp, num_classes, beta)

    def forward(self, index, output, label):
        return F.cross_entropy(output, label)


class PlainLossb0(AbstractLoss):
    def __init__(self, num_examp, num_classes=10, beta=0.0):
        super().__init__(num_examp, num_classes, beta)

    def update_rate(self, tmp_loss):
        if self.b0 == 0:
            raise ValueError("b0 cannot be set to 0 for PlainLossb0!")
        self.rate = 1.0 / self.b0

    def forward(self, index, output, label):
        tmp_loss = F.cross_entropy(output, label, reduction='mean')
        self.update_rate(tmp_loss)
        return tmp_loss * self.rate


class PlainLossSqrt(PlainLossb0):
    def __init__(self, num_examp, num_classes=10, beta=0.0):
        super().__init__(num_examp, num_classes, beta)
        self.t = 0

    def update_rate(self, tmp_loss):
        self.t += 1
        self.rate = 1.0 / np.sqrt(self.b0 ** 2 + self.t * (1 - self.beta))


class AdaLoss(PlainLossb0):
    def __init__(self, num_examp, num_classes=10, beta=0.0):
        super().__init__(num_examp, num_classes, beta)
        self.acc_loss_array = np.array(self.b0 ** 2)

    def update_rate(self, tmp_loss):
        self.acc_loss_array += tmp_loss.item() * (1 - self.beta)
        self.rate = 1.0 / self.acc_loss_array ** 0.5

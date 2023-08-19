
import torch
import torch.nn as nn


class DiffBCE(nn.Module):
    def __init__(self, reduction: str='mean') -> None:
        super(DiffBCE, self).__init__()
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, data, output):

        diff_in, diff_out, label_in, label_out = output['diff_in'], output['diff_out'], output['label_in'], output['label_out']
        loss_in = self.bce(diff_in, label_in)
        loss_out = self.bce(diff_out, label_out)

        return loss_in + loss_out
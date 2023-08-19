import torch
import torch.nn as nn


class L2(nn.Module):
    def __init__(self, reduction: str='mean') -> None:
        super(L2, self).__init__()
        self.reduction = reduction

    def forward(self, data, output):
        target = data['y']
        loc, scale = output['loc'].chunk(2, dim=-1)
        reg_mask = output['reg_mask']

        l2 = torch.norm(target.unsqueeze(0) - loc, p=2, dim=-1)

        ade = l2.clone()
        ade[:,~reg_mask] = 0
        made_idcs = torch.argmin(ade.mean(-1), dim=0)
        minl2 = l2[made_idcs, torch.arange(l2.size(1))]

        if reg_mask.sum() > 0:
            if self.reduction == 'mean':
                return minl2[reg_mask].mean()
            else:
                raise ValueError(f'{self.reduction} is not a valid value for reduction')
        else:
            return 0
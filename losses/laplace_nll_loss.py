# opyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn


class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self,
                data, output) -> torch.Tensor:
        target = data['y']
        loc, scale = output['loc'].chunk(2, dim=-1)
        reg_mask = output['reg_mask']

        diff = torch.norm(target.unsqueeze(0) - loc, dim=-1)
        diff_ = diff.clone()
        diff_[:,~reg_mask] = 0
        best_mode = torch.argmin(diff_.mean(-1), dim=0)
        node_nums = best_mode.size(0)

        loc, scale = loc[best_mode, torch.arange(node_nums)], scale[best_mode, torch.arange(node_nums)]
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == 'mean':
            return nll[reg_mask].mean()
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))


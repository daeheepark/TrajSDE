# Copyright (c) 2022, Zikang Zhou. All rights reserved.
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
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class MR_T(Metric):

    def __init__(self,
                 dataset,
                 end_idcs,
                 sources=[0,1],
                 miss_threshold: float = 2.0,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 **kwargs) -> None:
        super(MR_T, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                 process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.miss_threshold = miss_threshold
        self.dataset = dataset
        self.end_idcs = end_idcs
        self.target_sources = sources

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               reg_mask: torch.Tensor,
               source) -> None:
        
        if self.dataset == 'nuScenes':
            l2 = torch.norm(pred - target.unsqueeze(0), p=2, dim=-1)
            reg_mask_any = reg_mask.any(-1)
            l2, reg_mask = l2[:,reg_mask_any], reg_mask[reg_mask_any]
            l2[:, ~reg_mask] = 0
            maxmin_l2 = l2.max(-1)[0].min(0)[0]

            missed = maxmin_l2 > self.miss_threshold
        elif self.dataset == 'Argoverse':
            K, NA, TS, _ = pred.shape
        
            count_0, count_1 = (source==self.target_sources[0]).sum(), (source==self.target_sources[1]).sum()
            end_idcs_ = torch.repeat_interleave(torch.tensor(self.end_idcs), torch.tensor([count_0, count_1]))

            l2 = torch.norm(pred[:,torch.arange(NA),end_idcs_,:] - target[torch.arange(NA),end_idcs_].unsqueeze(0), p=2, dim=-1)
            reg_mask_any = reg_mask[torch.arange(NA),end_idcs_]

            l2 = l2[:, reg_mask_any]
            best_idx = torch.argmin(l2, dim=0)
            fde_best = l2[best_idx, torch.arange(reg_mask_any.sum())]
            missed = fde_best > self.miss_threshold

        self.sum += missed.sum().item()
        self.count += reg_mask_any.sum().item()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count

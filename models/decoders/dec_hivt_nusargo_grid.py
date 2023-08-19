from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.util import init_weights


class MLPDecoder(nn.Module):

    def __init__(self,
                 **kwargs) -> None:
        super(MLPDecoder, self).__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.input_size = self.global_channels
        self.hidden_size = self.local_channels

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        if self.uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                data,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        out = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale  # [F, N, H, 2]
            out = {'loc': torch.cat((loc, scale), dim=-1), 'pi': pi, 'local_embed': local_embed, 'global_embed': global_embed} # [F, N, H, 4], [N, F]
        
        else:
            out = {'loc': loc, 'pi': pi} # [F, N, H, 2], [N, F]
        
        out['reg_mask'] = ~data['padding_mask'][:,-self.future_steps:]
        return out
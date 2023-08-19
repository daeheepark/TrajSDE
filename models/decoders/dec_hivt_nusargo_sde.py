from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchsde

from models.utils.util import init_weights
from torchsde import sdeint


class SDEDecoder(nn.Module):

    def __init__(self,
                 **kwargs) -> None:
        super(SDEDecoder, self).__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.input_size = self.global_channels
        self.hidden_size = self.local_channels

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))

        # ode_func_netD = create_net(self.hidden_size, self.hidden_size,
        #                                 n_layers=self.ode_func_layers,
        #                                 n_units=self.hidden_size,
        #                                 nonlinear=nn.Tanh)

        # gen_ode_func = ODEFunc(ode_func_net=ode_func_netD)
        
        # self.diffeq_solver = DiffeqSolver(gen_ode_func,
        #                                   'euler',
        #                                   odeint_rtol=self.rtol,
        #                                   odeint_atol=self.atol)

        sigma, theta, mu = 0.5, 1.0, 0.0
        post_drift = FFunc(self.hidden_size)
        prior_drift = HFunc(theta=theta, mu=mu)
        diffusion= GFunc(self.hidden_size, sigma=sigma)
        self.lsde_func = LSDEFunc(f=post_drift, g=diffusion, h=prior_drift, embed_dim=self.hidden_size)
        self.lsde_func.noise_type, self.lsde_func.sde_type = 'diagonal', 'ito'
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        
        if self.uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 2))
        
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        
        self.hidden = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.normal_(self.hidden, mean=0., std=.02)
        
        self.ts_pred = torch.linspace(0, self.max_fut_t, self.future_steps+1)
        # self.tstp, self.tstp_mask = self.interp_timesteps(self.ts_to_predict, self.min_stepsize, return_mask=True)
        
        self.apply(init_weights)

    def forward(self,
                data,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        loc_emb = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
        
        ### Diff solver 풀어씀 ### (sol_y = self.diffeq_solver(loc_emb, self.ts_pred))
        _, num_actors, _ = loc_emb.shape
        hidden_0 = loc_emb.view(self.num_modes*num_actors, self.hidden_size)

        sol_y = sdeint(self.lsde_func, hidden_0, self.ts_pred, dt=self.min_stepsize, dt_min=self.min_stepsize, rtol=self.rtol, atol=self.atol, method=self.method)[1:].permute(1,0,2)

        ##########################
        

        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        loc = self.decoder(sol_y).view(self.num_modes, num_actors, self.future_steps, 2)  # [F, N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(sol_y), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale  # [F, N, H, 2]
            out = {'loc': torch.cat((loc, scale), dim=-1), 'pi': pi,} # [F, N, H, 4], [N, F]
        
        else:
            out = {'loc': loc, 'pi': pi} # [F, N, H, 2], [N, F]
        
        out['reg_mask'] = ~data['padding_mask'][:,-self.future_steps:]
        return out
    
class FFunc(nn.Module):
    """Posterior drift."""
    def __init__(self, embed_dim):
        super(FFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim+2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t, y):
        # if t.dim() == 0:
        #     t = float(t) * torch.ones_like(y)
        # # Positional encoding in transformers; must use `t`, since the posterior is likely inhomogeneous.
        # inp = torch.cat((torch.sin(t), torch.cos(t), y), dim=-1)
        _t = torch.ones(y.size(0), 1) * float(t)
        _t = _t.to(y)
        inp = torch.cat((y,torch.sin(_t), torch.cos(_t)), dim=-1)
        return self.net(inp)


class HFunc(nn.Module):
    """Prior drift"""
    def __init__(self, theta=1.0, mu=0.0):
        super(HFunc, self).__init__()
        self.theta = nn.Parameter(torch.tensor([[theta]]), requires_grad=False)
        self.mu = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)

    def forward(self, t, y):
        return self.theta * (self.mu - y)


class GFunc(nn.Module):
    """Diffusion"""
    def __init__(self, embed_dim, sigma=0.5):
        super(GFunc, self).__init__()
        # self.sigma = nn.Parameter(torch.tensor([[sigma]]), requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(embed_dim+2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, t, y):
        _t = torch.ones(y.size(0), 1) * float(t)
        _t = _t.to(y)
        out = self.net(torch.cat((y, torch.sin(_t), torch.cos(_t)), dim=-1))
        return torch.sigmoid(out)
    
class LSDEFunc(torchsde.SDEIto):
    def __init__(self, f, g, h, embed_dim, order=1):
        super().__init__(noise_type="diagonal")
        self.order, self.intloss, self.sensitivity = order, None, None
        self.f_func, self.g_func, self.h_func = f, g, h
        self.fnfe, self.gnfe, self.hnfe = 0, 0, 0
        self.embed_dim = embed_dim


    def forward(self, s, x):
        pass

    def h(self, s, x):
        """ Prior drift
        :param s:
        :param x:
        """
        self.hnfe += 1
        return self.h_func(t=s, y=x)

    def f(self, s, x):
        """Posterior drift.
        :param s:
        :param x:
        """
        self.fnfe += 1
        return self.f_func(t=s, y=x)

    def g(self, s, x):
        """Diffusion.
        :param s:
        :param x:
        """
        self.gnfe += 1
        out = self.g_func(t=s, y=x).repeat(1,self.embed_dim)
        return out
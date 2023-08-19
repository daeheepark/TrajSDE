from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models.utils.embedding import MultipleInputEmbedding
from models.utils.embedding import SingleInputEmbedding
from models.utils.util import DistanceDropEdge
from models.utils.util import TemporalData
from models.utils.util import init_weights
from models.utils.ode_utils import get_timesteps, GRU_Unit
from models.utils.sde_utils import SDiffeqSolver, SDEFunc
import torchsde
from models.utils.sdeint import sdeint, sdeint_dual

class LocalEncoderSDESepPara2(nn.Module):

    def __init__(self,
                 **kwargs) -> None:
        super(LocalEncoderSDESepPara2, self).__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.drop_edge = DistanceDropEdge(self.local_radius)
        self.aa_encoder = AAEncoder(historical_steps=self.historical_steps,
                                    node_dim=self.node_dim,
                                    edge_dim=self.edge_dim,
                                    embed_dim=self.embed_dim,
                                    num_heads=self.num_heads,
                                    dropout=self.dropout,
                                    parallel=self.parallel)
        
        self.al_encoder = ALEncoder(node_dim=self.node_dim,
                                    edge_dim=self.edge_dim,
                                    embed_dim=self.embed_dim,
                                    num_heads=self.num_heads,
                                    dropout=self.dropout)
        
        self.gru_unit = GRU_Unit(self.embed_dim, self.embed_dim, n_units=self.embed_dim)

        sigma, theta, mu = 0.5, 1.0, 0.0
        post_drift = FFunc(self.embed_dim, num_layers=self.sde_layers)
        prior_drift = HFunc(theta=theta, mu=mu)
        diffusion_nus= GFunc(self.embed_dim, num_layers=self.sde_layers, sigma=sigma)
        diffusion_Argo2= GFunc(self.embed_dim, num_layers=self.sde_layers, sigma=sigma)
        self.lsde_func = LSDEFunc(f=post_drift, g_nus=diffusion_nus, g_Argo2=diffusion_Argo2, h=prior_drift, embed_dim=self.embed_dim)
        self.lsde_func.noise_type, self.lsde_func.sde_type = 'diagonal', 'ito'

        self.real_label, self.fake_label = 0, 1

        self.hidden = nn.Parameter(torch.Tensor(self.embed_dim))
        nn.init.normal_(self.hidden, mean=0., std=.02)

        self.apply(init_weights)

    def forward(self, data: TemporalData) -> torch.Tensor:

        lane_len = (1-data['lane_paddings']).sum(-1)
        lane_start_pos = data['lane_positions'][torch.arange(data['lane_positions'].size(0)),0,:]
        lane_end_pos = data['lane_positions'][torch.arange(data['lane_positions'].size(0)),(lane_len-1).long(),:]
        lane_feat = lane_end_pos - lane_start_pos

        nus_batches = torch.where(data.source == 0)[0]
        nus_mask = torch.isin(data.batch, nus_batches)

        actor_num, ts_obs, in_dim = data.x.shape

        prev_hidden = self.hidden.unsqueeze(0).repeat(actor_num+len(data['agent_index']),1)

        actors_pad = data['padding_mask']
        actors_past_pad, actors_fut_pad = actors_pad[:,:self.ref_time+1], actors_pad[:,self.ref_time+1:]
        actors_past_mask = ~actors_past_pad

        ######### Pre-compute AA encoding #########

        ## Add only agent to train SDENet ##
        
        edge_from = data.edge_index[0][torch.isin(data.edge_index[1], data['agent_index'])]
        edge_to = data.edge_index[1][torch.isin(data.edge_index[1], data['agent_index'])]
        _, new_edge_to = torch.unique(edge_to, return_inverse=True)
        new_edge_ = torch.stack((edge_from,new_edge_to+actor_num), 0)
        new_edge = torch.cat((data.edge_index, new_edge_), -1)
        
        x_agent = data.x[data['agent_index']]
        x_actors = torch.cat((data.x, x_agent+2*torch.randn_like(x_agent)), dim=0)
        actors_pad = torch.cat((actors_pad, actors_pad[data['agent_index']]), dim=0)
        actors_mask = ~actors_pad[:,:self.ref_time+1]
        new_positions = torch.cat((data['positions'], data['positions'][data['agent_index']]), 0)
        new_bos_mask = torch.cat((data['bos_mask'], data['bos_mask'][data['agent_index']]), 0)
        new_rotate_mat = torch.cat((data['rotate_mat'], data['rotate_mat'][data['agent_index']]), 0)
        new_agent_index = torch.cat((data['agent_index'], torch.arange(actor_num,actor_num+data['agent_index'].size(0)).to(data['agent_index'].device)))

        nus_mask = torch.cat((nus_mask,(data.source == 0)), dim=0)

        #####################################

        for t in range(self.historical_steps):
            data[f'edge_index_{t}'], _ = subgraph(subset=~actors_pad[:, t], edge_index=new_edge)
            data[f'edge_attr_{t}'] = \
                new_positions[data[f'edge_index_{t}'][0], t] - new_positions[data[f'edge_index_{t}'][1], t]
            
        if self.parallel:
            snapshots = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                snapshots[t] = Data(x=x_actors[:, t], edge_index=edge_index, edge_attr=edge_attr,
                                    num_nodes=data.num_nodes+len(data['agent_index']))
            batch = Batch.from_data_list(snapshots)
            aa_out = self.aa_encoder(x=batch.x, t=None, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                                  bos_mask=new_bos_mask, rotate_mat=new_rotate_mat)
            aa_out = aa_out.view(self.historical_steps, aa_out.shape[0] // self.historical_steps, -1)
        else:
            raise NotImplementedError

        ###########################################


        past_time_steps = torch.linspace(-self.max_past_t,0,self.historical_steps)
        past_time_steps = -1*past_time_steps
        time_points_iter = range(0, past_time_steps.size(-1))
        if self.run_backwards:
            prev_t, t_i = past_time_steps[-1] - 0.01, past_time_steps[-1]
            time_points_iter = reversed(time_points_iter)
        else:
            prev_t, t_i = past_time_steps[0] - 0.01, past_time_steps[0]

        latent_ys=[]
        diffusions=[]

        for idx, t in enumerate(time_points_iter):

            time_points = torch.tensor([prev_t, t_i])

            ############ DiffeqSolver 풀어서 쓰기 ##############
            first_point, time_steps_to_predict = prev_hidden, time_points
            n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]


            pred_y, diff_noise = sdeint_dual(self.lsde_func, first_point, time_steps_to_predict, nus_mask, dt=self.minimum_step,
                rtol = self.rtol, atol = self.atol, method = self.method)
            pred_y = pred_y.permute(1,2,0)

            assert(pred_y.size()[0] == n_traj_samples)
            assert(pred_y.size()[1] == n_traj)

            # ode_sol = pred_y[0].permute(0,2,1)
            ode_sol = pred_y
            ####################################################

            if torch.mean(ode_sol[:, :, 0] - prev_hidden) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0] - prev_hidden))
                exit()

            yi_ode = ode_sol[:, :, -1]
            xi = aa_out[t]
            maski = actors_mask[:,t]

            yi = self.gru_unit(input_tensor=xi, h_cur=yi_ode, mask=maski).squeeze(0)

            diffusion_t = diff_noise[new_agent_index]
            
            # return to iteration
            prev_hidden = yi
            if idx+1 < past_time_steps.size(-1):
                if self.run_backwards:
                    prev_t, t_i = past_time_steps[t], past_time_steps[t-1]
                else:
                    prev_t, t_i = past_time_steps[t], past_time_steps[t+1]

            latent_ys.append(yi)
            diffusions.append(diffusion_t)

        latent_ys = torch.stack(latent_ys)[:,:-len(data['agent_index'])]
        diffusions = torch.stack(diffusions)

        eos_idcs = self.ref_time - torch.argmax(data['bos_mask'].float(), dim=1)
        out = latent_ys[eos_idcs, torch.arange(latent_ys.size(1)),:]

        agent_eos_idcs = eos_idcs[data['agent_index']]
        diff_out = diffusions[agent_eos_idcs.repeat(2), torch.arange(diffusions.size(1))]

        
        diffusions_in, diffusions_out = torch.chunk(diff_out,2,0)
        in_labels = torch.full_like(diffusions_in, self.real_label)
        out_labels = torch.full_like(diffusions_out, self.fake_label)
        
        edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
        out = self.al_encoder(x=(lane_feat, out), edge_index=edge_index, edge_attr=edge_attr,
                              rotate_mat=data['rotate_mat'])
            
        return out, diffusions_in, diffusions_out, in_labels, out_labels
    
    def forward_ood(self, data: TemporalData) -> torch.Tensor:

        lane_len = (1-data['lane_paddings']).sum(-1)
        lane_start_pos = data['lane_positions'][torch.arange(data['lane_positions'].size(0)),0,:]
        lane_end_pos = data['lane_positions'][torch.arange(data['lane_positions'].size(0)),(lane_len-1).long(),:]
        lane_feat = lane_end_pos - lane_start_pos

        nus_batches = torch.where(data.source == 0)[0]
        nus_mask = torch.isin(data.batch, nus_batches)

        actor_num, ts_obs, in_dim = data.x.shape

        actors_pad = data['padding_mask']
        actors_past_pad, actors_fut_pad = actors_pad[:,:self.ref_time+1], actors_pad[:,self.ref_time+1:]
        actors_past_mask = ~actors_past_pad

        ######### Pre-compute AA encoding #########

        actors_mask = ~actors_pad[:,:self.ref_time+1]

        #####################################

        for t in range(self.historical_steps):
            data[f'edge_index_{t}'], _ = subgraph(subset=~actors_pad[:, t], edge_index=data.edge_index)
            data[f'edge_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][0], t] - data['positions'][data[f'edge_index_{t}'][1], t]
            
        if self.parallel:
            snapshots = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                snapshots[t] = Data(x=data.x[:, t], edge_index=edge_index, edge_attr=edge_attr,
                                    num_nodes=data.num_nodes)
            batch = Batch.from_data_list(snapshots)
            aa_out = self.aa_encoder(x=batch.x, t=None, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                                  bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
            aa_out = aa_out.view(self.historical_steps, aa_out.shape[0] // self.historical_steps, -1)
        else:
            raise NotImplementedError

        ###########################################

        past_time_steps = torch.linspace(-self.max_past_t,0,self.historical_steps)
        past_time_steps = -1*past_time_steps
        
        # eos_idcs = self.ref_time - torch.argmax(agent_past_mask.float(), dim=1)
        eos_idcs = self.ref_time - torch.argmax(data['bos_mask'].float(), dim=1)

        eval_iter = 10

        outs = []
        for j in range(eval_iter):

            prev_hidden = torch.zeros((actor_num, self.embed_dim), device=data.x.device)
            time_points_iter = range(0, past_time_steps.size(-1))

            if self.run_backwards:
                prev_t, t_i = past_time_steps[-1] - 0.01, past_time_steps[-1]
                
                time_points_iter = reversed(time_points_iter)
            else:
                prev_t, t_i = past_time_steps[0] - 0.01, past_time_steps[0]

            latent_ys=[]
            for idx, t in enumerate(time_points_iter):
                time_points = torch.tensor([prev_t, t_i])
                ############ DiffeqSolver 풀어서 쓰기 ##############
                first_point, time_steps_to_predict = prev_hidden, time_points
                n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

                pred_y, diff_noise = sdeint_dual(self.lsde_func, first_point, time_steps_to_predict, nus_mask, dt=self.minimum_step,
                rtol = self.rtol, atol = self.atol, method = self.method)
                pred_y = pred_y.permute(1,2,0)

                assert(pred_y.size()[0] == n_traj_samples)
                assert(pred_y.size()[1] == n_traj)

                # ode_sol = pred_y[0].permute(0,2,1)
                ode_sol = pred_y
                ####################################################

                if torch.mean(ode_sol[:, :, 0] - prev_hidden) >= 0.001:
                    print("Error: first point of the ODE is not equal to initial value")
                    print(torch.mean(ode_sol[:, :, 0] - prev_hidden))
                    exit()

                yi_ode = ode_sol[:, :, -1]
                xi = aa_out[t]
                maski = actors_mask[:,t]
                
                yi = self.gru_unit(input_tensor=xi, h_cur=yi_ode, mask=maski).squeeze(0)

                # return to iteration
                prev_hidden = yi
                if idx+1 < past_time_steps.size(-1):
                    if self.run_backwards:
                        prev_t, t_i = past_time_steps[t], past_time_steps[t-1]
                    else:
                        prev_t, t_i = past_time_steps[t], past_time_steps[t+1]

                latent_ys.append(yi)

            latent_ys = torch.stack(latent_ys)

            out = latent_ys[eos_idcs, torch.arange(latent_ys.size(1)),:]
            outs.append(out)
        
        outs = torch.stack(outs)
        actors_std = outs.std(0).mean(-1)
        out = outs.mean(0)
        
        edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
        out = self.al_encoder(x=(lane_feat, out), edge_index=edge_index, edge_attr=edge_attr,
                              rotate_mat=data['rotate_mat'])
            

        # # OOD, IND visualization 해보는 코드
        # num_samples2show = 10

        # past_ts = torch.linspace(-2, 0,21)

        # ood_actors = data['positions'][torch.topk(actors_std,num_samples2show,0,True)[1]][:,:21]
        # ood_masks = ~data['padding_mask'][torch.topk(actors_std,num_samples2show,0,True)[1]][:,:21]
        # import matplotlib.pyplot as plt
        # for ai, pos in enumerate(ood_actors[:100]):
        #     fig, ax = plt.subplots(1,1,figsize=(5,5))
        #     pos_ = pos[ood_masks[ai]].detach().cpu()
        #     ts_ = past_ts[ood_masks[ai]]
        #     for i_, pos__ in enumerate(pos_):
        #         ax.scatter(pos__[0], pos__[1], c='b')
        #         ax.text(pos__[0], pos__[1], f'{ts_[i_].item():.1f}')
        #     ax.scatter(pos_[-1,0], pos_[-1,1], c='r')
        #     ax.set_aspect('equal')
        #     plt.margins(0.5)
        #     plt.savefig(f'tmp/tmp_ood/{ai}.jpg')
        #     plt.close()

        # in_actors = data['positions'][torch.topk(actors_std,num_samples2show,0,False)[1]][:,:21]
        # in_masks = ~data['padding_mask'][torch.topk(actors_std,num_samples2show,0,False)[1]][:,:21]
        # import matplotlib.pyplot as plt
        # for ai, pos in enumerate(in_actors[:100]):
        #     fig, ax = plt.subplots(1,1,figsize=(5,5))
        #     pos_ = pos[in_masks[ai]].detach().cpu()
        #     ts_ = past_ts[in_masks[ai]]
        #     for i_, pos__ in enumerate(pos_):
        #         ax.scatter(pos__[0], pos__[1], c='b')
        #         ax.text(pos__[0], pos__[1], f'{ts_[i_].item():.1f}')
        #     ax.scatter(pos_[-1,0], pos_[-1,1], c='r')
        #     ax.set_aspect('equal')
        #     plt.margins(0.5)
        #     plt.savefig(f'tmp/tmp_in/{ai}.jpg')
        #     plt.close()

        # mask = ((actors_std>3.5).float() * (actors_std<4.5).float()).bool()
        # in_actors = data['positions'][mask]
        # in_masks = ~data['padding_mask'][mask]
        # import matplotlib.pyplot as plt
        # for ai, pos in enumerate(in_actors[:100]):
        #     fig, ax = plt.subplots(1,1,figsize=(5,5))
        #     pos_ = pos[in_masks[ai]].detach().cpu()
        #     ax.scatter(pos_[-1,0], pos_[-1,1])
        #     ax.plot(pos_[:,0], pos_[:,1])
        #     ax.set_aspect('equal')
        #     plt.savefig(f'tmp_norm/{ai}.jpg')
        #     plt.close()

        return out, actors_std

class FFunc(nn.Module):
    """Posterior drift."""
    def __init__(self, embed_dim, num_layers=2):
        super(FFunc, self).__init__()
        net_list = []
        net_list.append(nn.Linear(embed_dim+2, embed_dim))
        for _ in range(num_layers):
            net_list.append(nn.Tanh())
            net_list.append(nn.Linear(embed_dim, embed_dim))
        # self.net = nn.Sequential(
        #     nn.Linear(embed_dim+2, embed_dim),
        #     nn.Tanh(),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.Tanh(),
        #     nn.Linear(embed_dim, embed_dim)
        # )
        self.net = nn.Sequential(*nn.ModuleList(net_list))

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
    def __init__(self, embed_dim, sigma=0.5, num_layers=2):
        super(GFunc, self).__init__()
        # self.sigma = nn.Parameter(torch.tensor([[sigma]]), requires_grad=False)

        net_list = []
        net_list.append(nn.Linear(embed_dim+2, embed_dim))
        for _ in range(num_layers-1):
            net_list.append(nn.Tanh())
            net_list.append(nn.Linear(embed_dim, embed_dim))
        net_list.append(nn.Tanh())
        net_list.append(nn.Linear(embed_dim, 1))

        self.net = nn.Sequential(*nn.ModuleList(net_list))

        # self.net = nn.Sequential(
        #     nn.Linear(embed_dim+2, embed_dim),
        #     nn.Tanh(),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.Tanh(),
        #     nn.Linear(embed_dim, 1)
        # )

    def forward(self, t, y):
        _t = torch.ones(y.size(0), 1) * float(t)
        _t = _t.to(y)
        out = self.net(torch.cat((y, torch.sin(_t), torch.cos(_t)), dim=-1))
        return torch.sigmoid(out)
    
class LSDEFunc(torchsde.SDEIto):
    def __init__(self, f, g_nus, g_Argo2, h, embed_dim, order=1):
        super().__init__(noise_type="diagonal")
        self.order, self.intloss, self.sensitivity = order, None, None
        self.f_func, self.g_nus, self.g_argo, self.h_func = f, g_nus, g_Argo2, h
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

    def g(self, s, x, nus_mask):
        """Diffusion. Dual diffusion network, separated by nus, argo2
        :param s:
        :param x:
        """
        self.gnfe += 1
        # out = self.g_nus(t=s, y=x).repeat(1,self.embed_dim)
        out = torch.empty(x.size(0), self.embed_dim, device=x.device)
        out_0 = self.g_nus(t=s, y=x[nus_mask])
        out_1 = self.g_argo(t=s, y=x[~nus_mask])
        out[nus_mask,:] = out_0.repeat(1,self.embed_dim)
        out[~nus_mask,:] = out_1.repeat(1,self.embed_dim)
        return out

    # def g(self, s, x):
    #     """Diffusion.
    #     :param s:
    #     :param x:
    #     """
    #     self.gnfe += 1
    #     out = self.g_nus(t=s, y=x).repeat(1,self.embed_dim)
    #     # out = torch.zeros(x.size(0),self.embed_dim).to(x.device)
    #     # out_0 = self.g_nus(t=s, y=x[nus_mask])
    #     # out_1 = self.g_argo(t=s, y=x[~nus_mask])
    #     # out[nus_mask,:] = out_0.repeat(1,self.embed_dim)
    #     # out[~nus_mask,:] = out_1.repeat(1,self.embed_dim)
    #     return out

class AAEncoder(MessagePassing):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 parallel: bool = False,
                 **kwargs) -> None:
        super(AAEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.nbr_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                t: Optional[int],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                bos_mask: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        if self.parallel:
            if rotate_mat is None:
                center_embed = self.center_embed(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1))
            else:
                center_embed = self.center_embed(
                    torch.matmul(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1).unsqueeze(-2),
                                 rotate_mat.expand(self.historical_steps, *rotate_mat.shape)).squeeze(-2))
                
            center_embed = torch.where(bos_mask.t().unsqueeze(-1),
                                       self.bos_token.unsqueeze(-2),
                                       center_embed).contiguous().view(x.shape[0], -1)
        else:
            if rotate_mat is None:
                center_embed = self.center_embed(x)
            else:
                center_embed = self.center_embed(torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2))
            center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed)
        center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat,
                                                      size)
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        return center_embed

    def message(self,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if rotate_mat is None:
            nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            if self.parallel:
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                center_rotate_mat = rotate_mat[edge_index[1]]
            nbr_embed = self.nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                        torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        return inputs + gate * (self.lin_self(center_embed) - inputs)

    def _mha_block(self,
                   center_embed: torch.Tensor,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                    edge_attr=edge_attr, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)


class ALEncoder(MessagePassing):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
        nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
        nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                is_intersections: torch.Tensor = None,
                turn_directions: torch.Tensor = None,
                traffic_controls: torch.Tensor = None,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        x_lane, x_actor = x
        
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, rotate_mat, size)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                # is_intersections_j,
                # turn_directions_j,
                # traffic_controls_j,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if rotate_mat is None:
            x_j = self.lane_embed([x_j, edge_attr])
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                                   torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2)])
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                #    is_intersections: torch.Tensor,
                #    turn_directions: torch.Tensor,
                #    traffic_controls: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                                rotate_mat=rotate_mat, size=size))
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)
import os
from pathlib import Path
import pickle as pkl
from copy import deepcopy
import json
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import Linear

from models.utils.util import TemporalData
from debug_util import viz_result_batch_base, viz_result_batch_goalpred, viz_result_batch_ood, viz_result_batch_ood_load

import importlib
from importlib.machinery import SourceFileLoader

import matplotlib.pyplot as plt
import sys


class PredictionModelSDENet(pl.LightningModule):

    def __init__(self,
                 **kwargs) -> None:

        super(PredictionModelSDENet, self).__init__()
        self.save_hyperparameters()

        for key, value in kwargs.items():
            if key == 'training_specific':
                for k, v in value.items():
                    setattr(self, k, v)
            elif key == 'model_specific':
                for k, v in value['kwargs'].items():
                    setattr(self, k, v)

        enc_args, agg_args, dec_args = kwargs['encoder'], kwargs['aggregator'], kwargs['decoder']
        encoder = getattr(SourceFileLoader(enc_args['module_name'], enc_args['file_path']).load_module(enc_args['module_name']), enc_args['module_name'])
        aggregator = getattr(SourceFileLoader(agg_args['module_name'], agg_args['file_path']).load_module(agg_args['module_name']), agg_args['module_name'])
        decoder = getattr(SourceFileLoader(dec_args['module_name'], dec_args['file_path']).load_module(dec_args['module_name']), dec_args['module_name'])

        self.encoder = encoder(**dict(kwargs['encoder']['kwargs']))
        self.aggregator = aggregator(**dict(kwargs['aggregator']['kwargs']))
        self.decoder = decoder(**dict(kwargs['decoder']['kwargs']))

        self.losses = []
        self.loss_names = []
        for i, loss_path in enumerate(kwargs['losses']):
            loss_module_name = kwargs['losses_module'][i]

            loss = getattr(SourceFileLoader(loss_module_name, loss_path).load_module(loss_module_name), loss_module_name)
            loss = loss(**dict(kwargs['loss_args'][i]))
            self.losses.append(loss)
            self.loss_names.append(loss_module_name)
        self.loss_weights = kwargs['loss_weights']
        
        self.metrics_tr = []
        self.metrics_vl = []
        self.metric_names = []
        for i, metric_path in enumerate(kwargs['metrics']):
            metric_module_name = kwargs['metrics_module'][i]

            metric = getattr(SourceFileLoader(metric_module_name, metric_path).load_module(metric_module_name), metric_module_name)
            metric = metric(**dict(kwargs['metric_args'][i]))
            self.metrics_tr.append(metric)
            self.metrics_vl.append(deepcopy(metric))
            self.metric_names.append(metric_module_name)

        if hasattr(self, 'stds_fn'):
            with open(self.stds_fn, 'rb') as f:
                self.stds_loaded = pkl.load(f)

    def forward(self, data: TemporalData):
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = False
        
        if hasattr(self, 'ood') and self.ood:
            local_embed, stds = self.encoder.forward_ood(data=data)
        else:
            local_embed, diffusions_in, diffusionts_out, in_labels, out_labels = self.encoder(data=data)
            
        global_embed = self.aggregator(data=data, local_embed=local_embed)
        out = self.decoder(data=data, local_embed=local_embed, global_embed=global_embed)
                    
        if hasattr(self, 'ood') and self.ood:
            out['stds'] = stds
        else:
            out['diff_in'], out['diff_out'], out['label_in'], out['label_out'] = diffusions_in, diffusionts_out, in_labels, out_labels
            
        return out

    def training_step(self, data, batch_idx):
        output = self(data)
        
        loss = 0
        for lidx, lossfn in enumerate(self.losses):
            lossname = self.loss_names[lidx]
            loss_i = lossfn(data, output)
            loss = loss + self.loss_weights[lidx]*loss_i
            self.log(f'train/{lossname}', loss_i, prog_bar=True, on_step=True, on_epoch=True, batch_size=output['loc'].size(1))
        self.log('lr', self.scheduler.get_lr()[0], prog_bar=False, on_step=False, on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        output = self(data)

        y_hat_agent = output['loc'][:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        agent_reg_mask = output['reg_mask'][data['agent_index']]
        agent_source = data['source']

        if not self.is_gtabs:
            y_hat_agent = torch.cumsum(y_hat_agent, dim=-2)
            y_agent = torch.cumsum(y_agent, dim=-2)

        for midx, metric in enumerate(self.metrics_vl):
            metricname = self.metric_names[midx]
            metric.update(y_hat_agent.detach().cpu(), y_agent.detach().cpu(), agent_reg_mask.detach().cpu(), agent_source.detach().cpu())
            
    def test_step(self, data, batch_idx):
        output = self(data)

        if self.only_agent:
            self.leave_only_agent(data, output)

        y_hat_agent = output['loc'][:, data['agent_index'], :, : 2]
        if data.y is not None: y_agent = data.y[data['agent_index']]
        pi_agent = output['pi'][data['agent_index']]
        origin_agent = data['positions'][data['agent_index'], self.ref_time]
        agent_reg_mask = output['reg_mask'][data['agent_index']]
        agent_source = data['source']

        if data.y is not None:
            for metric in self.metrics_vl:
                metric.update(y_hat_agent.detach().cpu(), y_agent.detach().cpu(), agent_reg_mask.detach().cpu(), agent_source.detach().cpu())
    
    def test_epoch_end(self, outputs) -> None:

        ckpt_path = Path(self.trainer._ckpt_path)
        out_dir = os.path.join(ckpt_path.parent.parent, 'out')
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        
        metrics = dict()
        for midx, metric in enumerate(self.metrics_vl):
            metricname = self.metric_names[midx]
            metrics[metricname] = metric.compute().item()

        ckpt_name = ckpt_path.stem
        ckpt_fn = os.path.join(out_dir, f'result_{ckpt_name}.json')
        with open(ckpt_fn, 'w') as f:
            json.dump(metrics, f)


    @staticmethod
    def leave_only_agent(data, output):
        data.num_nodes = data.x.size(0)
        data.bos_mask = data.bos_mask[data['agent_index']]
        data.y = data.y[data['agent_index']]
        data.x = data.x[data['agent_index']]
        if 'category' in data.keys: data.category = data.category[data['agent_index']]
        data.positions = data.positions[data['agent_index']]
        data.rotate_mat = data.rotate_mat[data['agent_index']]
        data.rotate_angles = data.rotate_angles[data['agent_index']]
        data.has_goal = data.has_goal[data['agent_index']]
        data.padding_mask = data.padding_mask[data['agent_index']]

        al_agent_mask = torch.isin(data['lane_actor_index'][1], data['agent_index'])
        agent_has_lane = torch.isin(data['agent_index'], data['lane_actor_index'][1])
        data.goal_idcs = data.goal_idcs[al_agent_mask]
        data.lane_actor_vectors = data.lane_actor_vectors[al_agent_mask]
        
        output['loc'] = output['loc'][:,data['agent_index']]
        output['pi'] = output['pi'][data['agent_index']]
        output['reg_mask'] = output['reg_mask'][data['agent_index']]
        if 'cls_mask' in output: output['cls_mask'] = output['cls_mask'][data['agent_index']]
        if 'goal_prob' in output:
            output['goal_prob'] = output['goal_prob'][al_agent_mask]
        if 'goal_cls_mask' in output:
            output['goal_cls_mask'] = output['goal_cls_mask'][al_agent_mask]

        data.lane_actor_index = data.lane_actor_index[:,al_agent_mask]
        for i, agent_i in enumerate(data['agent_index']):
            if agent_has_lane[i]:
                data.lane_actor_index[1][data.lane_actor_index[1] == agent_i] = i

        data.agent_index = torch.arange(data.x.size(0)).to(data.x.device)
        data.av_index = torch.arange(data.x.size(0)).to(data.x.device)
        data.batch = torch.arange(data.x.size(0)).to(data.x.device)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.T_max, eta_min=0.0)
        return [self.optimizer], [self.scheduler]


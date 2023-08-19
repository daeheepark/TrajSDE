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

        if self.only_agent:
            self.leave_only_agent(data, output)
        
        loss = 0
        for lidx, lossfn in enumerate(self.losses):
            lossname = self.loss_names[lidx]
            loss_i = lossfn(data, output)
            loss = loss + self.loss_weights[lidx]*loss_i
            self.log(f'train/{lossname}', loss_i, prog_bar=True, on_step=True, on_epoch=True, batch_size=output['loc'].size(1))
        self.log('lr', self.scheduler.get_lr()[0], prog_bar=False, on_step=False, on_epoch=True, batch_size=1)

        y_hat_agent = output['loc'][:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        agent_reg_mask = output['reg_mask'][data['agent_index']]
        agent_source = data['source']
        
        if not self.is_gtabs:
            y_hat_agent = torch.cumsum(y_hat_agent, dim=-2)
            y_agent = torch.cumsum(y_agent, dim=-2)

        for midx, metric in enumerate(self.metrics_tr):
            metricname = self.metric_names[midx]
            metric.update(y_hat_agent.detach().cpu(), y_agent.detach().cpu(), agent_reg_mask.detach().cpu(), agent_source.detach().cpu())

        if batch_idx == 0 and self.viz:
            viz_result_batch_base(self, self.logger.log_dir, data, output, batch_idx, 'train', True, True)
        
        if batch_idx == 0 and self.viz_goalpred:
            viz_result_batch_goalpred(self, self.logger.log_dir, data, output, batch_idx, 'train', True, True)
        

        return loss

    def validation_step(self, data, batch_idx):
        output = self(data)

        if self.only_agent:
            self.leave_only_agent(data, output)

        for lidx, lossfn in enumerate(self.losses):
            lossname = self.loss_names[lidx]
            loss_i = lossfn(data, output)
            
            self.log(f'val/{lossname}', loss_i, prog_bar=True, on_step=True, on_epoch=True, batch_size=output['loc'].size(1))

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
            
        if batch_idx == 0 and self.viz:
            viz_result_batch_base(self, self.logger.log_dir, data, output, batch_idx, 'val', True, True)
        
        if batch_idx == 0 and self.viz_goalpred:
            viz_result_batch_goalpred(self, self.logger.log_dir, data, output, batch_idx, 'val', True, True)

    def training_epoch_end(self, outputs) -> None:
        for midx, metric in enumerate(self.metrics_tr):
            metricname = self.metric_names[midx]
            metric = metric.to(self.device)
            metric_value = metric.compute().item()
            self.log(f'train/{metricname}', metric_value, on_step=False, on_epoch=True)
            # self.logger.log_hyperparams(self.hparams, {f'hparam_train_{metricname}': metric_value})
            metric.reset()

    def validation_epoch_end(self, outputs) -> None:
        for midx, metric in enumerate(self.metrics_vl):
            metricname = self.metric_names[midx]
            metric = metric.to(self.device)
            metric_value = metric.compute().item()
            self.log(f'val/{metricname}', metric_value, on_step=False, on_epoch=True)
            # self.logger.log_hyperparams(self.hparams, {f'hparam_val_{metricname}': metric_value})
            if metricname == 'ADE_T':
                self.log('hp_metric', metric_value)
            metric.reset()

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

        if not self.is_gtabs:
            y_hat_agent = torch.cumsum(y_hat_agent, dim=-2)
            if data.y is not None: y_agent = torch.cumsum(y_agent, dim=-2)

        if data.y is not None:
            for metric in self.metrics_vl:
                metric.update(y_hat_agent.detach().cpu(), y_agent.detach().cpu(), agent_reg_mask.detach().cpu(), agent_source.detach().cpu())

        # if True: 
        if self.ood:
            log_dir = Path(self.trainer._ckpt_path).parent.parent

            # seq_ids, mades, stds, reg_masks, locs, pis = viz_result_batch_ood_load(self, log_dir, data, output, batch_idx, 'test', self.stds_loaded, True, True)

            seq_ids, mades, stds, reg_masks, locs, pis = viz_result_batch_ood(self, log_dir, data, output, batch_idx, 'test', True, True)
            return seq_ids, mades, stds, reg_masks, locs, pis

        if self.viz:
            log_dir = Path(self.trainer._ckpt_path).parent.parent
            viz_result_batch_base(self, log_dir, data, output, batch_idx, 'test', True, True)
            
        if self.viz_goalpred:
            log_dir = Path(self.trainer._ckpt_path).parent.parent
            viz_result_batch_goalpred(self, log_dir, data, output, batch_idx, 'test', True, True)

        if self.submit:
            assert (data['source'][0] != data['source']).sum() == 0, 'in submit, all data sources should be same'
            if data['source'][0] == 0:
                y_hat_agent = y_hat_agent[:,:,4::5,:]
            elif data['source'][0] == 1:
                y_hat_agent = y_hat_agent[:,:,:30,:]

            translation = data['origin']
            
            # rotmat agent
            rotate_mat = torch.index_select(data['rotate_mat'], dim=0, index=data['agent_index'])
            rotate_invmat = torch.eye(2).to(rotate_mat.device) * rotate_mat - (1-torch.eye(2)).to(rotate_mat.device) * rotate_mat
            
            # rotmat av
            sin_vals = torch.sin(data['theta'])
            cos_vals = torch.cos(data['theta'])
            inv_rot_mat = torch.empty(data['theta'].size(0), 2, 2, device=self.device)
            inv_rot_mat[:, 0, 0] = cos_vals
            inv_rot_mat[:, 0, 1] = sin_vals
            inv_rot_mat[:, 1, 0] = -sin_vals
            inv_rot_mat[:, 1, 1] = cos_vals
            
            assert data['theta'].size(0) == y_hat_agent.size(1)
            K, A, Ts, _ =  y_hat_agent.shape
            y_hat_agent_ori = y_hat_agent.permute(1,0,2,3).reshape(A,-1,2)

            y_hat_agent_global = torch.bmm(y_hat_agent_ori, rotate_invmat)
            y_hat_agent_global = torch.bmm(y_hat_agent_global, inv_rot_mat)
            # y_hat_agent_global = y_hat_agent_global + origin_agent.unsqueeze(1) + translation.unsqueeze(1)
            y_hat_agent_global = y_hat_agent_global + torch.bmm(origin_agent.unsqueeze(1),inv_rot_mat)  + translation.unsqueeze(1)
            # y_hat_agent_global

            # y_hat_agent_ori = torch.bmm(y_hat_agent_ori, inv_rot_mat) + origin_agent.unsqueeze(1)
            y_hat_agent_global = y_hat_agent_global.reshape(A,K,Ts,2)

            return [y_hat_agent_global.detach().cpu(), pi_agent.detach().cpu(), data['seq_id']]
    
    def test_epoch_end(self, outputs) -> None:
        with open('tmp/result_uncertain/womd_sdesep_10_out.pt', 'wb') as f:
            pkl.dump(outputs, f)
        sys.exit()

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

        if True: 
        # if self.ood:
            stds_dict = {}

            made_full, std_full, reg_mask_full = [], [], []
            for output_batch in outputs:
                seq_ids, mades, stds, reg_masks, _, _ = output_batch

                made_full.append(torch.cat(mades))
                std_full.append(torch.cat(stds))
                reg_mask_full.append(torch.cat(reg_masks))

                for ii, seq_id in enumerate(seq_ids):
                    stds_dict[seq_id] = stds[ii]

            made_full, std_full, reg_mask_full = torch.cat(made_full), torch.cat(std_full), torch.cat(reg_mask_full)
            reg_any_mask = reg_mask_full.sum(-1) != 0

            made_full_, std_full_, reg_mask_full_ = made_full[reg_any_mask], std_full[reg_any_mask], reg_mask_full[reg_any_mask]

            ood_thres = [0.01, 0.06] # TODO: 여기 dataset에 따라 바뀌도록, config 파일에 적힌대로 바꾸기
            minADE = made_full_.mean()
            minADE_ood = made_full_[std_full_>ood_thres[1]].mean()
            minADE_in = made_full_[std_full_<ood_thres[0]].mean()

            ood_out = {}
            ood_out['minADE'] = minADE.item()
            ood_out['minADE_ood'] = minADE_ood.item()
            ood_out['minADE_in'] = minADE_in.item()
            ood_out['thres'] = ood_thres

            if len(list(stds_dict.keys())[0].split('_')) == 2:
                dset = 'nuScenes'
            elif len(list(stds_dict.keys())[0].split('-')) == 1:
                dset = 'Argoverse'
            elif len(list(stds_dict.keys())[0].split('-')) > 1:
                dset = 'Argoverse2'


            ckpt_fn = os.path.join(out_dir, f'ood_{dset}_{ckpt_name}.json')
            with open(ckpt_fn, 'w') as f:
                json.dump(ood_out, f)


            stds_fn = os.path.join(out_dir, f'stds_{dset}_{ckpt_name}.pt')
            with open(stds_fn, 'wb') as f:
                pkl.dump(stds_dict, f)
                    
        if self.submit:
            if self.dataset == 'nuScenes':
                from nuscenes.eval.prediction.data_classes import Prediction
                preds = []

                print('Submission generating...')
                for batch in outputs:
                    traj, pi, seq_id = batch
                    for bi in range(len(seq_id)):
                        instance_token, sample_token = seq_id[bi].split('_')
                        pred = Prediction(instance=instance_token, sample=sample_token,
                                                        prediction=traj[bi].numpy(), probabilities=pi[bi].numpy()).serialize()
                        preds.append(pred)

                print('Submission saving...')
                json.dump(preds, open(os.path.join(out_dir, f"submission_{ckpt_name}.json"), "w"))
                print('Submission saved')
            
            elif self.dataset == 'Argoverse':
                import sys
                sys.path.append('/home/user/Repos/argoverse-api/argoverse')
                from argoverse.evaluation.competition_util import generate_forecasting_h5

                softmax = torch.nn.Softmax(dim=-1)

                trajectories = {}
                probabilities = {}

                for batch in outputs:
                    traj, pi, seq_id = batch
                    for bi in range(len(seq_id)):
                        traj_ = traj[bi]
                        pi_ = softmax(pi[bi])
                        seq_id_ = int(seq_id[bi])

                        trajectories[seq_id_] = traj_.detach().cpu().numpy()
                        probabilities[seq_id_] = pi_.detach().cpu().numpy().tolist()
                
                generate_forecasting_h5(data=trajectories, output_path=out_dir, probabilities=probabilities, filename=f'submission_{ckpt_name}')
            
            else:
                raise KeyError

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

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group('HiVT')
    #     parser.add_argument('--historical_steps', type=int, default=20)
    #     parser.add_argument('--future_steps', type=int, default=30)
    #     parser.add_argument('--num_modes', type=int, default=6)
    #     parser.add_argument('--rotate', type=bool, default=True)
    #     parser.add_argument('--node_dim', type=int, default=2)
    #     parser.add_argument('--edge_dim', type=int, default=2)
    #     parser.add_argument('--embed_dim', type=int, required=True)
    #     parser.add_argument('--num_heads', type=int, default=8)
    #     parser.add_argument('--dropout', type=float, default=0.1)
    #     parser.add_argument('--num_temporal_layers', type=int, default=4)
    #     parser.add_argument('--num_global_layers', type=int, default=3)
    #     parser.add_argument('--local_radius', type=float, default=50)
    #     parser.add_argument('--parallel', type=bool, default=False)
    #     parser.add_argument('--lr', type=float, default=5e-4)
    #     parser.add_argument('--weight_decay', type=float, default=1e-4)
    #     parser.add_argument('--T_max', type=int, default=64)
    #     return parent_parser


if __name__ == '__main__':
    import yaml

    config_file = '/home/user/ssd4tb/frm_lightning/configs/original_hivt.yml'
    with open(config_file, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
    model = PredictionModel(**dict(cfg['model_specific']))
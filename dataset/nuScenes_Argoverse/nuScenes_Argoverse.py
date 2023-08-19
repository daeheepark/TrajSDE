import os
import os.path as osp
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import torch

from multiprocessing import Process
from multiprocessing import Pool
from itertools import repeat

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from scipy.spatial.distance import cdist
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data.dataset import files_exist
from tqdm import tqdm
import pickle as pkl
from shapely.geometry import LineString, Point
import random

import sys
sys.path.append('/mnt/ssd2/frm_lightning_backup/')
from models.utils.util import TemporalData
from debug_util import *

SPLIT_NAME = {'nuScenes': {'train': 'train', 'val': 'val', 'test': 'val', 'mini_train': 'mini_train', 'mini_val': 'mini_val'},
                'Argoverse': {'train': 'train', 'val': 'train', 'test': 'test_obs', 'sample': 'forecasting_sample'}}
DATA_SOURCE = {0: 'nuScenes', 1: 'Argoverse'}
CATEGORY_INTEREST = [0, 1, 2, 3, 4, 5, 7, 8]


class nuArgoDataset(Dataset):

    def __init__(self,
                 split: str,
                 nu_root,
                 Argo_root,
                 nu_dir,
                 Argo_dir,
                 spec_args=None) -> None:
        self._split = split
        self.nu_root = nu_root
        self.Argo_root = Argo_root
        self.nu_dir = nu_dir
        self.Argo_dir = Argo_dir

        for k,v in spec_args.items():
            self.__setattr__(k, v)

        self.nu_directory = SPLIT_NAME['nuScenes'][split]
        self.Argo_directory = SPLIT_NAME['Argoverse'][split]

        nu_raw_file_names = sorted(get_prediction_challenge_split(self.nu_directory, self.nu_root))
        nu_processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in nu_raw_file_names]
        nu_processed_paths = [os.path.join(self.nu_dir, self.nu_directory, f) for f in nu_processed_file_names]
        
        _Argo_raw_file_names_fn = os.path.join(self.Argo_dir, f'raw_processed_fns_{self.Argo_directory}.pt')
        if os.path.isfile(_Argo_raw_file_names_fn):
            with open(file=_Argo_raw_file_names_fn, mode='rb') as f:
                Argo_raw_file_names, Argo_processed_file_names, Argo_processed_paths = pkl.load(f)
        else:
            raise FileExistsError('Argo file name file should be exist')

        self._raw_file_names = []
        self._processed_file_names = []
        self._processed_paths = []
        self._sources = []
        if self.nus:
            self._raw_file_names = self._raw_file_names + nu_raw_file_names
            self._processed_file_names = self._processed_file_names + nu_processed_file_names
            self._processed_paths = self._processed_paths + nu_processed_paths
            self._sources = self._sources + [0]*len(nu_raw_file_names)
        if self.Argo:
            self._raw_file_names = self._raw_file_names + Argo_raw_file_names
            self._processed_file_names = self._processed_file_names + Argo_processed_file_names
            self._processed_paths = self._processed_paths + Argo_processed_paths
            self._sources = self._sources + [1]*len(Argo_raw_file_names)  # 0 for nuScenes, 1 for Argoverse

        if self.type == 'grid':
            self.max_past, self.max_fut = 21, 60
            grid_past, grid_fut = torch.zeros(self.max_past, dtype=torch.bool), torch.zeros(self.max_fut, dtype=torch.bool)

            self.ts_pasts, self.ts_futs = torch.linspace(-20,0,21).int(), torch.linspace(0,60,61)[1:].int()
            self.ts_nus_past, self.ts_nus_fut = torch.linspace(-20,0,5).int(), torch.linspace(0,60,13)[1:].int()
            self.ts_Argo_past, self.ts_Argo_fut = torch.linspace(-20,0,21)[1:].int(), torch.linspace(0,30,31)[1:].int()

            self.mask_nus_past, self.mask_nus_fut = grid_past.clone(), grid_fut.clone()
            self.mask_nus_past[torch.isin(self.ts_pasts, self.ts_nus_past)] = True
            self.mask_nus_fut[torch.isin(self.ts_futs, self.ts_nus_fut)] = True
            self.mask_nus_tot = torch.cat((self.mask_nus_past, self.mask_nus_fut))

            self.mask_Argo_past, self.mask_Argo_fut = grid_past.clone(), grid_fut.clone()
            self.mask_Argo_past[torch.isin(self.ts_pasts, self.ts_Argo_past)] = True
            self.mask_Argo_fut[torch.isin(self.ts_futs, self.ts_Argo_fut)] = True
            self.mask_Argo_tot = torch.cat((self.mask_Argo_past, self.mask_Argo_fut))

            self.ts_nus_past, self.ts_nus_fut = self.ts_nus_past.float(), self.ts_nus_fut.float()
            self.ts_nus_tot = torch.cat((self.ts_nus_past,self.ts_nus_fut))
            self.ts_Argo_past, self.ts_Argo_fut = self.ts_Argo_past.float(), self.ts_Argo_fut.float()
            self.ts_Argo_tot = torch.cat((self.ts_Argo_past, self.ts_Argo_fut))

        elif self.type == 'continuous':
            self.nus_ts_past, self.nus_ts_fut = torch.linspace(-2,0,5), torch.linspace(0.5,6,12)
            self.nus_ts_tot = torch.cat((self.nus_ts_past, self.nus_ts_fut))
            self.Argo_ts_past, self.Argo_ts_fut = torch.linspace(-1.9,0,20), torch.linspace(0.1,3,30)
            self.Argo_ts_tot = torch.cat((self.Argo_ts_past, self.Argo_ts_fut))
            self.max_past, self.max_fut = 20, 30

        super(nuArgoDataset, self).__init__()
    
    def _download(self):
        return
    
    def _process(self):
        return

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths
    
    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        data = torch.load(self.processed_paths[idx])
        
        data['source'] = self._sources[idx]
        data['seq_id'] = str(data['seq_id'])

        if 'traffic_controls' in data.keys: del data.traffic_controls
        if 'turn_directions' in data.keys: del data.turn_directions
        if 'is_intersections' in data.keys: del data.is_intersections
        if 'city' in data.keys: del data.city
        if 'lane_rotate_angles' in data.keys: del data.lane_rotate_angles
        if 'lane_edge_index' in data.keys: del data.lane_edge_index
        if 'lane_edge_type' in data.keys: del data.lane_edge_type
        if 'lane_edge_index2_succ' in data.keys: del data.lane_edge_index2_succ
        if 'lane_edge_index2_pred' in data.keys: del data.lane_edge_index2_pred
        if 'lane_edge_index2_neigh' in data.keys: del data.lane_edge_index2_neigh
        # if isinstance(data.agent_index, torch.Tensor): data.agent_index = data.agent_index.item()
        # if isinstance(data.av_index, torch.Tensor): data.av_index = data.av_index.item()
        data.agent_index = torch.tensor(data.agent_index, dtype=torch.long)
        data.av_index = torch.tensor(data.av_index, dtype=torch.long)

        if data['source'] == 0:
            data.x = data.x/5
        
        if not self.is_gtabs:
            y_pad0 = torch.cat((torch.zeros(data.y.size(0),1,2), data.y), dim=1)
            data.y = y_pad0[:,1:] - y_pad0[:,:-1]
            if data['source'] == 0:
                data.y = data.y/5

        if 'category' in data.keys:
            data['category'] = data['category'].float().to(data['x'].device)
            data.padding_mask[~torch.isin(data['category'], torch.tensor(CATEGORY_INTEREST)), -self.max_fut:] = True
            del data.category

        if self.type == 'grid':
            _x = torch.zeros((data.x.size(0), self.max_past, data.x.size(-1)), dtype=data.x.dtype)
            if data.y is not None:
                _y = torch.zeros((data.y.size(0), self.max_fut, data.y.size(-1)), dtype=data.y.dtype)  
            else:
                _y = data.y
            _bos_mask = torch.zeros((data.x.size(0), self.max_past), dtype=torch.bool)
            _padding_mask = torch.ones((data.x.size(0), self.max_past+self.max_fut), dtype=torch.bool)
            _positions = torch.zeros((data.x.size(0), self.max_past+self.max_fut, data.positions.size(-1)), dtype=data.positions.dtype)

            if data['source'] == 0:
                past_mask, fut_mask, tot_mask = self.mask_nus_past, self.mask_nus_fut, self.mask_nus_tot
            elif data['source'] == 1:
                past_mask, fut_mask, tot_mask = self.mask_Argo_past, self.mask_Argo_fut, self.mask_Argo_tot

            _x[:,past_mask] = data.x
            if data.y is not None: _y[:,fut_mask] = data.y
            _bos_mask[:, past_mask] = data.bos_mask
            _padding_mask[:,tot_mask] = data.padding_mask
            _positions[:,tot_mask] = data.positions

            data.x, data.y, data.bos_mask, data.padding_mask, data.positions = _x, _y, _bos_mask, _padding_mask, _positions
        
        elif self.type == 'continuous':
            raise NotImplementedError('continuous is not used for now')
            if data['source'] == 0:
                _x = torch.zeros((data.x.size(0), self.max_past, data.x.size(-1)), dtype=data.x.dtype)
                _y = torch.zeros((data.y.size(0), self.max_fut, data.y.size(-1)), dtype=data.y.dtype)
                _bos_mask = torch.zeros((data.x.size(0), self.max_past), dtype=torch.bool)
                _padding_mask = torch.ones((data.x.size(0), self.max_past+self.max_fut), dtype=torch.bool)
                _positions = torch.zeros((data.x.size(0), self.max_past+self.max_fut, data.positions.size(-1)), dtype=data.positions.dtype)

                _x[:,-5:] = data.x
                _y[:,:12] = data.y
                _bos_mask[:, -5:] = data.bos_mask
                _padding_mask[:,19-5+1:19+12+1] = data.padding_mask
                _positions[:,19-5+1:19+12+1] = data.positions

                data.x, data.y, data.bos_mask, data.padding_mask, data.positions = _x, _y, _bos_mask, _padding_mask, _positions
                
                _ts_past, _ts_fut = torch.zeros((data.x.size(0), self.max_past), dtype=self.nus_ts_past.dtype), torch.zeros((data.y.size(0), self.max_fut), dtype=self.nus_ts_fut.dtype)
                _ts_past[:,-5:] = self.nus_ts_past
                _ts_fut[:,:12] = self.nus_ts_fut
                data['ts_past'], data['ts_fut'], data['ts_tot'] = _ts_past, _ts_fut, torch.cat((_ts_past, _ts_fut), dim=-1)
            
            elif data['source'] == 1:
                _ts_past, _ts_fut = torch.zeros((data.x.size(0), self.max_past), dtype=self.nus_ts_past.dtype), torch.zeros((data.x.size(0), self.max_fut), dtype=self.nus_ts_fut.dtype)
                
                _ts_past[:] = self.Argo_ts_past
                _ts_fut[:] = self.Argo_ts_fut
                data['ts_past'], data['ts_fut'], data['ts_tot'] = _ts_past, _ts_fut, torch.cat((_ts_past, _ts_fut), dim=-1)
            else:
                raise KeyError('source should be nuScenes(0) or Argoverse(1)')

        if self._split == 'train':
            data = self.augment(data) 

        return data

    def augment(self, data):
        if self.random_flip:
            if random.choice([0, 1]):
                data.x = data.x * torch.tensor([-1,1])
                data.y = data.y * torch.tensor([-1,1])
                data.positions = data.positions * torch.tensor([-1,1])
                theta_x = torch.cos(data.theta)
                theta_y = torch.sin(data.theta)
                data.theta = torch.atan2(theta_y, -1*theta_x)
                angle_x = torch.cos(data.rotate_angles)
                angle_y = torch.sin(data.rotate_angles)
                data.rotate_angles = torch.atan2(angle_y, -1*angle_x)
                data.lane_positions = data.lane_positions * torch.tensor([-1,1])
                data.lane_vectors = data.lane_vectors * torch.tensor([-1,1])
                data.lane_actor_vectors = data.lane_actor_vectors * torch.tensor([-1,1])
            if random.choice([0, 1]):
                data.x = data.x * torch.tensor([1,-1])
                data.y = data.y * torch.tensor([1,-1])
                data.positions = data.positions * torch.tensor([1,-1])
                theta_x = torch.cos(data.theta)
                theta_y = torch.sin(data.theta)
                data.theta = torch.atan2(-1*theta_y, theta_x)
                angle_x = torch.cos(data.rotate_angles)
                angle_y = torch.sin(data.rotate_angles)
                data.rotate_angles = torch.atan2(-1*angle_y, angle_x)
                data.lane_positions = data.lane_positions * torch.tensor([1,-1])
                data.lane_vectors = data.lane_vectors * torch.tensor([1,-1])
                data.lane_actor_vectors = data.lane_actor_vectors * torch.tensor([1,-1])

        return data
    
    def _process(self):
        if files_exist(self.processed_paths):  # pragma: no cover
            print('Found processed files')
            return
        else:
            raise FileExistsError('Both nuScenes and Argoverse dataset should be parsed')


if __name__ == '__main__':
    split = 'val'
    spec_args = {'type': 'grid', 'nus': True, 'Argo': True} # input type should be in ['grid', 'continuous'] -> grid for Conventional method, continuous for Neural DE method

    A1D = nuArgoDataset(split, nu_root='data/nuScenes', Argo_root='data/argodataset', nu_dir='preprocessed/nuScenes_frm', Argo_dir='preprocessed/Argoverse_abs', spec_args=spec_args)
    
    from debug_util import viz_data_goal

    for data in A1D:
        viz_data_goal(data, 'tmp', 20)

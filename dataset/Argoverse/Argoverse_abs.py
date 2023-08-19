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

SPLIT_NAME = {'nuScenes': {'train': 'train', 'val': 'train_val', 'test': 'val', 'mini_train': 'mini_train', 'mini_val': 'mini_val'},
                'Argoverse': {'train': 'train', 'val': 'val', 'test': 'test_obs', 'sample': 'forecasting_sample'}}


class ArgoverseDataset(Dataset):

    def __init__(self,
                 split: str,
                 root: str,
                 process_dir: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50,
                 spec_args: Dict = None) -> None:
        self._split = split
        self._local_radius = local_radius
        for k,v in spec_args.items():
            self.__setattr__(k, v)

        self._directory = SPLIT_NAME[self.dataset][split]
        self.root = root
        self.process_dir = process_dir

        raw_file_names_fn = os.path.join(self.process_dir, f'raw_processed_fns_{split}.pt')

        if os.path.isfile(raw_file_names_fn):
            with open(file=raw_file_names_fn, mode='rb') as f:
                self._raw_file_names, self._processed_file_names, self._processed_paths = pkl.load(f)
        else:
            self._raw_file_names = os.listdir(self.raw_dir)
            self._processed_file_names, self._processed_paths = [], []
            for f in self._raw_file_names:
                processed_file_name = os.path.splitext(f)[0] + '.pt'
                processed_path = os.path.join(self.processed_dir, processed_file_name)
                self._processed_file_names.append(processed_file_name)
                self._processed_paths.append(processed_path)

            os.makedirs(self.process_dir, exist_ok=True)
            with open(file=raw_file_names_fn, mode='wb') as f:
                pkl.dump([self._raw_file_names, self._processed_file_names, self._processed_paths], f)

        super(ArgoverseDataset, self).__init__(root, transform=transform)
    
    def _download(self):
        return
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.process_dir, self._directory)

    # @property
    # def raw_file_names(self) -> Union[str, List[str], Tuple]:
    #     return self._raw_file_names

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
                # lane_angle_x = torch.cos(data.lane_rotate_angles)
                # lane_angle_y = torch.sin(data.lane_rotate_angles)
                # data.lane_rotate_angles = torch.atan2(lane_angle_y, -1*lane_angle_x)
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
                # lane_angle_x = torch.cos(data.lane_rotate_angles)
                # lane_angle_y = torch.sin(data.lane_rotate_angles)
                # data.lane_rotate_angles = torch.atan2(-1*lane_angle_y, lane_angle_x)
                data.lane_positions = data.lane_positions * torch.tensor([1,-1])
                data.lane_vectors = data.lane_vectors * torch.tensor([1,-1])
                data.lane_actor_vectors = data.lane_actor_vectors * torch.tensor([1,-1])

        if self.random_rotate:
            rotate_scale = self.random_rotate_deg / 180 * torch.pi
            rotate_angle_pert = torch.clamp(torch.normal(torch.zeros_like(data.rotate_angles), torch.ones_like(data.rotate_angles)), -1, 1) * rotate_scale
            data.rotate_angles = data.rotate_angles + rotate_angle_pert
            data.rotate_angles = data.rotate_angles + 2*torch.pi*(data.rotate_angles < -torch.pi) - 2*torch.pi*(data.rotate_angles > torch.pi)


        return data
    
    def _process(self):

        if files_exist(self.processed_paths):  # pragma: no cover
            print('Found processed files')
            return

        print('Processing...', file=sys.stderr)

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        print('Done!', file=sys.stderr)

    def process(self) -> None:

        from argoverse.map_representation.map_api import ArgoverseMap

        am = ArgoverseMap()

        self.process_argoverse(self._split, self._raw_file_names, am, self._local_radius)

    def process_argoverse(self, split, raw_file_names: str, am, radius: float) -> Dict:

        for raw_fn in tqdm(raw_file_names):
            df = pd.read_csv(os.path.join(self.raw_dir, raw_fn))

            # filter out actors that are unseen during the historical time steps
            timestamps = list(np.sort(df['TIMESTAMP'].unique()))
            historical_timestamps = timestamps[: 20]
            ref_timestep = [timestamps[19]]
            historical_df = df[df['TIMESTAMP'].isin(ref_timestep)]
            actor_ids = list(historical_df['TRACK_ID'].unique())
            df = df[df['TRACK_ID'].isin(actor_ids)]
            num_nodes = len(actor_ids)

            av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
            av_index = actor_ids.index(av_df[0]['TRACK_ID'])
            agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
            agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
            city = df['CITY_NAME'].values[0]

            # make the scene centered at AV
            origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
            av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)
            theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
            rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                    [torch.sin(theta), torch.cos(theta)]])
            
            # initialization
            x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
            edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
            padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
            bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
            rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

            for actor_id, actor_df in df.groupby('TRACK_ID'):
                node_idx = actor_ids.index(actor_id)
                node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
                padding_mask[node_idx, node_steps] = False
                if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
                    padding_mask[node_idx, 20:] = True
                xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
                x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
                node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
                if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
                    heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
                    rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
                else:  # make no predictions for the actor if the number of valid time steps is less than 2
                    padding_mask[node_idx, 20:] = True

            # bos_mask is True if time step t is valid and time step t-1 is invalid
            bos_mask[:, 0] = ~padding_mask[:, 0]
            bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]

            positions = x.clone()
            x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                                    torch.zeros(num_nodes, 30, 2),
                                    x[:, 20:] - x[:, 19].unsqueeze(-2))
            x[:, 0: 20] = torch.where(padding_mask[:, : 20].unsqueeze(-1),
                                    torch.zeros(num_nodes, 20, 2),
                                    x[:, 0: 20] - x[:, 19].unsqueeze(1))

            # get lane features at the current time step
            df_19 = df[df['TIMESTAMP'] == timestamps[19]]
            node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']]
            node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()
            lane_positions, lane_vectors, lane_lengths, is_intersections, turn_directions, traffic_controls = self.get_lane_features(am, node_inds_19, node_positions_19, origin, rotate_mat, city, radius)

            node_positions_goal = positions[:,-1]
            node_diff_goal = positions[:,-1] - positions[:,-2]
            node_goal_mask = ~padding_mask[:,-1]

            goal_idcs, has_goal = self.get_goal_lane(node_positions_goal, node_diff_goal, node_goal_mask, lane_positions, lane_vectors)

            node_inds_ref = torch.arange(num_nodes, dtype=torch.float)[~padding_mask[:,self.ref_time]]
            lane_positions, lane_vectors, lane_rotate_angles, lane_paddings, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors, lane_lengths, goal_idcs, has_goal = \
                                self.get_lane_tensors(node_inds_ref, node_positions_19, lane_positions, lane_vectors, lane_lengths, is_intersections, turn_directions, traffic_controls, goal_idcs, has_goal, origin, rotate_mat, rotate_angles, radius)
            
            y = None if split == 'test' else x[:, 20:]
            seq_id = os.path.splitext(os.path.basename(raw_fn))[0]

            processed = {
                'x': x[:, : 20],  # [N, 20, 2]
                'positions': positions,  # [N, 50, 2]
                'edge_index': edge_index,  # [2, N x N - 1]
                'y': y,  # [N, 30, 2]
                'num_nodes': num_nodes,
                'padding_mask': padding_mask,  # [N, 50]
                'bos_mask': bos_mask,  # [N, 20]
                'rotate_angles': rotate_angles,  # [N]
                'lane_positions': lane_positions,  # [L, 6, 2]
                'lane_vectors': lane_vectors,  # [L, 2]
                'lane_paddings': lane_paddings, # [L,6]
                'lane_lengths': lane_lengths,
                'is_intersections': is_intersections,  # [L]
                'turn_directions': turn_directions,  # [L]
                'traffic_controls': traffic_controls,  # [L]
                'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
                'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
                'goal_idcs': goal_idcs,
                'has_goal': has_goal,
                'seq_id': int(seq_id),
                'av_index': av_index,
                'agent_index': agent_index,
                'city': city,
                'origin': origin.unsqueeze(0),
                'theta': theta,
            }
        
            data = TemporalData(**processed)
            torch.save(data, os.path.join(self.processed_dir, seq_id+'.pt'))
        return 
    
    
    def get_lane_features(self, am,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
        radius = 80
        lane_positions, lane_vectors, lane_lengths, is_intersections, turn_directions, traffic_controls = [], [], [], [], [], []
        lane_ids = set()
        for node_position in node_positions:
            lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
        node_positions = torch.matmul(node_positions - origin, rotate_mat).float()

        for lane_id in lane_ids:
            lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
            lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)

            is_intersection = am.lane_is_in_intersection(lane_id, city)
            turn_direction = am.get_lane_turn_direction(lane_id, city)
            traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
            if turn_direction == 'NONE':
                turn_direction = 0
            elif turn_direction == 'LEFT':
                turn_direction = 1
            elif turn_direction == 'RIGHT':
                turn_direction = 2
            else:
                raise ValueError('turn direction is not valid')

            line = LineString(lane_centerline)
            total_length = line.length
            new_points = []
            for i in range(int(total_length)):
                # get the point at the i-th distance along the line
                point = line.interpolate(i)
                new_points.append([point.x, point.y])
            new_points = torch.tensor(new_points)

            if len(new_points) < 1:
                continue

            n_segments = int(np.ceil(len(new_points) / (self.lseg_len+1)))
            n_poses = int(np.ceil(len(new_points) / n_segments))
            for n in range(n_segments):
                lane_segment = new_points[n * n_poses: (n+1) * n_poses]
                count = len(lane_segment) - 1
                if count > 0:
                    lane_positions.append((lane_segment[1:] + lane_segment[:-1])/2)
                    lane_vectors.append(lane_segment[1:] - lane_segment[:-1])
                    lane_lengths.append(count)
                    is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
                    turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
                    traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))

        return lane_positions, lane_vectors, lane_lengths, is_intersections, turn_directions, traffic_controls
    
    def get_goal_lane(self, node_positions_goal, node_diff_goal, node_goal_mask, lane_positions, lane_vectors):
        lane_num = len(lane_positions)

        goal_idcs = []
        has_goal = []
        for nidx in range(node_positions_goal.size(0)):
            if not node_goal_mask[nidx]:
                goal_idcs.append(torch.zeros(lane_num))
                has_goal.append(torch.zeros(lane_num))
            else:
                query_pose = node_positions_goal[nidx]
                query_diff = node_diff_goal[nidx]
                query_angle = torch.atan2(query_diff[1], query_diff[0])

                dist_vals = []
                angle_diffs = []
                for lidx in range(len(lane_positions)):
                    lane_poses = lane_positions[lidx]
                    lane_diffs = lane_vectors[lidx]

                    distances = torch.norm(lane_poses - query_pose, p=2, dim=-1)
                    dist_vals.append(torch.min(distances))
                    idx = torch.argmin(distances)

                    lane_angle = torch.atan2(lane_diffs[idx,1], lane_diffs[idx,0])
                    angle_diff = torch.abs(normalize_angle(query_angle-lane_angle))
                    angle_diffs.append(angle_diff)

                if torch.norm(query_diff, p=2, dim=-1) < 0.1: # if difference is too small, angle is within noise.
                    idcs_yaw = torch.arange(lane_num)
                else:
                    idcs_yaw = torch.where(torch.tensor(angle_diffs) <= self.lseg_angle_thres*torch.pi/180)[0]
                idcs_dist = torch.where(torch.tensor(dist_vals) <= self.lseg_dist_thres)[0]
                idcs = np.intersect1d(idcs_dist, idcs_yaw)

                if len(idcs) > 0:
                    assigned_node_id = idcs[int(torch.argmin(torch.tensor(dist_vals)[idcs]))]
                    goal_ = torch.zeros(lane_num)
                    goal_[assigned_node_id] = 1.
                    goal_idcs.append(goal_)

                    has_goal_ = torch.zeros(lane_num)
                    has_goal_[assigned_node_id] = nidx+1 
                    has_goal.append(has_goal_)
                else:
                    goal_idcs.append(torch.zeros(lane_num))
                    has_goal.append(torch.zeros(lane_num))

        return torch.cat(goal_idcs,-1), torch.cat(has_goal,-1)
    
    def get_lane_tensors(self, node_inds: List[int],
                                node_positions: torch.Tensor, # global coordi
                                lane_positions: List[torch.Tensor], # local coordi
                                lane_vectors: List[torch.Tensor],
                                lane_lengths: List[int],
                                is_intersections, turn_directions, traffic_controls,
                                goal_idcs: torch.Tensor,
                                has_goal, 
                                origin: torch.Tensor,
                                rotate_mat: torch.Tensor,
                                rotate_angles,
                                radius: float):

        lane_positions_ = torch.zeros(len(lane_positions), self.lseg_len, 2)
        lane_vectors_ = torch.zeros(len(lane_positions), self.lseg_len, 2)
        lane_padding_ = torch.ones(len(lane_positions), self.lseg_len)
        is_intersections_ = torch.zeros(len(lane_positions), self.lseg_len, 1)
        turn_directions_ = torch.zeros(len(lane_positions), self.lseg_len, 1)
        traffic_controls_ = torch.zeros(len(lane_positions), self.lseg_len, 1)
        for lidx in range(len(lane_positions)):
            lane_length = lane_lengths[lidx]
            lane_positions_[lidx, :lane_length] = lane_positions[lidx]
            lane_vectors_[lidx, :lane_length] = lane_vectors[lidx]
            lane_padding_[lidx, :lane_length] = 0
            is_intersections_[lidx, :lane_length] = is_intersections[lidx].unsqueeze(-1)
            turn_directions_[lidx, :lane_length] = turn_directions[lidx].unsqueeze(-1)
            traffic_controls_[lidx, :lane_length] = traffic_controls[lidx].unsqueeze(-1)
        lane_rotate_angles_ = torch.arctan2(lane_vectors_[:,0,1], lane_vectors_[:,0,0])

        node_positions = torch.matmul(node_positions - origin, rotate_mat)
        lane_actor_index = torch.flip(torch.LongTensor(list(product(node_inds.int(), torch.arange(lane_vectors_.size(0))))).t().contiguous(), [0])
        assert (torch.tensor(lane_lengths) == 0).float().sum() == 0, '0 length lanes are included'
        lane_actor_vectors = \
                lane_positions_[torch.arange(len(lane_positions)),(torch.tensor(lane_lengths)-1).long(),:].repeat(len(node_inds),1) - node_positions.repeat_interleave(lane_vectors_.size(0), dim=0)
        actors_rotate_mat = torch.empty(node_inds.size(0),2,2)
        sin_vals, cos_vals = torch.sin(rotate_angles), torch.cos(rotate_angles)
        actors_rotate_mat[:,0,0] = cos_vals
        actors_rotate_mat[:,0,1] = -sin_vals
        actors_rotate_mat[:,1,0] = sin_vals
        actors_rotate_mat[:,1,1] = cos_vals
        lane_actor_vectors_norm = torch.bmm(lane_actor_vectors.unsqueeze(1), actors_rotate_mat[lane_actor_index[1]]).squeeze(1)
        mask = (-20<lane_actor_vectors_norm[:,0]) & (lane_actor_vectors_norm[:,0]<80) & (-50<lane_actor_vectors_norm[:,1]) & (lane_actor_vectors_norm[:,1]<50)
        
        lane_actor_index = lane_actor_index.reshape(2,-1)[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]
        goal_idcs = goal_idcs[mask]

        has_goal_msk = has_goal[mask]
        has_goal_idx = has_goal_msk[torch.nonzero(has_goal_msk)] - 1 # 

        has_goal_ = torch.zeros_like(node_inds, dtype=torch.bool)
        has_goal_[has_goal_idx.squeeze(-1).long()] = True

        assert goal_idcs.sum() == has_goal_.sum(), f'goal number: {goal_idcs.sum().item()}, has_goal number: {has_goal_.sum().item()}'

        return lane_positions_, lane_vectors_, lane_rotate_angles_, lane_padding_, is_intersections_, turn_directions_, traffic_controls_, lane_actor_index, lane_actor_vectors, \
                torch.LongTensor(lane_lengths), goal_idcs, has_goal_ #, lane_lane_index, lane_lane_vectors
    
def normalize_angle(angle):
    if angle < -torch.pi:
        angle += 2*torch.pi
    elif angle > torch.pi:
        angle -= 2*torch.pi
    return angle

if __name__ == '__main__':
    split = 'val'
    spec_args = {'dataset': 'Argoverse', 'n_jobs': 0, 't_h': 2, 't_f': 3, 'res': 10, 'ref_time':19, 'lseg_len': 10, 'lseg_angle_thres': 30, 'lseg_dist_thres': 2.5}
    A1D = ArgoverseDataset(split, root='data/argodataset', process_dir='preprocessed/Argoverse', spec_args=spec_args)
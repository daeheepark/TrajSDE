import torch.nn as nn
import os
import shutil
import time
import torch

from torchdiffeq import odeint as odeint

#####################################################################################################
class ODE_T_Func(nn.Module):
	def __init__(self, ode_func_net):
		"""
		ode_func_net: neural net that used to transform hidden state in ode
		"""
		super(ODE_T_Func, self).__init__()
		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards = False):
		"""
		Perform one step in solving ODE. Given current data point y and
		current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		return self.gradient_net(y, t_local)

	def sample_next_point_from_prior(self, t_local, y):
		"""
		t_local: current time point
		y: value at the current time point
		"""
		return self.get_ode_gradient_nn(t_local, y)
	
class ODEFunc(nn.Module):
	def __init__(self, ode_func_net):
		"""
		ode_func_net: neural net that used to transform hidden state in ode
		"""
		super(ODEFunc, self).__init__()
		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards = False):
		"""
		Perform one step in solving ODE. Given current data point y and
		current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		return self.gradient_net(y)

	def sample_next_point_from_prior(self, t_local, y):
		"""
		t_local: current time point
		y: value at the current time point
		"""
		return self.get_ode_gradient_nn(t_local, y)


class DiffeqSolver(nn.Module):
	def __init__(self, ode_func, method, odeint_rtol=1e-4, odeint_atol=1e-5):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

	def forward(self, first_point, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver.
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

		pred_y = odeint(self.ode_func, first_point, time_steps_to_predict, 
			rtol = self.odeint_rtol, atol = self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)

		# assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y

	def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver using samples from the prior
		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
		func = self.ode_func.sample_next_point_from_prior

		pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)
		return pred_y
	
class GRU_Unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100):
        super(GRU_Unit, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim))
        init_network_weights(self.new_state_net)


    def forward(self, h_cur, input_tensor, mask):
        y_concat = torch.cat([h_cur, input_tensor], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        new_state = self.new_state_net(combined)

        h_next = (1 - update_gate) * new_state + update_gate * h_cur
	
        # mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
		# mask = mask.unsqueeze(-1)

        h_next = mask.unsqueeze(-1) * h_next + ~mask.unsqueeze(-1) * h_cur

        return h_next
    
class GRU_Unit2(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100):
        super(GRU_Unit, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim))
        init_network_weights(self.new_state_net)


    def forward(self, h_cur, input_tensor, mask):
        y_concat = torch.cat([h_cur, input_tensor], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        new_state = self.new_state_net(combined)

        h_next = (1 - update_gate) * new_state + update_gate * h_cur
	
        # mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
		# mask = mask.unsqueeze(-1)

        h_next = mask.unsqueeze(-1) * h_next + ~mask.unsqueeze(-1) * h_cur

        return h_next
    
def get_timesteps(dataset):
    if dataset == 'Argoverse':
        ref_step = 19
        past_t, future_t = 2, 3
        t_res = 10
    elif dataset == 'nuScenes':
        ref_step = 4
        past_t, future_t = 2, 6
        t_res = 2

    timesteps = torch.arange(0,past_t+future_t, 1/t_res) - past_t + 1/t_res
    timesteps[ref_step] = 0
    return timesteps

def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)
import math
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import pytorch_lightning as pl
import torchsde
from torchsde import sdeint, sdeint_adjoint



class SAdjDiffeqSolver(nn.Module):
	def __init__(self, ode_func, method, dt=0.05, odeint_rtol=1e-4, odeint_atol=1e-5):
		super(SAdjDiffeqSolver, self).__init__()
		self.dt = dt
		self.ode_method = method
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol
                

	def forward(self, first_point, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver.
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

		pred_y = sdeint_adjoint(self.ode_func, first_point, time_steps_to_predict, dt = self.dt,
			rtol = self.odeint_rtol, atol = self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0)

		# assert(torch.mean(pred_y  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y

class SDiffeqSolver(nn.Module):
	def __init__(self, ode_func, method, dt=0.05, odeint_rtol=1e-4, odeint_atol=1e-5):
		super(SDiffeqSolver, self).__init__()
		self.dt = dt
		self.ode_method = method
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol
                

	def forward(self, first_point, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver.
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

		pred_y = sdeint(self.ode_func, first_point, time_steps_to_predict, dt = self.dt,
			rtol = self.odeint_rtol, atol = self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0)

		# assert(torch.mean(pred_y  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y

	def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver using samples from the prior
		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
		func = self.ode_func.sample_next_point_from_prior

		pred_y = sdeint(func, starting_point_enc, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)
		return pred_y

class SDiffeqSolverAug(torchsde.SDEIto):
	def __init__(self, ode_func, method, dt=0.05, odeint_rtol=1e-4, odeint_atol=1e-5):
		super(SDiffeqSolverAug, self).__init__(noise_type="diagonal")
		self.dt = dt
		self.ode_method = method
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol
                

	def forward(self, first_point, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver.
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
                
		fitst_point_aug = torch.cat([first_point, torch.zeros(n_traj_samples, 1).to(first_point)], dim=1)

		pred_y_aug = sdeint(self.ode_func, fitst_point_aug, time_steps_to_predict, dt = self.dt,
			rtol = self.odeint_rtol, atol = self.odeint_atol, method = self.ode_method,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'})
        
		pred_y = pred_y_aug[:,:,:-1].permute(1,2,0)
		logqp_path = pred_y_aug[-1,:,-1]

		# assert(torch.mean(pred_y  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y, logqp_path.mean(dim=0)

class SDEFunc(nn.Module):
    def __init__(self, f, g, order=1):
        super().__init__()  
        self.order, self.intloss, self.sensitivity = order, None, None
        self.f_func, self.g_func = f, g
        self.fnfe, self.gnfe = 0, 0

    def forward(self, s, x):
        pass
    
    def f(self, s, x):
        """Posterior drift."""
        self.fnfe += 1
        return self.f_func(x)
    
    def g(self, s, x):
        """Diffusion"""
        self.gnfe += 1
        return self.g_func(x).diag_embed()

class LSDEFunc(torchsde.SDEIto):
    def __init__(self, f, g, h, order=1):
        super().__init__(noise_type="diagonal")
        self.order, self.intloss, self.sensitivity = order, None, None
        self.f_func, self.g_func, self.h_func = f, g, h
        self.fnfe, self.gnfe, self.hnfe = 0, 0, 0

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
        return self.g_func(t=s, y=x)


class LSDEFuncAug(torchsde.SDEIto):
    def __init__(self, f, g, h, order=1):
        super().__init__(noise_type="diagonal")
        self.order, self.intloss, self.sensitivity = order, None, None
        self.f_func, self.g_func, self.h_func = f, g, h
        self.fnfe, self.gnfe, self.hnfe = 0, 0, 0

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
        return self.g_func(t=s, y=x)
    
    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:,:-1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:,:-1]
        g = self.g(t, y)
        g_logqp = torch.zeros(y.size(0),1).to(y)
        return torch.cat([g, g_logqp], dim=1)
    
def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b
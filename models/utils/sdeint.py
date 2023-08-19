import warnings

import torch
import abc
from torch import nn

from torchsde._core import base_sde
from torchsde._core import methods
from torchsde._core import misc
from torchsde._brownian import BaseBrownian, BrownianInterval
from torchsde.settings import LEVY_AREA_APPROXIMATIONS, METHODS, NOISE_TYPES, SDE_TYPES
from torchsde.types import Any, Dict, Tuple, Optional, Scalar, Tensor, Tensors, TensorOrTensors, Vector

from torchsde._core import base_solver
from torchsde.settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS

from torchsde._core.base_sde import BaseSDE

from torchsde._core.methods.euler import Euler
from torchsde._core.base_solver import interp, adaptive_stepping

def sdeint(sde,
           y0: Tensor,
           ts: Vector,
           nus_mask: Tensor,
           bm: Optional[BaseBrownian] = None,
           method: Optional[str] = None,
           dt: Scalar = 1e-3,
           adaptive: bool = False,
           rtol: Scalar = 1e-5,
           atol: Scalar = 1e-4,
           dt_min: Scalar = 1e-5,
           options: Optional[Dict[str, Any]] = None,
           names: Optional[Dict[str, str]] = None,
           logqp: bool = False,
           extra: bool = False,
           extra_solver_state: Optional[Tensors] = None,
           **unused_kwargs) -> TensorOrTensors:
    """Numerically integrate an SDE.

    Args:
        sde: Object with methods `f` and `g` representing the
            drift and diffusion. The output of `g` should be a single tensor of
            size (batch_size, d) for diagonal noise SDEs or (batch_size, d, m)
            for SDEs of other noise types; d is the dimensionality of state and
            m is the dimensionality of Brownian motion.
        y0 (Tensor): A tensor for the initial state.
        ts (Tensor or sequence of float): Query times in non-descending order.
            The state at the first time of `ts` should be `y0`.
        bm (Brownian, optional): A 'BrownianInterval', `BrownianPath` or
            `BrownianTree` object. Should return tensors of size (batch_size, m)
            for `__call__`. Defaults to `BrownianInterval`.
        method (str, optional): Numerical integration method to use. Must be
            compatible with the SDE type (Ito/Stratonovich) and the noise type
            (scalar/additive/diagonal/general). Defaults to a sensible choice
            depending on the SDE type and noise type of the supplied SDE.
        dt (float, optional): The constant step size or initial step size for
            adaptive time-stepping.
        adaptive (bool, optional): If `True`, use adaptive time-stepping.
        rtol (float, optional): Relative tolerance.
        atol (float, optional): Absolute tolerance.
        dt_min (float, optional): Minimum step size during integration.
        options (dict, optional): Dict of options for the integration method.
        names (dict, optional): Dict of method names for drift and diffusion.
            Expected keys are "drift" and "diffusion". Serves so that users can
            use methods with names not in `("f", "g")`, e.g. to use the
            method "foo" for the drift, we supply `names={"drift": "foo"}`.
        logqp (bool, optional): If `True`, also return the log-ratio penalty.
        extra (bool, optional): If `True`, also return the extra hidden state
            used internally in the solver.
        extra_solver_state: (tuple of Tensors, optional): Additional state to
            initialise the solver with. Some solvers keep track of additional
            state besides y0, and this offers a way to optionally initialise
            that state.

    Returns:
        A single state tensor of size (T, batch_size, d).
        if logqp is True, then the log-ratio penalty is also returned.
        If extra is True, the any extra internal state of the solver is also
        returned.

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method,
            or if `sde` is missing required methods.
    """
    misc.handle_unused_kwargs(unused_kwargs, msg="`sdeint`")
    del unused_kwargs

    sde, y0, ts, bm, method, options = check_contract(sde, y0, ts, nus_mask, bm, method, adaptive, options, names, logqp)
    misc.assert_no_grad(['ts', 'dt', 'rtol', 'atol', 'dt_min'],
                        [ts, dt, rtol, atol, dt_min])

    solver_fn = Euler
    solver = solver_fn(
        sde=sde,
        bm=bm,
        dt=dt,
        adaptive=adaptive,
        rtol=rtol,
        atol=atol,
        dt_min=dt_min,
        options=options
    )
    if extra_solver_state is None:
        extra_solver_state = solver.init_extra_solver_state(ts[0], y0)
    ys, extra_solver_state = solver.integrate(y0, ts, extra_solver_state)

    return parse_return(y0, ys, extra_solver_state, extra, logqp)

def sdeint_dual(sde,
           y0: Tensor,
           ts: Vector,
           nus_mask: Tensor,
           bm: Optional[BaseBrownian] = None,
           method: Optional[str] = None,
           dt: Scalar = 1e-3,
           adaptive: bool = False,
           rtol: Scalar = 1e-5,
           atol: Scalar = 1e-4,
           dt_min: Scalar = 1e-5,
           options: Optional[Dict[str, Any]] = None,
           names: Optional[Dict[str, str]] = None,
           logqp: bool = False,
           extra: bool = False,
           extra_solver_state: Optional[Tensors] = None,
           **unused_kwargs) -> TensorOrTensors:
    """Numerically integrate an SDE.

    Args:
        sde: Object with methods `f` and `g` representing the
            drift and diffusion. The output of `g` should be a single tensor of
            size (batch_size, d) for diagonal noise SDEs or (batch_size, d, m)
            for SDEs of other noise types; d is the dimensionality of state and
            m is the dimensionality of Brownian motion.
        y0 (Tensor): A tensor for the initial state.
        ts (Tensor or sequence of float): Query times in non-descending order.
            The state at the first time of `ts` should be `y0`.
        bm (Brownian, optional): A 'BrownianInterval', `BrownianPath` or
            `BrownianTree` object. Should return tensors of size (batch_size, m)
            for `__call__`. Defaults to `BrownianInterval`.
        method (str, optional): Numerical integration method to use. Must be
            compatible with the SDE type (Ito/Stratonovich) and the noise type
            (scalar/additive/diagonal/general). Defaults to a sensible choice
            depending on the SDE type and noise type of the supplied SDE.
        dt (float, optional): The constant step size or initial step size for
            adaptive time-stepping.
        adaptive (bool, optional): If `True`, use adaptive time-stepping.
        rtol (float, optional): Relative tolerance.
        atol (float, optional): Absolute tolerance.
        dt_min (float, optional): Minimum step size during integration.
        options (dict, optional): Dict of options for the integration method.
        names (dict, optional): Dict of method names for drift and diffusion.
            Expected keys are "drift" and "diffusion". Serves so that users can
            use methods with names not in `("f", "g")`, e.g. to use the
            method "foo" for the drift, we supply `names={"drift": "foo"}`.
        logqp (bool, optional): If `True`, also return the log-ratio penalty.
        extra (bool, optional): If `True`, also return the extra hidden state
            used internally in the solver.
        extra_solver_state: (tuple of Tensors, optional): Additional state to
            initialise the solver with. Some solvers keep track of additional
            state besides y0, and this offers a way to optionally initialise
            that state.

    Returns:
        A single state tensor of size (T, batch_size, d).
        if logqp is True, then the log-ratio penalty is also returned.
        If extra is True, the any extra internal state of the solver is also
        returned.

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method,
            or if `sde` is missing required methods.
    """
    misc.handle_unused_kwargs(unused_kwargs, msg="`sdeint`")
    del unused_kwargs

    sde, y0, ts, bm, method, options = check_contract(sde, y0, ts, nus_mask, bm, method, adaptive, options, names, logqp)
    misc.assert_no_grad(['ts', 'dt', 'rtol', 'atol', 'dt_min'],
                        [ts, dt, rtol, atol, dt_min])

    # solver_fn = methods.select(method=method, sde_type=sde.sde_type)
    solver_fn = Euler_private
    solver = solver_fn(
        sde=sde,
        bm=bm,
        dt=dt,
        adaptive=adaptive,
        rtol=rtol,
        atol=atol,
        dt_min=dt_min,
        options=options
    )
    if extra_solver_state is None:
        extra_solver_state = solver.init_extra_solver_state(ts[0], y0)
    ys, extra_solver_state, diff_noise = solver.integrate(y0, ts, nus_mask, extra_solver_state)

    return parse_return(y0, ys, extra_solver_state, extra, logqp), diff_noise


def parse_return(y0, ys, extra_solver_state, extra, logqp):
    if logqp:
        ys, log_ratio = ys.split(split_size=(y0.size(1) - 1, 1), dim=2)
        log_ratio_increments = torch.stack(
            [log_ratio_t_plus_1 - log_ratio_t
             for log_ratio_t_plus_1, log_ratio_t in zip(log_ratio[1:], log_ratio[:-1])], dim=0
        ).squeeze(dim=2)

        if extra:
            return ys, log_ratio_increments, extra_solver_state
        else:
            return ys, log_ratio_increments
    else:
        if extra:
            return ys, extra_solver_state
        else:
            return ys
        
def sdeint_all_dual(sde,
           y0: Tensor,
           ts: Vector,
           nus_mask: Tensor,
           bm: Optional[BaseBrownian] = None,
           method: Optional[str] = None,
           dt: Scalar = 1e-3,
           adaptive: bool = False,
           rtol: Scalar = 1e-5,
           atol: Scalar = 1e-4,
           dt_min: Scalar = 1e-5,
           options: Optional[Dict[str, Any]] = None,
           names: Optional[Dict[str, str]] = None,
           logqp: bool = False,
           extra: bool = False,
           extra_solver_state: Optional[Tensors] = None,
           **unused_kwargs) -> TensorOrTensors:
    """Numerically integrate an SDE.

    Args:
        sde: Object with methods `f` and `g` representing the
            drift and diffusion. The output of `g` should be a single tensor of
            size (batch_size, d) for diagonal noise SDEs or (batch_size, d, m)
            for SDEs of other noise types; d is the dimensionality of state and
            m is the dimensionality of Brownian motion.
        y0 (Tensor): A tensor for the initial state.
        ts (Tensor or sequence of float): Query times in non-descending order.
            The state at the first time of `ts` should be `y0`.
        bm (Brownian, optional): A 'BrownianInterval', `BrownianPath` or
            `BrownianTree` object. Should return tensors of size (batch_size, m)
            for `__call__`. Defaults to `BrownianInterval`.
        method (str, optional): Numerical integration method to use. Must be
            compatible with the SDE type (Ito/Stratonovich) and the noise type
            (scalar/additive/diagonal/general). Defaults to a sensible choice
            depending on the SDE type and noise type of the supplied SDE.
        dt (float, optional): The constant step size or initial step size for
            adaptive time-stepping.
        adaptive (bool, optional): If `True`, use adaptive time-stepping.
        rtol (float, optional): Relative tolerance.
        atol (float, optional): Absolute tolerance.
        dt_min (float, optional): Minimum step size during integration.
        options (dict, optional): Dict of options for the integration method.
        names (dict, optional): Dict of method names for drift and diffusion.
            Expected keys are "drift" and "diffusion". Serves so that users can
            use methods with names not in `("f", "g")`, e.g. to use the
            method "foo" for the drift, we supply `names={"drift": "foo"}`.
        logqp (bool, optional): If `True`, also return the log-ratio penalty.
        extra (bool, optional): If `True`, also return the extra hidden state
            used internally in the solver.
        extra_solver_state: (tuple of Tensors, optional): Additional state to
            initialise the solver with. Some solvers keep track of additional
            state besides y0, and this offers a way to optionally initialise
            that state.

    Returns:
        A single state tensor of size (T, batch_size, d).
        if logqp is True, then the log-ratio penalty is also returned.
        If extra is True, the any extra internal state of the solver is also
        returned.

    Raises:
        ValueError: An error occurred due to unrecognized noise type/method,
            or if `sde` is missing required methods.
    """
    misc.handle_unused_kwargs(unused_kwargs, msg="`sdeint`")
    del unused_kwargs

    sde, y0, ts, bm, method, options = check_contract_all(sde, y0, ts, nus_mask, bm, method, adaptive, options, names, logqp)
    misc.assert_no_grad(['ts', 'dt', 'rtol', 'atol', 'dt_min'],
                        [ts, dt, rtol, atol, dt_min])

    # solver_fn = methods.select(method=method, sde_type=sde.sde_type)
    solver_fn = Euler_private
    solver = solver_fn(
        sde=sde,
        bm=bm,
        dt=dt,
        adaptive=adaptive,
        rtol=rtol,
        atol=atol,
        dt_min=dt_min,
        options=options
    )
    if extra_solver_state is None:
        extra_solver_state = solver.init_extra_solver_state(ts[0], y0)
    ys, extra_solver_state, diff_noise = solver.integrate(y0, ts, nus_mask, extra_solver_state)

    return parse_return(y0, ys, extra_solver_state, extra, logqp), diff_noise


def parse_return(y0, ys, extra_solver_state, extra, logqp):
    if logqp:
        ys, log_ratio = ys.split(split_size=(y0.size(1) - 1, 1), dim=2)
        log_ratio_increments = torch.stack(
            [log_ratio_t_plus_1 - log_ratio_t
             for log_ratio_t_plus_1, log_ratio_t in zip(log_ratio[1:], log_ratio[:-1])], dim=0
        ).squeeze(dim=2)

        if extra:
            return ys, log_ratio_increments, extra_solver_state
        else:
            return ys, log_ratio_increments
    else:
        if extra:
            return ys, extra_solver_state
        else:
            return ys

class BaseSDESolver_private(base_solver.BaseSDESolver):
    def integrate(self, y0: Tensor, ts: Tensor, nus_mask: Tensors, extra0: Tensors) -> Tuple[Tensor, Tensors]:
        """Integrate along trajectory.

        Args:
            y0: Tensor of size (batch_size, d)
            ts: Tensor of size (T,).
            extra0: Any extra state for the solver.

        Returns:
            ys, where ys is a Tensor of size (T, batch_size, d).
            extra_solver_state, which is a tuple of Tensors of shape (T, ...), where ... is arbitrary and
                solver-dependent.
        """
        step_size = self.dt

        prev_t = curr_t = ts[0]
        prev_y = curr_y = y0
        curr_extra = extra0

        ys = [y0]
        prev_error_ratio = None

        for out_t in ts[1:]:
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                if self.adaptive:
                    # Take 1 full step.
                    next_y_full, _ = self.step(curr_t, next_t, curr_y, curr_extra)
                    # Take 2 half steps.
                    midpoint_t = 0.5 * (curr_t + next_t)
                    midpoint_y, midpoint_extra = self.step(curr_t, midpoint_t, curr_y, curr_extra)
                    next_y, next_extra = self.step(midpoint_t, next_t, midpoint_y, midpoint_extra)

                    # Estimate error based on difference between 1 full step and 2 half steps.
                    with torch.no_grad():
                        error_estimate = adaptive_stepping.compute_error(next_y_full, next_y, self.rtol, self.atol)
                        step_size, prev_error_ratio = adaptive_stepping.update_step_size(
                            error_estimate=error_estimate,
                            prev_step_size=step_size,
                            prev_error_ratio=prev_error_ratio
                        )

                    if step_size < self.dt_min:
                        warnings.warn("Hitting minimum allowed step size in adaptive time-stepping.")
                        step_size = self.dt_min
                        prev_error_ratio = None

                    # Accept step.
                    if error_estimate <= 1 or step_size <= self.dt_min:
                        prev_t, prev_y = curr_t, curr_y
                        curr_t, curr_y, curr_extra = next_t, next_y, next_extra
                else:
                    prev_t, prev_y = curr_t, curr_y
                    curr_y, curr_extra, g_prod = self.step(curr_t, next_t, curr_y, nus_mask, curr_extra)
                    curr_t = next_t
            ys.append(interp.linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        return torch.stack(ys, dim=0), curr_extra, g_prod
    
class BaseSDESolver_private2(base_solver.BaseSDESolver):
    def integrate(self, y0: Tensor, ts: Tensor, nus_mask: Tensors, extra0: Tensors) -> Tuple[Tensor, Tensors]:
        """Integrate along trajectory.

        Args:
            y0: Tensor of size (batch_size, d)
            ts: Tensor of size (T,).
            extra0: Any extra state for the solver.

        Returns:
            ys, where ys is a Tensor of size (T, batch_size, d).
            extra_solver_state, which is a tuple of Tensors of shape (T, ...), where ... is arbitrary and
                solver-dependent.
        """
        step_size = self.dt

        prev_t = curr_t = ts[0]
        prev_y = curr_y = y0
        curr_extra = extra0

        ys = [y0]
        prev_error_ratio = None

        for out_t in ts[1:]:
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                if self.adaptive:
                    # Take 1 full step.
                    next_y_full, _ = self.step(curr_t, next_t, curr_y, curr_extra)
                    # Take 2 half steps.
                    midpoint_t = 0.5 * (curr_t + next_t)
                    midpoint_y, midpoint_extra = self.step(curr_t, midpoint_t, curr_y, curr_extra)
                    next_y, next_extra = self.step(midpoint_t, next_t, midpoint_y, midpoint_extra)

                    # Estimate error based on difference between 1 full step and 2 half steps.
                    with torch.no_grad():
                        error_estimate = adaptive_stepping.compute_error(next_y_full, next_y, self.rtol, self.atol)
                        step_size, prev_error_ratio = adaptive_stepping.update_step_size(
                            error_estimate=error_estimate,
                            prev_step_size=step_size,
                            prev_error_ratio=prev_error_ratio
                        )

                    if step_size < self.dt_min:
                        warnings.warn("Hitting minimum allowed step size in adaptive time-stepping.")
                        step_size = self.dt_min
                        prev_error_ratio = None

                    # Accept step.
                    if error_estimate <= 1 or step_size <= self.dt_min:
                        prev_t, prev_y = curr_t, curr_y
                        curr_t, curr_y, curr_extra = next_t, next_y, next_extra
                else:
                    prev_t, prev_y = curr_t, curr_y
                    curr_y, curr_extra = self.step(curr_t, next_t, curr_y, curr_extra)
                    curr_t = next_t
            ys.append(interp.linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        g_prod = curr_y
        return torch.stack(ys, dim=0), curr_extra, g_prod

# class Euler(BaseSDESolver_private2):
#     weak_order = 1.0
#     sde_type = SDE_TYPES.ito
#     noise_types = NOISE_TYPES.all()
#     levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

#     def __init__(self, sde, **kwargs):
#         self.strong_order = 1.0 if sde.noise_type == NOISE_TYPES.additive else 0.5
#         super(Euler, self).__init__(sde=sde, **kwargs)

#     def step(self, t0, t1, y0, extra0):
#         del extra0
#         dt = t1 - t0
#         I_k = self.bm(t0, t1)

#         f, g_prod = self.sde.f_and_g_prod(t0, y0, I_k)

#         y1 = y0 + f * dt + g_prod
#         return y1, ()

class Euler_private(BaseSDESolver_private):
    weak_order = 1.0
    sde_type = SDE_TYPES.ito
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, sde, **kwargs):
        self.strong_order = 1.0 if sde.noise_type == NOISE_TYPES.additive else 0.5
        super(Euler_private, self).__init__(sde=sde, **kwargs)

    def step(self, t0, t1, y0, nus_mask, extra0):
        del extra0
        dt = t1 - t0
        I_k = self.bm(t0, t1)

        f, g, g_prod = self.sde.f_and_g_prod(t0, y0, nus_mask, I_k)

        y1 = y0 + f * dt + g_prod
        return y1, (), g


class ForwardSDE_private(BaseSDE):

    def __init__(self, sde, fast_dg_ga_jvp_column_sum=False):
        super(ForwardSDE_private, self).__init__(sde_type=sde.sde_type, noise_type=sde.noise_type)
        self._base_sde = sde

        # Register the core functions. This avoids polluting the codebase with if-statements and achieves speed-ups
        # by making sure it's a one-time cost.

        if hasattr(sde, 'f_and_g_prod'):
            self.f_and_g_prod = sde.f_and_g_prod
        elif hasattr(sde, 'f') and hasattr(sde, 'g_prod'):
            self.f_and_g_prod = self.f_and_g_prod_default1
        else:  # (f_and_g,) or (f, g,).
            self.f_and_g_prod = self.f_and_g_prod_default2

        self.f = getattr(sde, 'f', self.f_default)
        self.g = getattr(sde, 'g', self.g_default)
        self.f_and_g = getattr(sde, 'f_and_g', self.f_and_g_default)
        self.g_prod = getattr(sde, 'g_prod', self.g_prod_default)
        self.prod = {
            NOISE_TYPES.diagonal: self.prod_diagonal
        }.get(sde.noise_type, self.prod_default)
        self.g_prod_and_gdg_prod = {
            NOISE_TYPES.diagonal: self.g_prod_and_gdg_prod_diagonal,
            NOISE_TYPES.additive: self.g_prod_and_gdg_prod_additive,
        }.get(sde.noise_type, self.g_prod_and_gdg_prod_default)
        self.dg_ga_jvp_column_sum = {
            NOISE_TYPES.general: (
                self.dg_ga_jvp_column_sum_v2 if fast_dg_ga_jvp_column_sum else self.dg_ga_jvp_column_sum_v1
            )
        }.get(sde.noise_type, self._return_zero)

    ########################################
    #                  f                   #
    ########################################
    def f_default(self, t, y):
        raise RuntimeError("Method `f` has not been provided, but is required for this method.")

    ########################################
    #                  g                   #
    ########################################
    def g_default(self, t, y):
        raise RuntimeError("Method `g` has not been provided, but is required for this method.")

    ########################################
    #               f_and_g                #
    ########################################

    def f_and_g_default(self, t, y, nus_mask):
        return self.f(t, y), self.g(t, y, nus_mask)

    ########################################
    #                prod                  #
    ########################################

    def prod_diagonal(self, g, v):
        return g * v

    def prod_default(self, g, v):
        return misc.batch_mvp(g, v)

    ########################################
    #                g_prod                #
    ########################################

    def g_prod_default(self, t, y, v):
        return self.prod(self.g(t, y), v)

    ########################################
    #             f_and_g_prod             #
    ########################################

    def f_and_g_prod_default1(self, t, y, v):
        return self.f(t, y), self.g_prod(t, y, v)

    def f_and_g_prod_default2(self, t, y, nus_mask, v):
        f, g = self.f_and_g(t, y, nus_mask)
        return f, g, self.prod(g, v)

    ########################################
    #          g_prod_and_gdg_prod         #
    ########################################

    # Computes: g_prod and sum_{j, l} g_{j, l} d g_{j, l} d x_i v2_l.
    def g_prod_and_gdg_prod_default(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v2.unsqueeze(-2),
                retain_graph=True,
                create_graph=requires_grad,
                allow_unused=True
            )
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_diagonal(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v2,
                retain_graph=True,
                create_graph=requires_grad,
                allow_unused=True
            )
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_additive(self, t, y, v1, v2):
        return self.g_prod(t, y, v1), 0.

    ########################################
    #              dg_ga_jvp               #
    ########################################

    # Computes: sum_{j,k,l} d g_{i,l} / d x_j g_{j,k} A_{k,l}.
    def dg_ga_jvp_column_sum_v1(self, t, y, a):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)
            dg_ga_jvp = [
                misc.jvp(
                    outputs=g[..., col_idx],
                    inputs=y,
                    grad_inputs=ga[..., col_idx],
                    retain_graph=True,
                    create_graph=requires_grad,
                    allow_unused=True
                )[0]
                for col_idx in range(g.size(-1))
            ]
            dg_ga_jvp = sum(dg_ga_jvp)
        return dg_ga_jvp

    def dg_ga_jvp_column_sum_v2(self, t, y, a):
        # Faster, but more memory intensive.
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)

            batch_size, d, m = g.size()
            y_dup = torch.repeat_interleave(y, repeats=m, dim=0)
            g_dup = self.g(t, y_dup)
            ga_flat = ga.transpose(1, 2).flatten(0, 1)
            dg_ga_jvp, = misc.jvp(
                outputs=g_dup,
                inputs=y_dup,
                grad_inputs=ga_flat,
                create_graph=requires_grad,
                allow_unused=True
            )
            dg_ga_jvp = dg_ga_jvp.reshape(batch_size, m, d, m).permute(0, 2, 1, 3)
            dg_ga_jvp = dg_ga_jvp.diagonal(dim1=-2, dim2=-1).sum(-1)
        return dg_ga_jvp

    def _return_zero(self, t, y, v):  # noqa
        return 0.
    
class ForwardSDE_private_all(BaseSDE):

    def __init__(self, sde, fast_dg_ga_jvp_column_sum=False):
        super(ForwardSDE_private, self).__init__(sde_type=sde.sde_type, noise_type=sde.noise_type)
        self._base_sde = sde

        # Register the core functions. This avoids polluting the codebase with if-statements and achieves speed-ups
        # by making sure it's a one-time cost.

        if hasattr(sde, 'f_and_g_prod'):
            self.f_and_g_prod = sde.f_and_g_prod
        elif hasattr(sde, 'f') and hasattr(sde, 'g_prod'):
            self.f_and_g_prod = self.f_and_g_prod_default1
        else:  # (f_and_g,) or (f, g,).
            self.f_and_g_prod = self.f_and_g_prod_default2

        self.f = getattr(sde, 'f', self.f_default)
        self.g = getattr(sde, 'g', self.g_default)
        self.f_and_g = getattr(sde, 'f_and_g', self.f_and_g_default)
        self.g_prod = getattr(sde, 'g_prod', self.g_prod_default)
        self.prod = {
            NOISE_TYPES.diagonal: self.prod_diagonal
        }.get(sde.noise_type, self.prod_default)
        self.g_prod_and_gdg_prod = {
            NOISE_TYPES.diagonal: self.g_prod_and_gdg_prod_diagonal,
            NOISE_TYPES.additive: self.g_prod_and_gdg_prod_additive,
        }.get(sde.noise_type, self.g_prod_and_gdg_prod_default)
        self.dg_ga_jvp_column_sum = {
            NOISE_TYPES.general: (
                self.dg_ga_jvp_column_sum_v2 if fast_dg_ga_jvp_column_sum else self.dg_ga_jvp_column_sum_v1
            )
        }.get(sde.noise_type, self._return_zero)

    ########################################
    #                  f                   #
    ########################################
    def f_default(self, t, y):
        raise RuntimeError("Method `f` has not been provided, but is required for this method.")

    ########################################
    #                  g                   #
    ########################################
    def g_default(self, t, y):
        raise RuntimeError("Method `g` has not been provided, but is required for this method.")

    ########################################
    #               f_and_g                #
    ########################################

    def f_and_g_default(self, t, y, nus_mask):
        return self.f(t, y, nus_mask), self.g(t, y, nus_mask)

    ########################################
    #                prod                  #
    ########################################

    def prod_diagonal(self, g, v):
        return g * v

    def prod_default(self, g, v):
        return misc.batch_mvp(g, v)

    ########################################
    #                g_prod                #
    ########################################

    def g_prod_default(self, t, y, v):
        return self.prod(self.g(t, y), v)

    ########################################
    #             f_and_g_prod             #
    ########################################

    def f_and_g_prod_default1(self, t, y, v):
        return self.f(t, y), self.g_prod(t, y, v)

    def f_and_g_prod_default2(self, t, y, nus_mask, v):
        f, g = self.f_and_g(t, y, nus_mask)
        return f, g, self.prod(g, v)

    ########################################
    #          g_prod_and_gdg_prod         #
    ########################################

    # Computes: g_prod and sum_{j, l} g_{j, l} d g_{j, l} d x_i v2_l.
    def g_prod_and_gdg_prod_default(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v2.unsqueeze(-2),
                retain_graph=True,
                create_graph=requires_grad,
                allow_unused=True
            )
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_diagonal(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v2,
                retain_graph=True,
                create_graph=requires_grad,
                allow_unused=True
            )
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_additive(self, t, y, v1, v2):
        return self.g_prod(t, y, v1), 0.

    ########################################
    #              dg_ga_jvp               #
    ########################################

    # Computes: sum_{j,k,l} d g_{i,l} / d x_j g_{j,k} A_{k,l}.
    def dg_ga_jvp_column_sum_v1(self, t, y, a):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)
            dg_ga_jvp = [
                misc.jvp(
                    outputs=g[..., col_idx],
                    inputs=y,
                    grad_inputs=ga[..., col_idx],
                    retain_graph=True,
                    create_graph=requires_grad,
                    allow_unused=True
                )[0]
                for col_idx in range(g.size(-1))
            ]
            dg_ga_jvp = sum(dg_ga_jvp)
        return dg_ga_jvp

    def dg_ga_jvp_column_sum_v2(self, t, y, a):
        # Faster, but more memory intensive.
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)

            batch_size, d, m = g.size()
            y_dup = torch.repeat_interleave(y, repeats=m, dim=0)
            g_dup = self.g(t, y_dup)
            ga_flat = ga.transpose(1, 2).flatten(0, 1)
            dg_ga_jvp, = misc.jvp(
                outputs=g_dup,
                inputs=y_dup,
                grad_inputs=ga_flat,
                create_graph=requires_grad,
                allow_unused=True
            )
            dg_ga_jvp = dg_ga_jvp.reshape(batch_size, m, d, m).permute(0, 2, 1, 3)
            dg_ga_jvp = dg_ga_jvp.diagonal(dim1=-2, dim2=-1).sum(-1)
        return dg_ga_jvp

    def _return_zero(self, t, y, v):  # noqa
        return 0.
    

def check_contract(sde, y0, ts, nus_mask, bm, method, adaptive, options, names, logqp):
    if names is None:
        names_to_change = {}
    else:
        names_to_change = {key: names[key] for key in ("drift", "diffusion", "prior_drift", "drift_and_diffusion",
                                                       "drift_and_diffusion_prod") if key in names}
    if len(names_to_change) > 0:
        sde = base_sde.RenameMethodsSDE(sde, **names_to_change)

    if not hasattr(sde, "noise_type"):
        raise ValueError(f"sde does not have the attribute noise_type.")

    if sde.noise_type not in NOISE_TYPES:
        raise ValueError(f"Expected noise type in {NOISE_TYPES}, but found {sde.noise_type}.")

    if not hasattr(sde, "sde_type"):
        raise ValueError(f"sde does not have the attribute sde_type.")

    if sde.sde_type not in SDE_TYPES:
        raise ValueError(f"Expected sde type in {SDE_TYPES}, but found {sde.sde_type}.")

    if not torch.is_tensor(y0):
        raise ValueError("`y0` must be a torch.Tensor.")
    if y0.dim() != 2:
        raise ValueError("`y0` must be a 2-dimensional tensor of shape (batch, channels).")

    # --- Backwards compatibility: v0.1.1. ---
    if logqp:
        sde = base_sde.SDELogqp(sde)
        y0 = torch.cat((y0, y0.new_zeros(size=(y0.size(0), 1))), dim=1)
    # ----------------------------------------

    if method is None:
        method = {
            SDE_TYPES.ito: {
                NOISE_TYPES.diagonal: METHODS.srk,
                NOISE_TYPES.additive: METHODS.srk,
                NOISE_TYPES.scalar: METHODS.srk,
                NOISE_TYPES.general: METHODS.euler
            }[sde.noise_type],
            SDE_TYPES.stratonovich: METHODS.midpoint,
        }[sde.sde_type]

    if method not in METHODS:
        raise ValueError(f"Expected method in {METHODS}, but found {method}.")

    if not torch.is_tensor(ts):
        if not isinstance(ts, (tuple, list)) or not all(isinstance(t, (float, int)) for t in ts):
            raise ValueError(f"Evaluation times `ts` must be a 1-D Tensor or list/tuple of floats.")
        ts = torch.tensor(ts, dtype=y0.dtype, device=y0.device)
    if not misc.is_strictly_increasing(ts):
        raise ValueError("Evaluation times `ts` must be strictly increasing.")

    batch_sizes = []
    state_sizes = []
    noise_sizes = []
    batch_sizes.append(y0.size(0))
    state_sizes.append(y0.size(1))
    if bm is not None:
        if len(bm.shape) != 2:
            raise ValueError("`bm` must be of shape (batch, noise_channels).")
        batch_sizes.append(bm.shape[0])
        noise_sizes.append(bm.shape[1])

    def _check_2d(name, shape):
        if len(shape) != 2:
            raise ValueError(f"{name} must be of shape (batch, state_channels), but got {shape}.")
        batch_sizes.append(shape[0])
        state_sizes.append(shape[1])

    def _check_2d_or_3d(name, shape):
        if sde.noise_type == NOISE_TYPES.diagonal:
            if len(shape) != 2:
                raise ValueError(f"{name} must be of shape (batch, state_channels), but got {shape}.")
            batch_sizes.append(shape[0])
            state_sizes.append(shape[1])
            noise_sizes.append(shape[1])
        else:
            if len(shape) != 3:
                raise ValueError(f"{name} must be of shape (batch, state_channels, noise_channels), but got {shape}.")
            batch_sizes.append(shape[0])
            state_sizes.append(shape[1])
            noise_sizes.append(shape[2])

    has_f = False
    has_g = False
    if hasattr(sde, 'f'):
        has_f = True
        f_drift_shape = tuple(sde.f(ts[0], y0).size())
        _check_2d('Drift', f_drift_shape)
    if hasattr(sde, 'g'):
        has_g = True
        g_diffusion_shape = tuple(sde.g(ts[0], y0, nus_mask).size())
        # g_diffusion_shape = tuple(sde.g(ts[0], y0).size())
        _check_2d_or_3d('Diffusion', g_diffusion_shape)
    if hasattr(sde, 'f_and_g'):
        has_f = True
        has_g = True
        _f, _g = sde.f_and_g(ts[0], y0)
        f_drift_shape = tuple(_f.size())
        g_diffusion_shape = tuple(_g.size())
        _check_2d('Drift', f_drift_shape)
        _check_2d_or_3d('Diffusion', g_diffusion_shape)
    if hasattr(sde, 'g_prod'):
        has_g = True
        if len(noise_sizes) == 0:
            raise ValueError("Cannot infer noise size (i.e. number of Brownian motion channels). Either pass `bm` "
                             "explicitly, or specify one of the `g`, `f_and_g` functions.`")
        v = torch.randn(batch_sizes[0], noise_sizes[0], dtype=y0.dtype, device=y0.device)
        g_prod_shape = tuple(sde.g_prod(ts[0], y0, v).size())
        _check_2d('Diffusion-vector product', g_prod_shape)
    if hasattr(sde, 'f_and_g_prod'):
        has_f = True
        has_g = True
        if len(noise_sizes) == 0:
            raise ValueError("Cannot infer noise size (i.e. number of Brownian motion channels). Either pass `bm` "
                             "explicitly, or specify one of the `g`, `f_and_g` functions.`")
        v = torch.randn(batch_sizes[0], noise_sizes[0], dtype=y0.dtype, device=y0.device)
        _f, _g_prod = sde.f_and_g_prod(ts[0], y0, v)
        f_drift_shape = tuple(_f.size())
        g_prod_shape = tuple(_g_prod.size())
        _check_2d('Drift', f_drift_shape)
        _check_2d('Diffusion-vector product', g_prod_shape)

    if not has_f:
        raise ValueError("sde must define at least one of `f`, `f_and_g`, or `f_and_g_prod`. (Or possibly more "
                         "depending on the method chosen.)")
    if not has_g:
        raise ValueError("sde must define at least one of `g`, `f_and_g`, `g_prod` or `f_and_g_prod`. (Or possibly "
                         "more depending on the method chosen.)")

    for batch_size in batch_sizes[1:]:
        if batch_size != batch_sizes[0]:
            raise ValueError("Batch sizes not consistent.")
    for state_size in state_sizes[1:]:
        if state_size != state_sizes[0]:
            raise ValueError("State sizes not consistent.")
    for noise_size in noise_sizes[1:]:
        if noise_size != noise_sizes[0]:
            raise ValueError("Noise sizes not consistent.")

    if sde.noise_type == NOISE_TYPES.scalar:
        if noise_sizes[0] != 1:
            raise ValueError(f"Scalar noise must have only one channel; the diffusion has {noise_sizes[0]} noise "
                             f"channels.")

    sde = ForwardSDE_private(sde)
    # sde = ForwardSDE_private_debug(sde)

    if bm is None:
        if method == METHODS.srk:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.space_time
        elif method == METHODS.log_ode_midpoint:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.foster
        else:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.none
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], size=(batch_sizes[0], noise_sizes[0]), dtype=y0.dtype,
                              device=y0.device, levy_area_approximation=levy_area_approximation)

    if options is None:
        options = {}
    else:
        options = options.copy()

    if adaptive and method == METHODS.euler and sde.noise_type != NOISE_TYPES.additive:
        warnings.warn(f"Numerical solution is not guaranteed to converge to the correct solution when using adaptive "
                      f"time-stepping with the Euler--Maruyama method with non-additive noise.")

    return sde, y0, ts, bm, method, options


def check_contract_all(sde, y0, ts, nus_mask, bm, method, adaptive, options, names, logqp):
    if names is None:
        names_to_change = {}
    else:
        names_to_change = {key: names[key] for key in ("drift", "diffusion", "prior_drift", "drift_and_diffusion",
                                                       "drift_and_diffusion_prod") if key in names}
    if len(names_to_change) > 0:
        sde = base_sde.RenameMethodsSDE(sde, **names_to_change)

    if not hasattr(sde, "noise_type"):
        raise ValueError(f"sde does not have the attribute noise_type.")

    if sde.noise_type not in NOISE_TYPES:
        raise ValueError(f"Expected noise type in {NOISE_TYPES}, but found {sde.noise_type}.")

    if not hasattr(sde, "sde_type"):
        raise ValueError(f"sde does not have the attribute sde_type.")

    if sde.sde_type not in SDE_TYPES:
        raise ValueError(f"Expected sde type in {SDE_TYPES}, but found {sde.sde_type}.")

    if not torch.is_tensor(y0):
        raise ValueError("`y0` must be a torch.Tensor.")
    if y0.dim() != 2:
        raise ValueError("`y0` must be a 2-dimensional tensor of shape (batch, channels).")

    # --- Backwards compatibility: v0.1.1. ---
    if logqp:
        sde = base_sde.SDELogqp(sde)
        y0 = torch.cat((y0, y0.new_zeros(size=(y0.size(0), 1))), dim=1)
    # ----------------------------------------

    if method is None:
        method = {
            SDE_TYPES.ito: {
                NOISE_TYPES.diagonal: METHODS.srk,
                NOISE_TYPES.additive: METHODS.srk,
                NOISE_TYPES.scalar: METHODS.srk,
                NOISE_TYPES.general: METHODS.euler
            }[sde.noise_type],
            SDE_TYPES.stratonovich: METHODS.midpoint,
        }[sde.sde_type]

    if method not in METHODS:
        raise ValueError(f"Expected method in {METHODS}, but found {method}.")

    if not torch.is_tensor(ts):
        if not isinstance(ts, (tuple, list)) or not all(isinstance(t, (float, int)) for t in ts):
            raise ValueError(f"Evaluation times `ts` must be a 1-D Tensor or list/tuple of floats.")
        ts = torch.tensor(ts, dtype=y0.dtype, device=y0.device)
    if not misc.is_strictly_increasing(ts):
        raise ValueError("Evaluation times `ts` must be strictly increasing.")

    batch_sizes = []
    state_sizes = []
    noise_sizes = []
    batch_sizes.append(y0.size(0))
    state_sizes.append(y0.size(1))
    if bm is not None:
        if len(bm.shape) != 2:
            raise ValueError("`bm` must be of shape (batch, noise_channels).")
        batch_sizes.append(bm.shape[0])
        noise_sizes.append(bm.shape[1])

    def _check_2d(name, shape):
        if len(shape) != 2:
            raise ValueError(f"{name} must be of shape (batch, state_channels), but got {shape}.")
        batch_sizes.append(shape[0])
        state_sizes.append(shape[1])

    def _check_2d_or_3d(name, shape):
        if sde.noise_type == NOISE_TYPES.diagonal:
            if len(shape) != 2:
                raise ValueError(f"{name} must be of shape (batch, state_channels), but got {shape}.")
            batch_sizes.append(shape[0])
            state_sizes.append(shape[1])
            noise_sizes.append(shape[1])
        else:
            if len(shape) != 3:
                raise ValueError(f"{name} must be of shape (batch, state_channels, noise_channels), but got {shape}.")
            batch_sizes.append(shape[0])
            state_sizes.append(shape[1])
            noise_sizes.append(shape[2])

    has_f = False
    has_g = False
    if hasattr(sde, 'f'):
        has_f = True
        f_drift_shape = tuple(sde.f(ts[0], y0).size())
        _check_2d('Drift', f_drift_shape)
    if hasattr(sde, 'g'):
        has_g = True
        g_diffusion_shape = tuple(sde.g(ts[0], y0, nus_mask).size())
        # g_diffusion_shape = tuple(sde.g(ts[0], y0).size())
        _check_2d_or_3d('Diffusion', g_diffusion_shape)
    if hasattr(sde, 'f_and_g'):
        has_f = True
        has_g = True
        _f, _g = sde.f_and_g(ts[0], y0)
        f_drift_shape = tuple(_f.size())
        g_diffusion_shape = tuple(_g.size())
        _check_2d('Drift', f_drift_shape)
        _check_2d_or_3d('Diffusion', g_diffusion_shape)
    if hasattr(sde, 'g_prod'):
        has_g = True
        if len(noise_sizes) == 0:
            raise ValueError("Cannot infer noise size (i.e. number of Brownian motion channels). Either pass `bm` "
                             "explicitly, or specify one of the `g`, `f_and_g` functions.`")
        v = torch.randn(batch_sizes[0], noise_sizes[0], dtype=y0.dtype, device=y0.device)
        g_prod_shape = tuple(sde.g_prod(ts[0], y0, v).size())
        _check_2d('Diffusion-vector product', g_prod_shape)
    if hasattr(sde, 'f_and_g_prod'):
        has_f = True
        has_g = True
        if len(noise_sizes) == 0:
            raise ValueError("Cannot infer noise size (i.e. number of Brownian motion channels). Either pass `bm` "
                             "explicitly, or specify one of the `g`, `f_and_g` functions.`")
        v = torch.randn(batch_sizes[0], noise_sizes[0], dtype=y0.dtype, device=y0.device)
        _f, _g_prod = sde.f_and_g_prod(ts[0], y0, v)
        f_drift_shape = tuple(_f.size())
        g_prod_shape = tuple(_g_prod.size())
        _check_2d('Drift', f_drift_shape)
        _check_2d('Diffusion-vector product', g_prod_shape)

    if not has_f:
        raise ValueError("sde must define at least one of `f`, `f_and_g`, or `f_and_g_prod`. (Or possibly more "
                         "depending on the method chosen.)")
    if not has_g:
        raise ValueError("sde must define at least one of `g`, `f_and_g`, `g_prod` or `f_and_g_prod`. (Or possibly "
                         "more depending on the method chosen.)")

    for batch_size in batch_sizes[1:]:
        if batch_size != batch_sizes[0]:
            raise ValueError("Batch sizes not consistent.")
    for state_size in state_sizes[1:]:
        if state_size != state_sizes[0]:
            raise ValueError("State sizes not consistent.")
    for noise_size in noise_sizes[1:]:
        if noise_size != noise_sizes[0]:
            raise ValueError("Noise sizes not consistent.")

    if sde.noise_type == NOISE_TYPES.scalar:
        if noise_sizes[0] != 1:
            raise ValueError(f"Scalar noise must have only one channel; the diffusion has {noise_sizes[0]} noise "
                             f"channels.")

    sde = ForwardSDE_private_all(sde)

    if bm is None:
        if method == METHODS.srk:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.space_time
        elif method == METHODS.log_ode_midpoint:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.foster
        else:
            levy_area_approximation = LEVY_AREA_APPROXIMATIONS.none
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], size=(batch_sizes[0], noise_sizes[0]), dtype=y0.dtype,
                              device=y0.device, levy_area_approximation=levy_area_approximation)

    if options is None:
        options = {}
    else:
        options = options.copy()

    if adaptive and method == METHODS.euler and sde.noise_type != NOISE_TYPES.additive:
        warnings.warn(f"Numerical solution is not guaranteed to converge to the correct solution when using adaptive "
                      f"time-stepping with the Euler--Maruyama method with non-additive noise.")

    return sde, y0, ts, bm, method, options
"""Utility functions for running inference for diffusion models."""
import abc
from collections import deque
from typing import Callable

import sympy
import torch
from torchdiffeq import odeint

from diffusion import utils
from diffusion.denoisers import Denoiser

Tensor = torch.Tensor
SigmaFn = Callable[[Tensor], Tensor]


def _append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


# -----------------------------------------------------------------------------
# Noise schedules.
# -----------------------------------------------------------------------------
# Noise schedules determine sequences of t, sigma, and log-SNR values used as
# discretization points for solving the reverse diffusion ODE.
# -----------------------------------------------------------------------------


class BaseNoiseSchedule(abc.ABC):
    """Abstract base class for noise schedules."""

    @abc.abstractmethod
    def sigma_fn(self, t: Tensor) -> Tensor:
        """Defines element-wise function sigma(t). Must be implemented by subclasses."""

    @abc.abstractmethod
    def get_t_schedule(self, n: int) -> tuple[Tensor, Tensor]:
        """Rerturns a tensor of time steps and sigma0. Must be implemented by subclasses."""

    @abc.abstractmethod
    def get_sigma_schedule(self, n: int) -> tuple[Tensor, Tensor]:
        """Rerturns a tensor of sigma steps and sigma0. Must be implemented by subclasses."""

    @abc.abstractmethod
    def get_logsnr_schedule(self, n: int) -> tuple[Tensor, Tensor]:
        """Rerturns a tensor of log-SNR steps and sigma0. Must be implemented by subclasses."""


class KarrasNoiseSchedule(BaseNoiseSchedule):
    """Specifies noise schedule proposed by Karras et al. (2022).

    The schedule is defined in terms of sigma (Eq. 5 in the paper):
        sigma_i = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho, i=0,...,n-1,
        sigma_n = 0.

    Reference: https://arxiv.org/abs/2206.00364.
    """

    def __init__(
        self, sigma_data: float, sigma_min: float, sigma_max: float, rho: float = 7.0, device="cpu"
    ):
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.device = device

        # Precompute some constants.
        self.sigma_min_inv_rho = self.sigma_min ** (1 / self.rho)
        self.sigma_max_inv_rho = self.sigma_max ** (1 / self.rho)

    def sigma_fn(self, t: Tensor) -> Tensor:
        """Defines element-wise function sigma(t) = t."""
        return t

    def get_sigma_schedule(self, n: int) -> tuple[Tensor, Tensor]:
        """Rerturns a tensor of sigma steps."""
        steps = torch.linspace(0, 1, n)
        sigma = (
            self.sigma_max_inv_rho + steps * (self.sigma_min_inv_rho - self.sigma_max_inv_rho)
        ) ** self.rho
        sigma = _append_zero(sigma).to(self.device)
        return sigma, sigma[0]

    def get_t_schedule(self, n_steps: int) -> tuple[Tensor, Tensor]:
        """Returns a tensor of time steps calculated as t = sigma_inv(sigma)."""
        return self.get_sigma_schedule(n_steps)

    def get_logsnr_schedule(self, n: int) -> tuple[Tensor, Tensor]:
        """Rerturns a tensor of log-SNR steps computed from sigma."""
        sigma, sigma0 = self.get_sigma_schedule(n)
        return 2 * torch.log(self.sigma_data / sigma), sigma0


class LinearLogSnrNoiseSchedule(BaseNoiseSchedule):
    """Specifies a schedule linear in the log-SNR space."""

    def __init__(self, sigma_data: float, logsnr_min: float, logsnr_max: float, device="cpu"):
        self.sigma_data = sigma_data
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.device = device

    def sigma_fn(self, t: Tensor) -> Tensor:
        """Defines element-wise function sigma(t) = t."""
        return t

    def get_logsnr_schedule(self, n: int) -> tuple[Tensor, Tensor]:
        """Rerturns a tensor of log-SNR steps."""
        steps = torch.linspace(0, 1, n)
        logsnr = (self.logsnr_min + steps * (self.logsnr_max - self.logsnr_min)).to(self.device)
        sigma0 = self.sigma_data * torch.exp(-logsnr[0] / 2)
        return logsnr, sigma0

    def get_sigma_schedule(self, n: int) -> tuple[Tensor, Tensor]:
        """Rerturns a tensor of sigma steps computed from log-SNR."""
        logsnr, sigma0 = self.get_logsnr_schedule(n)
        return self.sigma_data * torch.exp(-logsnr / 2), sigma0

    def get_t_schedule(self, n_steps: int) -> tuple[Tensor, Tensor]:
        """Returns a tensor of time steps calculated as t = sigma_inv(sigma)."""
        return self.get_sigma_schedule(n_steps)


# -----------------------------------------------------------------------------
# ODE equations.
# -----------------------------------------------------------------------------
# Given a learned denoising model, we run infernece to generate samples by
# solving a reverse diffusion ODE. The are two ways to formulate these ODEs:
#
#   1)  The classic formulation defines ODE in the time domain and requires
#       specifying a time-dependent noise schedule sigma(t).
#   2)  Alternatively, it is possible to change variables, define ODE in the
#       log-SNR domain and solve it directly by integrating over log-SNR.
# -----------------------------------------------------------------------------


class BaseDiffEq(abc.ABC):
    """Abstract base class for ODEs."""

    @abc.abstractmethod
    def x_to_sigma(self, x: Tensor) -> Tensor:
        """Defines element-wise function that maps x to sigma.
        Must be implemented by subclasses.
        """

    @abc.abstractmethod
    def sigma_to_x(self, sigma: Tensor) -> Tensor:
        """Defines element-wise function that maps sigma to x.
        Must be implemented by subclasses.
        """

    @abc.abstractmethod
    def dy_dx(self, x: Tensor, y: Tensor) -> Tensor:
        """Computes dy/dx for the specified x and y.
        Must be implemented by subclasses.
        """


class KarrasDiffEq(BaseDiffEq):
    """Implements reverse diffusion ODE from Karras et al. (2022).

    The ODE is defined as follows (Eq. 1 in the paper):
        dy/dt = sigma'(t) / sigma(t) * (y - D(y, sigma(t))),
        where sigma'(t) := d sigma(t) / dt.

    The sigma(t) function is set to be sigma(t) = t.

    Args:
        denoiser: A denoising model.
        t_to_sigma: An element-wise function that maps t to sigma.
        sigma_to_t: An element-wise function that maps sigma to t.

    NOTE:
        Karras et al. (2022) also defined a version of the ODE with
        time-dependent signal scaling (Eq. 4 in the paper), which allows
        to implement variance preserving (VP) version of the ODE. Here,
        we consider only the simpler formulation where signal scaling is 1,
        i.e., the variance exploding (VE) version of the ODE, which also
        works better in practice anyway.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        t_to_sigma: Callable[[Tensor], Tensor],
        sigma_to_t: Callable[[Tensor], Tensor],
    ) -> None:
        self.denoiser = denoiser
        self._t_to_sigma = t_to_sigma
        self._sigma_to_t = sigma_to_t

    def x_to_sigma(self, x: Tensor) -> Tensor:
        return self._t_to_sigma(x)

    def sigma_to_x(self, sigma: Tensor) -> Tensor:
        return self._sigma_to_t(sigma)

    def dy_dx(self, x: Tensor, y: Tensor) -> Tensor:
        """Computes dy/dx for the specified x and y, where x is supposed to be time."""
        assert x.numel() == 1, "dy_dx expects a single x value as input."
        batch_size = y.shape[0]

        # Compute sigma and d sigma / dt for each time point in the batch.
        # NOTE: we assume that sigma_fn is an element-wise function.
        t = x.repeat(batch_size).detach().requires_grad_()  # shape: [batch_size]
        t = utils.expand_dims(t, y.ndim)  # shape: [batch_size, 1, ...]
        sigma = self.x_to_sigma(t)  # shape: [batch_size, 1, ...]
        dsigma_dt = torch.autograd.grad(sigma.sum(), t)[0]  # shape: [batch_size, 1, ...]

        # Compute dy/dx.
        with torch.inference_mode():
            dy_dx = (dsigma_dt / sigma) * (y - self.denoiser(y, sigma.squeeze()))

        return dy_dx


class LogSnrDiffEq(BaseDiffEq):
    """Implements reverse diffusion ODE from Karras et al. (2022) converted to the the log-SNR domain.

    The ODE after the change of variables takes the following simple form
    (modified Eq. 1 from the paper):
        dy/dx = (D(y, sigma(x)) - y) / 2,
        where x is defined as log-SNR: x := 2 log(sigma_data / sigma), which
        implies that sigma(x) = sigma_data * exp(-x / 2).
    """

    def __init__(self, denoiser: Denoiser) -> None:
        self.denoiser = denoiser

    def x_to_sigma(self, x: Tensor) -> Tensor:
        return utils.logsnr_to_sigma(x, self.denoiser.sigma_data)

    def sigma_to_x(self, sigma: Tensor) -> Tensor:
        return utils.sigma_to_logsnr(sigma, self.denoiser.sigma_data)

    @torch.inference_mode()
    def dy_dx(self, x: Tensor, y: Tensor) -> Tensor:
        """Computes dy/dx for the specified x and y, where x is supposed to be log-SNR."""
        assert x.numel() == 1, "dy_dx expects a single x value as input."
        batch_size = y.shape[0]

        # Compute sigma from log-SNR.
        sigma = self.x_to_sigma(x).repeat(batch_size)

        # Compute dy/dx.
        return (self.denoiser(y, sigma) - y) / 2


# -----------------------------------------------------------------------------
# ODE solvers.
# -----------------------------------------------------------------------------


class BaseDiffEqSolver(abc.ABC):
    """Abstract base class for ODE solvers.

    Subclasses must implement _solve method. This class wraps _solve and takes
    care of the last step of the trajectory, which requires special handling
    when sigma = 0: the dy_dx derivative is not defined at sigma = 0, so we
    use Euler method for the last step, as proposed by Karras et al. (2022).
    """

    def _euler_step(self, x: Tensor, dx: Tensor, y: Tensor, ode: BaseDiffEq) -> Tensor:
        """Computes Euler step."""
        dy_dx = ode.dy_dx(x, y)
        return y + dy_dx * dx

    @abc.abstractmethod
    def _solve(self, x: Tensor, y0: Tensor, ode: BaseDiffEq) -> Tensor:
        """Implements integration of the specified spcified ODE over x and returns the y trajectory.

        Must be implemented by subclasses.
        """

    def solve(
        self, x: Tensor, y0: Tensor, ode: BaseDiffEq, euler_last_step: bool | None = None
    ) -> Tensor:
        """Integrates the specified spcified ODE over x and returns the y trajectory."""
        # If the last step of the trajectory corresponds to sigma = 0, use Euler method for the last step.
        if euler_last_step is None:
            sigma_last = ode.x_to_sigma(x[-1]).item()
            euler_last_step = sigma_last == 0

        # Preapre x pairs for iteration.
        if euler_last_step:
            x, x_last_pair = x[:-1], (x[-2], x[-1])

        # Run integration.
        trajectory = self._solve(x, y0, ode)

        # Special case the last step.
        if euler_last_step:
            y_last = trajectory[-1]
            x_i, x_ip1 = x_last_pair
            y = self._euler_step(x_i, x_ip1 - x_i, y_last, ode)
            trajectory = torch.cat([trajectory, y.unsqueeze(0)], dim=0)

        return trajectory


class TorchDiffEqOdeintSolver(BaseDiffEqSolver):
    """A thin wrapper around `torchdiffeq.odeint`.

    Reference: https://github.com/rtqichen/torchdiffeq.
    """

    def __init__(self, rtol=1e-7, atol=1e-9, method=None, options=None) -> None:
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.options = options

    def _solve(self, x: Tensor, y0: Tensor, ode: BaseDiffEq) -> Tensor:
        """Integrates the specified spcified ODE over x and returns the y trajectory."""
        trajectory = odeint(
            func=ode.dy_dx,
            y0=y0,
            t=x,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            options=self.options,
        )

        return trajectory


class KarrasHeun2Solver(BaseDiffEqSolver):
    """Implements Heun's 2nd order method from Karras et al. (2022), Algorithm 1.

    Reference: https://arxiv.org/abs/2206.00364.
    """

    @staticmethod
    def _euler_step_correct(x: Tensor, dx: Tensor, y: Tensor, ode: BaseDiffEq) -> Tensor:
        """Computes Euler step with 2nd order correction."""
        dy_dx = ode.dy_dx(x, y)
        # Appy 2nd order correction.
        dy_dx_new = ode.dy_dx(x + dx, y + dy_dx * dx)
        return y + 0.5 * (dy_dx + dy_dx_new) * dx

    def _solve(self, x: Tensor, y0: Tensor, ode: BaseDiffEq) -> Tensor:
        trajectory = [y0]

        # Apply Euler step with 2nd order correction.
        y = y0
        for x_i, x_ip1 in zip(x[:-1], x[1:]):
            y = self._euler_step_correct(x_i, x_ip1 - x_i, y, ode)
            trajectory.append(y)

        return torch.stack(trajectory, dim=0)


class DPMppDiffEqSolver(BaseDiffEqSolver):
    """Implments DPM-Solver++ for diffusion ODE from Lu et al. (2022).

    This solver exploits the semi-linear structure of the diffusion ODE
    and does the following:
        1)  changes variables and converts ODE to the log-SNR domain,
        2)  analytically integrates the linear part of the ODE,
        3)  uses Taylor expansion of 1st, 2nd, or 3rd order to approximate
            the non-linear part of the ODE.

    DPM-Solver++ is not a black-box solver and requires knowing the exact form of the ODE.
    Here, we implement the solver for LogSnrDiffEq, which is the log-SNR version of
    the diffusion ODE from Karras et al. (2022):
        dy/dx = (D(y, sigma(x)) - y) / 2,
        where x is defined as log-SNR: x := 2 log(sigma_data / sigma).

    The solution of this ODE when integrating from x1 to x2 is given by:
        y(x2) = exp((x1 - x2)/2) * y(x1) +
                (1/2) * int_{x1}^{x2} exp((x - x2)/2) D(y(x), sigma(x)) dx.

    The integral is approximated by K terms of Taylor expansion:
        sum_{n=0}^{k-1} D^n(y(x1), sigma(x1)) * I(n),
        where I(n) := int_{x1}^{x2} exp((x - x2)/2) (x - x1)^n / n! dx,
        and D^n(y(x1), sigma(x1)) is the n-th derivative of D(y, sigma) at x1.

    The integral I(n) can be computed analytically (in the code, we compute them using sympy).
    And the derivatives D^n(y(x1), sigma(x1)) is computed using single-step or multi-step
    methods as proposed in the paper by Lu et al. (2022).

    NOTE: our implementation differs from the original one in the following ways:
        - we define log-SNR as `2 log(sigma_data / sigma)` instead of `log(sigma_data / sigma)`,
        - the ODE we are solving is the log-SNR of the Karras et al. (2022) ODE,
        - we use sympy for computing Taylor expansion coefficients automatically.
    As a result, the math in our code is sligtly different and the overall implementation
    turns out to be much simpler and more readable.

    References:
        - DPM-Solver: https://arxiv.org/abs/2206.00927.
        - DPM-Solver++: https://arxiv.org/abs/2211.01095.
        - Original code: https://github.com/LuChengTHU/dpm-solver.
    """

    def __init__(
        self, order: int = 2, multistep: bool = True, lower_order_final: bool = True
    ) -> None:
        self.order = order
        self.multistep = multistep
        self.lower_order_final = lower_order_final

        # Pre-compute expressions for I(n) coefficients in the Taylor expansion
        # of the non-linear part of the ODE solution.
        self._taylor_coeffs = self._compute_taylor_coeff_exprs()

    def _compute_taylor_coeff_exprs(self) -> list[sympy.Expr]:
        """Computes Taylor expansion coefficients for the specified order using sympy."""
        taylor_coeffs = {"vars": {}, "exprs": []}

        # Define symbols.
        x, x1, x2 = sympy.symbols("x x1 x2")
        taylor_coeffs["vars"]["x1"] = x1
        taylor_coeffs["vars"]["x2"] = x2

        # Compute Taylor expansion coefficients.
        for n in range(self.order):
            I_expr = sympy.integrate(
                sympy.exp((x - x2) / 2) * (x - x1) ** n / sympy.factorial(n), (x, x1, x2)
            )
            taylor_coeffs["exprs"].append(I_expr)

        return taylor_coeffs

    @staticmethod
    def _compute_denoised(x: Tensor, y: Tensor, ode: LogSnrDiffEq) -> Tensor:
        """Computes denoised y value at the specified x value."""
        assert x.numel() == 1, "_compute_denoised expects a single x value as input."
        batch_size = y.shape[0]

        # Compute sigma from log-SNR.
        sigma = ode.x_to_sigma(x).repeat(batch_size)

        return ode.denoiser(y, sigma)

    def _solve(self, x: Tensor, y0: Tensor, ode: LogSnrDiffEq) -> Tensor:
        """Depending on the settings, calls either single-step or multi-step method."""
        if self.multistep:
            return self._solve_multistep(x, y0, ode)
        else:
            return self._solve_singlestep(x, y0, ode)

    @torch.inference_mode()
    def solve(
        self, x: Tensor, y0: Tensor, ode: BaseDiffEq, euler_last_step: bool | None = None
    ) -> Tensor:
        """Sanity checks that ODE is LogSnrDiffEq and calls sovle from the parent class."""
        if not isinstance(ode, LogSnrDiffEq):
            raise ValueError("DPMppDiffEqSolver can only be used to solve LogSnrDiffEq.")
        return super().solve(x, y0, ode, euler_last_step)

    # --- Multi-step method ---------------------------------------------------

    @staticmethod
    def _compute_multistep_denoiser_derivatives(
        x_d_buffer: deque, order: int
    ) -> tuple[Tensor, ...]:
        """Computes derivatives of the denoiser at the specified order using multi-step method."""
        if order == 1:
            # Already computed denoised value is the 0-th order derivative.
            _, d1 = x_d_buffer[0]
            return (d1,)
        elif order == 2:
            x1, d1 = x_d_buffer[0]
            x0, d0 = x_d_buffer[1]
            # Compute first order derivative.
            dd1 = (d1 - d0) / (x1 - x0)
            return d1, dd1
        elif order == 3:
            x1, d1 = x_d_buffer[0]
            x0, d0 = x_d_buffer[1]
            xm1, dm1 = x_d_buffer[2]
            # Compute first order derivatives at x1 and x0.
            dd1_1 = (d1 - d0) / (x1 - x0)
            dd1_0 = (d0 - dm1) / (x0 - xm1)
            # Compute first order derivative using a 3-point formula.
            # NOTE: formula copied from the original implementation; double check derivation.
            dd1 = dd1_1 + (dd1_1 - dd1_0) * (x1 - x0) / (x1 - xm1)
            # Compute second order derivative.
            d2d1 = (dd1_1 - dd1_0) / (x1 - xm1)
            return d1, dd1, d2d1
        else:
            raise ValueError(f"Unsupported order: {order}.")

    def _compute_multistep_update(
        self, x: Tensor, y: Tensor, x_d_buffer: deque[tuple[Tensor, Tensor]], order: int
    ):
        """Computes DPM-Solver++ update of the specified order."""
        # Rename variables to match our notation.
        x1, _ = x_d_buffer[0]  # The first element in the buffer is the most recent one.
        x2, y1 = x, y

        # Compute the linear part of the update: exp((x1 - x2)/2) * y(x1).
        y2 = torch.exp((x1 - x2) / 2) * y1

        # Compute the non-linear part of the update.
        d_derivatives = self._compute_multistep_denoiser_derivatives(x_d_buffer, order=order)
        for n in range(order):
            # Compute values for the Taylor coefficients using sympy.
            coeff_i = self._taylor_coeffs["exprs"][n].subs(
                {
                    self._taylor_coeffs["vars"]["x1"]: x1.item(),
                    self._taylor_coeffs["vars"]["x2"]: x2.item(),
                }
            )
            y2 += float(coeff_i) * d_derivatives[n] / 2

        return y2

    def _solve_multistep(self, x: Tensor, y0: Tensor, ode: LogSnrDiffEq) -> Tensor:
        # Sanity check.
        n_steps = x.shape[0]
        if n_steps < self.order:
            raise ValueError(
                f"Number of steps must be at least {self.order} for the multi-step method."
            )

        # Initialize circular buffers for the specified order of the multistep solver.
        x_d_buffer = deque(maxlen=self.order)
        x_i, d_i = x[0], self._compute_denoised(x[0], y0, ode=ode)
        x_d_buffer.appendleft((x_i, d_i))

        # Run integration.
        trajectory = [y0]
        y_i = y0
        for n in range(1, n_steps):
            if n < self.order:
                # Use lower order method for the first few steps.
                order = n
            elif self.lower_order_final and n_steps < 10:
                # Use lower order method for the last few steps if total number of steps < 10.
                # This trick is important for stabilizing sampling with very few steps.
                order = min(self.order, n_steps - n)
            else:
                order = self.order
            x_i = x[n]
            y_i = self._compute_multistep_update(x_i, y_i, x_d_buffer, order=order)
            d_i = self._compute_denoised(x_i, y_i, ode=ode)
            x_d_buffer.appendleft((x_i, d_i))
            trajectory.append(y_i)

        return torch.stack(trajectory, dim=0)

    # --- Single-step method --------------------------------------------------

    def _solve_singlestep(self, x: Tensor, y0: Tensor, ode: LogSnrDiffEq) -> Tensor:
        # TODO: implement single-step method.
        raise NotImplementedError("Single-step method is not implemented yet.")


# -----------------------------------------------------------------------------
# SDE solvers.
# -----------------------------------------------------------------------------


class KarrasStochasticDiffEqSolver(BaseDiffEqSolver):
    """A custom SDE solver proposed by Karras et al. (2022).

    This solver uses an arbitrary ODE solver as a sub-routine. The algorithm does
    the following:
        1. Inject a little bit of noise into the input.
        2. Run the ODE solver for one step to denoise the input with the extra noise.
        3. Repeat steps 1 and 2 for n_steps steps.

    Args:
        ode_solver: An ODE solver to use as a sub-routine.
        s_churn: Determines how much sigma / log-SNR is adjusted by noise injection.
        s_noise: An additional multiplier that scales injected noise (typically, close to 1.0).
        s_x_min: The lower bound on the step at which noise is injected.
        s_x_max: The upper bound on the step at which noise is injected.

    Reference: https://arxiv.org/abs/2206.00364.
    """

    def __init__(
        self,
        ode_solver: BaseDiffEqSolver,
        s_churn: float,
        s_noise: float,
        s_x_min: float,
        s_x_max: float,
    ) -> None:
        self.ode_solver = ode_solver
        self.s_churn = s_churn
        self.s_noise = s_noise
        self.s_min = s_x_min
        self.s_max = s_x_max

        # Placeholders for variables computed during integration.
        self.gamma = None

    def _inject_noise(self, x: Tensor, y: Tensor, ode: BaseDiffEq) -> tuple[Tensor, Tensor]:
        # Convert x to sigma.
        sigma = ode.x_to_sigma(x)

        # Do not inject noise if x is outside of the specified range.
        if sigma < self.s_min or sigma > self.s_max:
            return x, y

        # Inject noise into the current sample (Agorithm 2, lines 5-6).
        sigma_hat = (1 + self.gamma) * sigma
        eps = self.s_noise * torch.randn_like(y)
        y_hat = y + torch.sqrt(sigma_hat**2 - sigma**2) * eps

        # Convert back sigma to x.
        x_hat = ode.sigma_to_x(sigma_hat)

        return x_hat, y_hat

    def _euler_step(self, x: Tensor, dx: Tensor, y: Tensor, ode: BaseDiffEq) -> Tensor:
        """Computes Euler step after injecting noise."""
        x_hat, y_hat = self._inject_noise(x, y, ode)
        dx = dx + x - x_hat  # Adjust dx to account for noise injection.
        return super()._euler_step(x_hat, dx, y_hat, ode)

    def _solve(self, x: Tensor, y0: Tensor, ode: BaseDiffEq) -> Tensor:
        trajectory = [y0]

        # Compute gamma (Algorithm 2, comment next to line 3).
        n_steps = x.shape[0]
        self.gamma = min(self.s_churn / n_steps, 2**0.5 - 1)

        # Integrate the SDE over the specified steps.
        y = y0
        for x_i, x_ip1 in zip(x[:-1], x[1:]):
            # Inject noise into the current sample.
            x_i_hat, y_hat = self._inject_noise(x_i, y, ode)
            # Run the ODE solver to x_ip1.
            y_i_traj = self.ode_solver.solve(
                x=torch.stack([x_i_hat, x_ip1]),
                y0=y_hat,
                ode=ode,
                euler_last_step=False,
            )
            y = y_i_traj[-1]
            trajectory.append(y)

        return torch.stack(trajectory, dim=0)


# -----------------------------------------------------------------------------


def sample(
    input_shape: tuple[int, ...],
    ode: BaseDiffEq,
    solver: BaseDiffEqSolver,
    noise_schedule: BaseNoiseSchedule,
    batch_size: int,
    n_steps: int,
    device="cpu",
) -> tuple[Tensor, list[Tensor]]:
    """Generates samples using the specified ODE and solver, and noise schedule."""
    # Generate grid.
    if isinstance(ode, KarrasDiffEq):
        x, sigma0 = noise_schedule.get_t_schedule(n_steps)
    elif isinstance(ode, LogSnrDiffEq):
        x, sigma0 = noise_schedule.get_logsnr_schedule(n_steps)
    else:
        raise ValueError(f"Unknown ODE type: {type(ode)}")

    # Generate a batch of initial noise.
    y0 = sigma0 * torch.randn((batch_size,) + input_shape, device=device)

    # Transfer denosing model to the specified device.
    ode.denoiser.to(device)

    # Run solver.
    trajectory = solver.solve(x, y0, ode)

    return trajectory

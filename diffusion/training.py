"""Common utility functions for training diffusion models."""
import abc
import functools
import math
from typing import Callable

import torch

from diffusion.denoisers import Denoiser, KarrasDenoiser
from diffusion.utils import expand_dims

Tensor = torch.Tensor
LossWeightFn = Callable[[Tensor, float], Tensor]  # (sigma, sigma_data) -> Tensor


# -----------------------------------------------------------------------------
# Loss weighting schemes for different noise levels.
# -----------------------------------------------------------------------------
# Assigning non-uniform weights to different noise levels has been shown to
# improve performance of the resulting diffusion models (Hang et al., 2022).
# TODO: add different VDM weighting schemes from Kingma et al. (2022).
# -----------------------------------------------------------------------------


def _loss_weighting_uniform(sigma, sigma_data):
    """Uniform weighting scheme that assigns equal weights to all noise levels."""
    del sigma_data  # Unused.
    return torch.ones_like(sigma)


def _loss_weighting_snr(sigma, sigma_data):
    """Weighting function that assigns weights proportional to the signal-to-noise ratio."""
    return (sigma_data / sigma) ** 2


def _loss_weighting_min_snr_gamma(sigma, sigma_data, gamma=5.0):
    """Weighting function based on the min-SNR-gamma weighting scheme from Hang et al. (2022).

    Reference: https://arxiv.org/abs/2303.09556.
    """
    snr = (sigma_data / sigma) ** 2
    return torch.minimum(snr, torch.ones_like(snr) * gamma)


def _loss_weighting_soft_min_snr(sigma, sigma_data):
    """Weighting function based on the soft-min-SNR: 4 * SNR / (1 + SNR) ** 2."""
    snr = (sigma_data / sigma) ** 2
    return 4 * snr / (1 + snr) ** 2


WEIGHTING_SCHEMES = {
    "uniform": _loss_weighting_uniform,
    "snr": _loss_weighting_snr,
    "soft-min-snr": _loss_weighting_soft_min_snr,
    "min-snr-5": functools.partial(_loss_weighting_min_snr_gamma, gamma=5.0),
}


# -----------------------------------------------------------------------------
# Loss functions.
# -----------------------------------------------------------------------------


class BaseLossFn(abc.ABC):
    """Abstract base class for loss functions."""

    @abc.abstractmethod
    def _loss(
        self, denoiser: Denoiser, input: Tensor, noise: Tensor, sigma: Tensor, **model_kwargs
    ) -> Tensor:
        """Computes the loss for a batch of inputs. Must be implemented by subclasses."""

    def __call__(
        self,
        denoiser: Denoiser,
        loss_weight_fn: LossWeightFn,
        input: Tensor,
        noise: Tensor,
        sigma: Tensor,
        **model_kwargs,
    ) -> Tensor:
        """Computes weighted loss for a batch of inputs."""
        loss = self._loss(denoiser, input, noise, sigma, **model_kwargs)  # shape: [batch_size]
        weight = loss_weight_fn(sigma, denoiser.sigma_data)  # shape: [batch_size]
        return (loss * weight).mean()


class SimpleLossFn(BaseLossFn):
    """Computes simple MSE loss between the predicted and true noise from Ho et al. (2020)."""

    def _loss(
        self, denoiser: Denoiser, input: Tensor, noise: Tensor, sigma: Tensor, **model_kwargs
    ) -> Tensor:
        noised_input = input + noise * expand_dims(sigma, input.ndim)
        denoised_input = denoiser(noised_input, sigma, **model_kwargs)
        eps = (input - denoised_input) / expand_dims(sigma, input.ndim)
        return (eps - noise).pow(2).flatten(1).mean(1)


class KarrasLossFn(BaseLossFn):
    """Computes preconditioned MSE loss between denoised and target inputs from Karras et al. (2022).

    The loss has the following form:
        loss = precond(sigma) * (D(y + n, sigma) - y) ** 2,
        where:
            - precond(sigma) is an element-wise function that assigns weights to different noise levels,
            - D is the Karras preconditioned denoiser (KarrasDenoiser),
            - y is the noiseless input and y + n is the noised input.
    """

    def _loss(
        self,
        denoiser: KarrasDenoiser,
        input: Tensor,
        noise: Tensor,
        sigma: Tensor,
        **model_kwargs,
    ) -> Tensor:
        noised_input = input + noise * expand_dims(sigma, input.ndim)
        denoised_input = denoiser(noised_input, sigma, **model_kwargs)
        precond = (sigma**2 + denoiser.sigma_data**2) / (sigma * denoiser.sigma_data) ** 2
        return precond * (denoised_input - input).pow(2).flatten(1).mean(1)


# -----------------------------------------------------------------------------
# Noise level samplers.
# -----------------------------------------------------------------------------
# Noise level samplers determine how noise levels are sampled during training.
# Each sampler first samples a batch of sigmas, and then generates a batch of
# noise vectors from the normal distribution with the corresponding sigmas.
# -----------------------------------------------------------------------------


class BaseSigmaSampler(abc.ABC):
    """Abstract base class for sampling sigma values during training."""

    @abc.abstractmethod
    def __call__(self, batch_size: int, device="cpu", dtype=torch.float32) -> Tensor:
        """Generates a batch of sigmas. Must be implemented by subclasses."""


class LogUniformSigmaSampler(BaseSigmaSampler):
    """Samples noise levels from a log-uniform distribution."""

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, batch_size: int, device="cpu", dtype=torch.float32) -> Tensor:
        """Generates a batch of sigmas."""
        rand_tensor = torch.rand(batch_size, device=device, dtype=dtype)
        log_min_value, log_max_value = math.log(self.min_value), math.log(self.max_value)
        return torch.exp(log_min_value + rand_tensor * (log_max_value - log_min_value))


class LogNormalNoiseSampler(BaseSigmaSampler):
    """Samples noise levels from a log-normal distribution."""

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __call__(self, batch_size: int, device="cpu", dtype=torch.float32) -> Tensor:
        """Generates a batch of noise samples."""
        rand_tensor = torch.randn(batch_size, device=device, dtype=dtype)
        return torch.exp(self.loc + rand_tensor * self.scale)


# -----------------------------------------------------------------------------
# Exponential moving average (EMA) update for model parameters.
# -----------------------------------------------------------------------------
# EMA code is adapted from: https://github.com/crowsonkb/k-diffusion.
# The original code is made available by Katherine Crowson under the MIT license.
# -----------------------------------------------------------------------------


@torch.inference_mode()
def ema_update(model, model_ema, decay):
    """Incorporates updated model parameters into an EMA model.

    Should be called after each optimizer step.
    """
    model_params = dict(model.named_parameters())
    averaged_params = dict(model_ema.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        param = param.to(averaged_params[name].device)
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(model_ema.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class EMAWarmupSchedule:
    """Implements an EMA warmup using an inverse decay schedule.

    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(
        self,
        inv_gamma=1.0,
        power=1.0,
        min_value=0.0,
        max_value=1.0,
        start_at=0,
        last_epoch=0,
    ):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_step = last_epoch

    def state_dict(self):
        """Returns the state of the class as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the class's state.
        Args:
            state_dict (dict): scaler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        """Gets the current EMA decay rate."""
        step = max(0, self.last_step - self.start_at)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return 0.0 if step < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_step += 1

"""Common utility functions for training diffusion models."""
import abc
import copy
import functools
import math
from typing import Any, Callable

import pytorch_lightning as pl
import torch

from diffusion.denoisers import Denoiser, KarrasDenoiser, KarrasOptimalDenoiser
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


# -----------------------------------------------------------------------------
# Lightning module for training diffusion models.
# -----------------------------------------------------------------------------


class DiffusionDatasetWrapper(torch.utils.data.Dataset):
    """A wrapper for a PyTorch dataset that generates noise tensors.

    Args:
        dataset: A PyTorch dataset.
        drop_last: Whether to drop the last batch if it is smaller than batch_size.
        fixed_noise: If True, the noise is generated during the first epoch and then reused.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        fixed_noise: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.fixed_noise = fixed_noise
        self.noise_cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, *_ = self.dataset[index]
        # Generate noise tensor if it is not cached.
        if index not in self.noise_cache:
            noise = torch.randn_like(x)
            if self.fixed_noise:
                self.noise_cache[index] = noise
        else:
            noise = self.noise_cache[index]
        return x, *_, noise


class DiffusionModel(pl.LightningModule):
    """A Pytorch Lightning module for training denoising diffusion models.

    Args:
        model: A denoising model.
        loss_fn: A denoising loss function.
        loss_weight_fn: A loss weighting function.
        sigma_sampler: A noise level sampler.
        ema_schedule: An EMA warmup schedule.
        optimizer_cls: A PyTorch optimizer class.
        optimizer_kwargs: Keyword arguments for the optimizer.
        lr_scheduler_cls: A PyTorch learning rate scheduler class.
        lr_scheduler_kwargs: Keyword arguments for the learning rate scheduler.
        lr_scheduler_monitor: The metric to monitor for the learning rate scheduler.
        validation_sigmas: A list of noise levels for which validation losses are computed.
        validation_optimal_denoiser: If provided, the optimal denoiser is used to compute optimal
            validation losses, which are then subtracted from the validation losses of the model.
    """

    def __init__(
        self,
        model: Denoiser,
        *,
        loss_fn: BaseLossFn,
        loss_weight_fn: LossWeightFn,
        sigma_sampler: BaseSigmaSampler,
        ema_schedule: EMAWarmupSchedule,
        optimizer_cls: type[torch.optim.Optimizer],
        optimizer_kwargs: dict[str, Any],
        lr_scheduler_cls: type[torch.optim.lr_scheduler._LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_monitor: str | None = None,
    ):
        super().__init__()

        # Save model and create EMA version.
        self.model = model
        self.model_ema = copy.deepcopy(model)

        # Save loss and weight functions, noise sampler, and EMA schedule.
        self.loss_fn = loss_fn
        self.train_loss_weight_fn = loss_weight_fn
        self.val_loss_weight_fn = WEIGHTING_SCHEMES["uniform"]
        self.sigma_sampler = sigma_sampler
        self.ema_schedule = ema_schedule

        # Save optimization parameters.
        self._optimizer_builder = functools.partial(optimizer_cls, **optimizer_kwargs)
        if lr_scheduler_cls is None:
            self._lr_scheduler_builder = lambda _: None
        else:
            self._lr_scheduler_builder = functools.partial(
                lr_scheduler_cls, **(lr_scheduler_kwargs or {})
            )
        self._lr_scheduler_monitor = lr_scheduler_monitor or "val/loss"

    def setup_validation(
        self,
        validation_sigmas: list[float] | None = None,
        validation_nll_fn: Callable[[Tensor], Tensor] | None = None,
        validation_optimal_denoiser: KarrasOptimalDenoiser | None = None,
    ):
        """Saves paramters necessary for computing validation metrics."""
        self.validation_sigmas = validation_sigmas
        self.validation_nll_fn = validation_nll_fn
        self.validation_optimal_denoiser = validation_optimal_denoiser
        self.validation_optimal_loss_cache = {}

    def forward(self, input, sigma, **model_kwargs):
        return self.model_ema(input, sigma, **model_kwargs)

    # --- PyTroch Lightning methods: start ------------------------------------

    @property
    def current_lr(self) -> float:
        optimizer = self.optimizers()
        return optimizer.optimizer.param_groups[0]["lr"]

    def configure_optimizers(self):
        optimizer = self._optimizer_builder(self.parameters())
        lr_scheduler = self._lr_scheduler_builder(optimizer)
        if lr_scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": self._lr_scheduler_monitor,
            }

    def optimizer_step(self, *args, **kwargs):
        """Updates model parameters and EMA model parameters."""
        super().optimizer_step(*args, **kwargs)
        # Log learning rate.
        self.log("lr", self.current_lr, prog_bar=True)
        # Update EMA model.
        ema_decay = self.ema_schedule.get_value()
        ema_update(self.model, self.model_ema, ema_decay)
        self.ema_schedule.step()

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """Samples noise level and computes loss."""
        del batch_idx  # Unused.
        x_batch, *_, n_batch = batch
        batch_size = x_batch.shape[0]
        sigma = self.sigma_sampler(batch_size, device=x_batch.device)
        loss = self.loss_fn(self.model, self.train_loss_weight_fn, x_batch, n_batch, sigma)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> None:
        """Computes and logs validation metrics."""
        x_batch, *_, n_batch = batch

        # Compute validation losses for each noise level.
        if self.validation_sigmas is not None:
            total_loss = 0.0
            total_loss_ema = 0.0
            batch_size = x_batch.shape[0]
            sigma = torch.empty((batch_size,), device=x_batch.device)
            for sigma_value in self.validation_sigmas:
                sigma.fill_(sigma_value)
                loss = self.loss_fn(self.model, self.val_loss_weight_fn, x_batch, n_batch, sigma)
                loss_ema = self.loss_fn(
                    self.model_ema, self.val_loss_weight_fn, x_batch, n_batch, sigma
                )
                if self.validation_optimal_denoiser is not None:
                    optimal_loss_idx = (batch_idx, sigma_value)
                    if optimal_loss_idx not in self.validation_optimal_loss_cache:
                        self.validation_optimal_loss_cache[optimal_loss_idx] = self.loss_fn(
                            self.validation_optimal_denoiser,
                            self.val_loss_weight_fn,
                            x_batch,
                            n_batch,
                            sigma,
                        )
                    loss -= self.validation_optimal_loss_cache[optimal_loss_idx]
                    loss_ema -= self.validation_optimal_loss_cache[optimal_loss_idx]
                self.log(f"val/loss/sigma_{sigma_value:.1e}", loss, sync_dist=True)
                self.log(f"val/loss_ema/sigma_{sigma_value:.1e}", loss_ema, sync_dist=True)
                total_loss += loss
                total_loss_ema += loss_ema
            total_loss /= len(self.validation_sigmas)
            total_loss_ema /= len(self.validation_sigmas)
            self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
            self.log("val/loss_ema", total_loss_ema, prog_bar=True, sync_dist=True)

        # Compute validation log-likelihoods.
        if self.validation_nll_fn is not None:
            nll = self.validation_nll_fn(x_batch).mean()
            self.log("val/nll", nll, prog_bar=True, sync_dist=True)

    # --- Lightning module methods: end ---------------------------------------

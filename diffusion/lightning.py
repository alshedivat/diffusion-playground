"""PyTorch Lightning API for training and inference."""
import copy
import functools
import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable

import pytorch_lightning as pl
import torch

from diffusion.denoisers import Denoiser, KarrasOptimalDenoiser
from diffusion.inference import BaseDiffEq, BaseDiffEqSolver, BaseNoiseSchedule
from diffusion.training import (
    WEIGHTING_SCHEMES,
    BaseLossFn,
    BaseSigmaSampler,
    EMAWarmupSchedule,
    LossWeightFn,
    ema_update,
)

# Setup logging.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

# Type aliases.
Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Datasets and wrappers.
# -----------------------------------------------------------------------------
# Classes deined below are used to define a dataset of noise samples and wrap
# other PyTorch datasets to include noise samples. The noise samples are used
# for training and inference. The noise tensors are generated on-the-fly and
# can be reused for each sample across epochs.
# -----------------------------------------------------------------------------


class StandardNormalNoiseDataset(torch.utils.data.Dataset):
    """A dataset that generates standard normal noise tensors.

    Args:
        shape: The shape of the noise tensors that will be generated.
        size: The total number of noise tensors in the dataset.
        fixed_noise: If True, the noise is generated during the first epoch and then reused.
    """

    def __init__(self, shape: tuple[int, ...], size: int, fixed_noise: bool = True):
        super().__init__()
        self.shape = shape
        self.size = size
        self.fixed_noise = fixed_noise
        self.noise_cache = {}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> dict[str, Tensor]:
        # Generate noise tensor if it is not cached.
        if index not in self.noise_cache:
            noise = torch.randn(self.shape)
            if self.fixed_noise:
                self.noise_cache[index] = noise
        else:
            noise = self.noise_cache[index]
        return {"noise": noise}


class DiffusionDatasetWrapper(torch.utils.data.Dataset):
    """A wrapper for a PyTorch dataset that generates noise tensors.

    Args:
        dataset: A PyTorch dataset.
        fixed_noise: If True, the noise is generated during the first epoch and then reused.
            Using the same noise for each sample in the dataset is useful for validation.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        fixed_noise: bool = False,
    ):
        super().__init__()
        self.dataset = dataset

        # Create noise dataset.
        self.noise_dataset = StandardNormalNoiseDataset(
            shape=dataset[0]["input"].shape, size=len(dataset), fixed_noise=fixed_noise
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> dict[str, Tensor]:
        return dict(**self.dataset[index], **self.noise_dataset[index])


# -----------------------------------------------------------------------------
# Callbacks.
# -----------------------------------------------------------------------------
# Various useful callbacks for training diffusion models.
# -----------------------------------------------------------------------------


class BatchNoisinessMonitor(pl.Callback):
    """A callback that monitors batch noisiness and logs crical batch size.

    The critical batch size is defined as the batch size that results in optimal computational
    efficiency of training, as described in McCandlish et al. (2018). It can be approximately
    computed using the simple estimate B_simple:
        B_simple := tr(Sigma(grad L)) / ||grad L||^2,
    where tr(Sigma(grad L)) is the trace of the covariance matrix of the gradients of the loos L.

    Computing B_simple using the above formula requires per-sample gradients, computing which
    adds overhead. Instead, we use an estimate that can be computed with no additional overhead
    in the distributed training setting, as described in Appendix A.1 of the paper:
        B_simple_hat := S / ||G||^2, where
        S := (||G_small||^2 - ||G_big||^2) / (1/ B_small - 1 / B_big),
        ||G||^2 := (B_big ||G_big||^2 - B_small ||G_small||^2) / (B_big - B_small),
    where:
        - B_small is the batch size used per device,
        - B_big is effective batch size across all devices,
        - ||G_small||^2 is the squared norm of the gradients computed using B_small,
        - ||G_big||^2 is the squared norm of the gradients computed using B_big.

    This callback essentially aggregates gradients before and after averaging them across all GPUs,
    then smooths S and ||G||^2 estimates using exponential moving averages, and logs the estimated
    critical batch size.

    NOTE: This callback requires the model to be trained using DDP with multiple devices.
          If the model is trained on a single GPU, the callback will log a warning and log NaNs.

    Reference: https://arxiv.org/abs/1812.06162.
    """

    def __init__(self, ema_decay=0.9, eps=1e-5):
        super().__init__()
        self.ema_decay = ema_decay
        self.eps = eps

        # Placeholders.
        self.batch_size_big = None
        self.batch_size_small = None
        self.no_gradient_sync_context = None

        # EMA buffers.
        self.s_ema = None
        self.g2_ema = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.num_devices * trainer.num_nodes < 2:
            logger.warning(
                "Cannot estimate batch noisiness with a single device. "
                "To estimate batch noisiness, runn DDP training with multiple devices."
            )
        else:
            if trainer.is_global_zero:
                logger.info("Disabling automatic gradient sync for computing B_simple estimates.")
            # Enter no gradient sync context.
            self.no_gradient_sync_context = trainer.strategy.block_backward_sync()
            self.no_gradient_sync_context.__enter__()

        # Initialize EMA buffers.
        self.s_ema = torch.ones(1, device=pl_module.device)
        self.g2_ema = torch.ones(1, device=pl_module.device)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer, pl_module  # Unused.
        if self.no_gradient_sync_context is not None:
            # Exit no gradient sync context.
            self.no_gradient_sync_context.__exit__(None, None, None)
            self.no_gradient_sync_context = None

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        del pl_module, batch_idx  # Unused.
        self.batch_size_small = batch["input"].shape[0]
        self.batch_size_big = self.batch_size_small * trainer.num_devices * trainer.num_nodes

    @torch.no_grad()
    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.num_devices * trainer.num_nodes < 2:
            if trainer.is_global_zero:
                pl_module.log("B_simple", float("nan"), on_step=True, on_epoch=True)
            return

        # Compute ||G_small||^2.
        g2_small = 0.0
        for param in pl_module.parameters():
            if param.requires_grad:
                # Compute g_small on each device separately.
                g2_small += torch.sum(param.grad**2)
        g2_small = trainer.strategy.reduce(g2_small, reduce_op="mean")

        # Compute ||G_big||^2.
        g2_big = 0.0
        for param in pl_module.parameters():
            if param.requires_grad:
                # Manually sync gradients across devices.
                param.grad = trainer.strategy.reduce(param.grad, reduce_op="mean")
                # Compute g_big on each device separately.
                g2_big += torch.sum(param.grad**2)
        g2_big = trainer.strategy.reduce(g2_big, reduce_op="mean")

        # Compute S, ||G||^2, and B_simple_hat.
        if trainer.is_global_zero:
            B_big = self.batch_size_big
            B_small = self.batch_size_small
            s = (g2_small - g2_big) / (1 / B_small - 1 / B_big)
            g2 = (B_big * g2_big - B_small * g2_small) / (B_big - B_small)
            # Update EMA.
            self.s_ema = self.ema_decay * self.s_ema + (1 - self.ema_decay) * s
            self.g2_ema = self.ema_decay * self.g2_ema + (1 - self.ema_decay) * g2
            # Compute B_simple_hat.
            B_simple_hat = self.s_ema / (self.g2_ema + self.eps)
            logger.info(f"Estimated critical batch size: {B_simple_hat.item():.2f}")
            self.log("B_simple", B_simple_hat, on_step=True)


# -----------------------------------------------------------------------------
# Configs.
# -----------------------------------------------------------------------------
# Dataclasses that contain configuration for training and inference.
# -----------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for training a denoising diffusion model.

    Args:
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
        validation_nll_fn: A function that computes negative log-likelihoods of the validation data.
        validation_optimal_denoiser: If provided, the optimal denoiser is used to compute optimal
            validation losses, which are then subtracted from the validation losses of the model.
    """

    # Training.
    loss_fn: BaseLossFn
    loss_weight_fn: LossWeightFn
    sigma_sampler: BaseSigmaSampler

    # Optimization.
    ema_schedule: EMAWarmupSchedule
    optimizer_cls: type[torch.optim.Optimizer]
    optimizer_kwargs: dict[str, Any]
    lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None
    lr_scheduler_kwargs: dict[str, Any] | None = None
    lr_scheduler_interval: str | None = None
    lr_scheduler_monitor: str | None = None

    # Validation.
    validation_sigmas: list[float] | None = None
    validation_nll_fn: Callable[[Tensor], Tensor] | None = None
    validation_optimal_denoiser: KarrasOptimalDenoiser | None = None


@dataclass
class InferenceConfig:
    """Configuration for sampling from a denoising model.

    Args:
        ode_builder: A function that given a Denoiser returns a probability flow ODE.
        solver: An ODE solver to use for integration.
        noise_schedule: A noise schedule that defines integration trajectory.
        n_steps: Number of steps in the integration trajectory.
        return_trajectory: If True, the entire integration trajectory is returned,
            otherwise only the final sample is returned.
    """

    ode_builder: Callable[[Denoiser], BaseDiffEq]
    solver: BaseDiffEqSolver
    noise_schedule: BaseNoiseSchedule
    n_steps: int
    return_trajectory: bool = False


# -----------------------------------------------------------------------------
# Lightning module.
# -----------------------------------------------------------------------------
# A PyTorch Lightning module for denoising diffusion training and inference.
# -----------------------------------------------------------------------------


class LightningDiffusion(pl.LightningModule):
    """A Pytorch Lightning module for denoising diffusion training and inference.

    Args:
        model: A denoising model.
        training_config: An optional dataclass that contains configuration for training.
        inference_config: An optional dataclass that contains configuration for inference.
    """

    def __init__(
        self,
        model: Denoiser,
        training_config: TrainingConfig | None = None,
        inference_config: InferenceConfig | None = None,
    ):
        super().__init__()

        # Save model and create EMA version.
        self.model = model
        self.model_ema = copy.deepcopy(model).eval().requires_grad_(False)

        # Setup training and inference.
        self.setup_training(training_config)
        self.setup_inference(inference_config)

    def setup_training(self, training_config: TrainingConfig):
        self.training_config = training_config
        if self.training_config is None:
            return

        # Save loss and weight functions, noise sampler, and EMA schedule.
        self.loss_fn = training_config.loss_fn
        self.train_loss_weight_fn = training_config.loss_weight_fn
        self.val_loss_weight_fn = WEIGHTING_SCHEMES["uniform"]
        self.sigma_sampler = training_config.sigma_sampler
        self.ema_schedule = training_config.ema_schedule

        # Save optimization parameters.
        self._optimizer_builder = functools.partial(
            training_config.optimizer_cls, **training_config.optimizer_kwargs
        )
        if training_config.lr_scheduler_cls is None:
            self._lr_scheduler_builder = lambda _: None
        else:
            self._lr_scheduler_builder = functools.partial(
                training_config.lr_scheduler_cls, **(training_config.lr_scheduler_kwargs or {})
            )
        self._lr_scheduler_interval = training_config.lr_scheduler_interval or "epoch"
        self._lr_scheduler_monitor = training_config.lr_scheduler_monitor or "val/loss"

        # Save validation parameters.
        self.validation_sigmas = training_config.validation_sigmas
        self.validation_nll_fn = training_config.validation_nll_fn
        self.validation_optimal_denoiser = training_config.validation_optimal_denoiser

        # Initialize cache for optimal validation losses.
        self._validation_optimal_loss_cache = {}

    def setup_inference(self, inference_config: InferenceConfig):
        self.inference_config = inference_config
        if self.inference_config is None:
            return

        self.inference_ode = inference_config.ode_builder(self.model_ema)
        self.inference_solver = inference_config.solver
        self.inference_noise_schedule = inference_config.noise_schedule
        self.inference_n_steps = inference_config.n_steps
        self.inference_return_trajectory = inference_config.return_trajectory

    def forward(self, input, sigma, **model_kwargs):
        # TODO: do we need this method?
        return self.model_ema(input, sigma, **model_kwargs)

    @property
    def current_lr(self) -> float:
        optimizer = self.optimizers()
        return optimizer.optimizer.param_groups[0]["lr"]

    # --- PyTroch Lightning methods: start ------------------------------------

    def configure_optimizers(self):
        optimizer = self._optimizer_builder(self.parameters())
        lr_scheduler = self._lr_scheduler_builder(optimizer)
        if lr_scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": self._lr_scheduler_interval,
                    "monitor": self._lr_scheduler_monitor,
                },
            }

    def optimizer_step(self, *args, **kwargs):
        """Updates model parameters and EMA model parameters."""
        super().optimizer_step(*args, **kwargs)
        # Remove NaNs from gradients.
        for param in self.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        # Update EMA model.
        ema_decay = self.ema_schedule.get_value()
        ema_update(self.model, self.model_ema, ema_decay)
        self.ema_schedule.step()
        # Log learning rate.
        self.log("lr", self.current_lr, on_step=True, on_epoch=False, prog_bar=True)
        # Log EMA decay rate.
        self.log("ema_decay", ema_decay, on_step=True, on_epoch=False, prog_bar=True)

    def training_step(self, batch_dict: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Samples noise level and computes loss."""
        del batch_idx  # Unused.
        batch_size = batch_dict["input"].shape[0]
        sigma = self.sigma_sampler(batch_size, device=batch_dict["input"].device)
        loss = self.loss_fn(self.model, self.train_loss_weight_fn, batch_dict, sigma)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch_dict: dict[str, Tensor], batch_idx: int) -> None:
        """Computes and logs validation metrics."""
        # Compute validation losses for each noise level.
        if self.validation_sigmas is not None:
            total_loss = 0.0
            total_loss_ema = 0.0
            batch_size = batch_dict["input"].shape[0]
            sigma = torch.empty((batch_size,), device=batch_dict["input"].device)
            for sigma_value in self.validation_sigmas:
                sigma.fill_(sigma_value)
                loss = self.loss_fn(self.model, self.val_loss_weight_fn, batch_dict, sigma)
                loss_ema = self.loss_fn(self.model_ema, self.val_loss_weight_fn, batch_dict, sigma)
                if self.validation_optimal_denoiser is not None:
                    optimal_loss_idx = (batch_idx, sigma_value)
                    if optimal_loss_idx not in self._validation_optimal_loss_cache:
                        self._validation_optimal_loss_cache[optimal_loss_idx] = self.loss_fn(
                            self.validation_optimal_denoiser,
                            self.val_loss_weight_fn,
                            batch_dict,
                            sigma,
                        )
                    loss -= self._validation_optimal_loss_cache[optimal_loss_idx]
                    loss_ema -= self._validation_optimal_loss_cache[optimal_loss_idx]
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
            nll = self.validation_nll_fn(batch_dict).mean()
            self.log("val/nll", nll, prog_bar=True, sync_dist=True)

    def predict_step(
        self, batch_dict: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Generates a batch of samples given a batch of noise."""
        del batch_idx, dataloader_idx  # Unused.

        if "noise" not in batch_dict or len(batch_dict) != 1:
            raise ValueError(
                f"`batch_dict` must contain only a single noise tensor, but got {batch_dict=}."
            )

        # Generate grid.
        x, sigma0 = self.inference_noise_schedule.get_x_schedule(
            self.inference_n_steps, domain=self.inference_ode.domain, device=self.device
        )

        # Generate initial noisy sample.
        y0 = sigma0 * batch_dict["noise"]

        # Run solver.
        trajectory, *_ = self.inference_solver.solve(x, y0_tuple=(y0,), ode=self.inference_ode)

        if self.inference_return_trajectory:
            # Swap the time and batch dimensions and return the entire trajectory.
            return trajectory.swapaxes(0, 1)
        else:
            # Return final sample in the trajectory.
            return trajectory[-1]

    def on_predict_epoch_end(self):
        """Gathers predictions from all processes and terminates worker processes, if needed."""
        if self.trainer.num_devices > 1:
            predictions = self.all_gather(self.trainer.predict_loop.predictions)
            if self.trainer.is_global_zero:
                # Flatten predictions.
                predictions = [torch.flatten(p, end_dim=1) for p in predictions]
                self.trainer.predict_loop._predictions = [predictions]
            else:
                # Make worker processes exit.
                sys.exit()

    # --- Lightning module methods: end ---------------------------------------

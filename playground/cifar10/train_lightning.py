"""Script for training a diffusion model on CIFAR10 data."""
import click
import pytorch_lightning as pl
import torch
from aim.pytorch_lightning import AimLogger
from augment import AugmentPipe
from data import create_dataloaders, load_data
from model import SongUNet
from torch.optim.lr_scheduler import LRScheduler

from diffusion.denoisers import KarrasDenoiser
from diffusion.lightning import (
    BatchNoisinessMonitor,
    DiffusionDatasetWrapper,
    LightningDiffusion,
    TrainingConfig,
)
from diffusion.training import (
    WEIGHTING_SCHEMES,
    EMAWarmupSchedule,
    KarrasLossFn,
    LogNormalNoiseSampler,
)


class WarmupLRScheduler(LRScheduler):
    """Ramps up the learning rate linearly during the first `warmup_steps` steps."""

    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [
                base_lr * min(self._step_count / max(self.warmup_steps, 1e-8), 1)
                for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs


def create_augment_collate_fn(augment_pipe):
    """A custom collate function that applies augmentation after batch collation."""

    def collate_fn(instance_list):
        batch_dict = torch.utils.data._utils.collate.default_collate(instance_list)
        augment_input, augment_labels = augment_pipe(batch_dict.pop("input"))
        batch_dict["input"] = augment_input
        batch_dict["augment_labels"] = augment_labels
        return batch_dict

    return collate_fn


@click.command()
@click.option("--augment_prob", default=0.12, help="Augmentation probability.")
@click.option("--batch_size", default=128, help="Training batch size.")
@click.option("--ckpt_dir", default="checkpoints", help="Checkpointing directory.")
@click.option("--dropout_prob", default=0.13, help="Dropout probability.")
@click.option("--lr", default=1e-3, help="Learning rate.")
@click.option("--loss_weighting", default="uniform", help="Name of the loss weighting scheme.")
@click.option("--mixed_precision", is_flag=True, default=False, help="Enable mixed precision.")
@click.option("--n_workers", default=16, help="Number of workers for data loading.")
@click.option("--n_steps", default=400_000, help="Number of training steps.")
@click.option("--n_gpus", default=4, help="Number of GPUs to train on.")
@click.option("--sigma_data", default=0.5, help="Average variance of the data.")
@click.option("--seed", default=42, help="Random seed.")
def main(
    augment_prob,
    batch_size,
    ckpt_dir,
    dropout_prob,
    lr,
    loss_weighting,
    mixed_precision,
    n_workers,
    n_steps,
    n_gpus,
    sigma_data,
    seed,
):
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = True
    if mixed_precision:
        torch.set_float32_matmul_precision("high")

    # Load data.
    train_dataset, test_dataset = load_data()
    train_dataset = DiffusionDatasetWrapper(train_dataset, fixed_noise=False)
    test_dataset = DiffusionDatasetWrapper(test_dataset, fixed_noise=True)

    # Augment training data, if necessary.
    train_collate_fn = None
    if augment_prob > 0:
        augment_pipe = AugmentPipe(
            p=augment_prob,
            xflip=1e8,
            yflip=1,
            scale=1,
            rotate_frac=1,
            aniso=1,
            translate_frac=1,
        )
        train_collate_fn = create_augment_collate_fn(augment_pipe)

    # Create dataloaders.
    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        prefetch_factor=2,
        pin_memory=True,
        train_collate_fn=train_collate_fn,
    )

    # Instantiate and compile diffusion model.
    unet = SongUNet(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=0,
        augment_dim=9,
        model_channels=128,
        channel_mult=[2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[16],
        dropout=dropout_prob,
        label_dropout=0,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        decoder_type="standard",
        resample_filter=[1, 3, 3, 1],
    )
    model = KarrasDenoiser(model=unet, sigma_data=sigma_data)

    # Create training config.
    training_config = TrainingConfig(
        loss_fn=KarrasLossFn(),
        loss_weight_fn=WEIGHTING_SCHEMES[loss_weighting],
        sigma_sampler=LogNormalNoiseSampler(loc=-1.2, scale=1.2),
        ema_schedule=EMAWarmupSchedule(inv_gamma=1.0, power=0.6, max_value=0.9999),
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": lr, "betas": [0.9, 0.999], "eps": 1e-8},
        lr_scheduler_cls=WarmupLRScheduler,
        lr_scheduler_kwargs={"warmup_steps": 20_000},
        lr_scheduler_interval="step",
        validation_sigmas=[1e-3, 1e-2, 1e-1, 1e0],
        # validation_optimal_denoiser=KarrasOptimalDenoiser(val_loader, sigma_data),
    )

    # Train model.
    diffusion = LightningDiffusion(model=model, training_config=training_config)
    aim_logger = AimLogger(
        experiment="CIFAR10-diffusion",
        train_metric_prefix="train/",
        test_metric_prefix="test/",
        val_metric_prefix="val/",
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=n_gpus,
        strategy="ddp",
        precision="16-mixed" if mixed_precision else "32",
        max_steps=n_steps,
        gradient_clip_val=None,
        logger=aim_logger,
        log_every_n_steps=20,
        callbacks=[
            model_checkpoint_callback,
            BatchNoisinessMonitor(),
        ],
    )
    trainer.fit(diffusion, train_loader, test_loader)


if __name__ == "__main__":
    main()

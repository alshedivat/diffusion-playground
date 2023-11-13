"""Script for training a diffusion model on CIFAR10 data."""
import click
import pytorch_lightning as pl
import torch
from aim.pytorch_lightning import AimLogger
from data import create_dataloaders, load_data
from model import DhariwalUNet
from pytorch_lightning.callbacks import ModelCheckpoint

from diffusion.denoisers import KarrasDenoiser, KarrasOptimalDenoiser
from diffusion.training import (
    WEIGHTING_SCHEMES,
    DiffusionDatasetWrapper,
    DiffusionModel,
    EMAWarmupSchedule,
    KarrasLossFn,
    LogUniformSigmaSampler,
)


@click.command()
@click.option("--batch_size", default=256, help="Training batch size.")
@click.option("--lr", default=2e-4, help="Learning rate.")
@click.option("--loss_weighting", default="uniform", help="Name of the loss weighting scheme.")
@click.option("--matmul-precision", default="medium", help="Precision of float32 matmul ops.")
@click.option("--n_workers", default=8, help="Number of workers for data loading.")
@click.option("--n_epochs", default=250, help="Number of training epochs.")
@click.option("--n_gpus", default=2, help="Number of GPUs to train on.")
@click.option("--sigma_data", default=0.6, help="Average variance of the data.")
@click.option("--seed", default=42, help="Random seed.")
def main(
    batch_size, lr, loss_weighting, matmul_precision, n_workers, n_epochs, n_gpus, sigma_data, seed
):
    torch.set_float32_matmul_precision(matmul_precision)
    pl.seed_everything(seed)

    # Load data.
    train_dataset, val_dataset, test_dataset = load_data()
    train_dataset = DiffusionDatasetWrapper(train_dataset, fixed_noise=False)
    val_dataset = DiffusionDatasetWrapper(val_dataset, fixed_noise=True)
    test_dataset = DiffusionDatasetWrapper(test_dataset, fixed_noise=True)

    # Create dataloaders.
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, num_workers=n_workers
    )

    # Instantiate and compile diffusion model.
    # TODO: set all parameters specified in Karras et al. (2022).
    sigma_data = 1.0
    optimizer_cls = torch.optim.Adam
    optimizer_kwargs = {"lr": lr}
    lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    lr_scheduler_kwargs = {"mode": "min", "factor": 0.5, "patience": 5, "min_lr": 1e-6}

    unet = DhariwalUNet(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        model_channels=192,
        channel_mult=[1, 2, 3, 4],
        channel_mult_emb=4,
        num_blocks=3,
        attn_resolutions=[],
        dropout=0.10,
        label_dropout=0,
    )
    model = KarrasDenoiser(model=unet, sigma_data=sigma_data)
    loss_fn = KarrasLossFn()
    loss_weight_fn = WEIGHTING_SCHEMES[loss_weighting]
    sigma_sampler = LogUniformSigmaSampler(min_value=1e-3, max_value=1e1)
    ema_schedule = EMAWarmupSchedule(inv_gamma=1.0, power=0.9)
    diffusion = DiffusionModel(
        model=model,
        loss_fn=loss_fn,
        loss_weight_fn=loss_weight_fn,
        sigma_sampler=sigma_sampler,
        ema_schedule=ema_schedule,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_cls=lr_scheduler_cls,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )
    diffusion.setup_validation(
        validation_sigmas=[1e-3, 1e-2, 1e-1, 1e0],
        # validation_optimal_denoiser=KarrasOptimalDenoiser(val_loader, sigma_data),
    )

    aim_logger = AimLogger(
        experiment="CIFAR10-diffusion",
        train_metric_prefix="train/",
        test_metric_prefix="test/",
        val_metric_prefix="val/",
    )
    callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=n_gpus,
        strategy="ddp_find_unused_parameters_true",
        precision="16-mixed",
        max_epochs=n_epochs,
        gradient_clip_val=0.1,
        log_every_n_steps=16,
        logger=aim_logger,
        callbacks=[callback],
    )
    trainer.fit(diffusion, train_loader, val_loader)


if __name__ == "__main__":
    main()

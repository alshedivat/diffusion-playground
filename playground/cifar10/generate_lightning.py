"""Script for sampling from a pretrained diffusion model on CIFAR10 data."""
import functools
import os
import pickle
import sys
from enum import Enum

import click
import pytorch_lightning as pl
import torch
from model import SongUNet
from PIL import Image

from diffusion.denoisers import KarrasDenoiser
from diffusion.inference import (
    DPMppDiffEqSolver,
    KarrasDiffEq,
    KarrasHeun2Solver,
    KarrasNoiseSchedule,
    LinearLogSnrNoiseSchedule,
    LogSnrDiffEq,
)
from diffusion.lightning import (
    InferenceConfig,
    LightningDiffusion,
    StandardNormalNoiseDataset,
)


class ODEType(str, Enum):
    KARRAS = "karras"
    LOGSNR = "logsnr"


class SolverType(str, Enum):
    KARRAS_HEUN2 = "karras_heun2"
    DPMPP_M2 = "dpmpp_m2"


class NoiseScheduleType(str, Enum):
    KARRAS = "karras"
    LINEAR_LOGSNR = "linear_logsnr"


def save_images(samples_batch, start_index: int, dir_path: str):
    image_batch = (
        torch.permute(samples_batch * 127.5 + 128, [0, 2, 3, 1])
        .clip(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    for i, image_array in enumerate(image_batch):
        img = Image.fromarray(image_array, "RGB")
        img.save(os.path.join(dir_path, f"{start_index + i}.png"))


def load_edm_model(ckeckpoint_path: str, edm_lib_path: str):
    """Loads a model trained using the original EDM codebase.

    Args:
        checkpoint_path: Path to the checkpoint file.
        edm_lib_path: Path to the EDM library.

    Returns:
        A torch.nn.Module that represents the EMA of the trained model.
    """
    sys.path.append(os.path.expandvars(edm_lib_path))
    with open(ckeckpoint_path, "rb") as fp:
        ckpt = pickle.load(fp)
    return ckpt["ema"]


def load_retrained_model(
    checkpoint_path: str, inference_config: InferenceConfig, sigma_data: float = 0.5
):
    """Loads a model retrained using the diffusion library."""
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
        dropout=0.13,
        label_dropout=0,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        decoder_type="standard",
        resample_filter=[1, 3, 3, 1],
    )
    model = KarrasDenoiser(model=unet, sigma_data=sigma_data)
    return LightningDiffusion.load_from_checkpoint(
        checkpoint_path, model=model, inference_config=inference_config
    )


@click.command()
@click.option("--ckpt_path", help="Path to the model checkpoint.")
@click.option(
    "--ckpt_type",
    type=click.Choice(["original", "retrained"]),
    default="original",
    help="Whether the checkpoint is origianl or retrained.",
)
@click.option("--output_dir", default="out", help="Output directory where images are saved.")
@click.option("--n_samples", default=50000, help="Number of samples to generate.")
@click.option("--n_steps", default=18, help="Number of denoising steps at inference time.")
@click.option("--batch_size", default=512, help="Batch size for inference.")
@click.option("--edm_lib_path", default="$HOME/edm/", help="Path to the EDM library.")
@click.option(
    "--ode_type",
    type=click.Choice([*ODEType]),
    default=ODEType.KARRAS,
    help="Type of the ODE to use for sampling.",
)
@click.option(
    "--solver_type",
    type=click.Choice([*SolverType]),
    default=SolverType.KARRAS_HEUN2,
    help="Type of the solver to use for sampling.",
)
@click.option(
    "--noise_schedule_type",
    type=click.Choice([*NoiseScheduleType]),
    default=NoiseScheduleType.KARRAS,
    help="Type of the noise schedule to use for sampling.",
)
@click.option("--mixed_precision", is_flag=True, default=False, help="Enable mixed precision.")
@click.option("--n_gpus", default=2, help="Number of GPUs to run infernece on.")
@click.option("--seed", default=42, help="Random seed.")
def main(
    ckpt_path,
    ckpt_type,
    output_dir,
    n_samples,
    n_steps,
    batch_size,
    ode_type,
    solver_type,
    noise_schedule_type,
    edm_lib_path,
    mixed_precision,
    n_gpus,
    seed,
):
    pl.seed_everything(seed, workers=True)
    if mixed_precision:
        torch.set_float32_matmul_precision("high")

    # Initialize noise dataset and loader for inference.
    noise_dataset = StandardNormalNoiseDataset(shape=(3, 32, 32), size=n_samples)
    noise_loader = torch.utils.data.DataLoader(
        noise_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_gpus,
    )

    # Select ODE.
    if ode_type == ODEType.KARRAS:
        ode_builder = functools.partial(
            KarrasDiffEq,
            t_to_sigma=lambda t: t,
            sigma_to_t=lambda sigma: sigma,
        )
    elif ode_type == ODEType.LOGSNR:
        ode_builder = LogSnrDiffEq
    else:
        raise ValueError(f"Unknown ODE type: {ode_type}")

    # Select solver.
    if solver_type == SolverType.KARRAS_HEUN2:
        solver = KarrasHeun2Solver()
    elif solver_type == SolverType.DPMPP_M2:
        solver = DPMppDiffEqSolver(order=2, multistep=True)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    # Select noise schedule.
    if noise_schedule_type == NoiseScheduleType.KARRAS:
        noise_schedule = KarrasNoiseSchedule(
            sigma_data=0.5, sigma_min=0.002, sigma_max=80.0, rho=7.0
        )
    elif noise_schedule_type == NoiseScheduleType.LINEAR_LOGSNR:
        noise_schedule = LinearLogSnrNoiseSchedule(
            sigma_data=0.5, logsnr_min=-10.0, logsnr_max=10.0
        )
    else:
        raise ValueError(f"Unknown noise schedule type: {noise_schedule_type}")

    # Setup inference.
    inference_config = InferenceConfig(
        ode_builder=ode_builder,
        solver=solver,
        noise_schedule=noise_schedule,
        n_steps=n_steps,
        return_trajectory=False,
    )

    # Load pretrained model.
    if ckpt_type == "original":
        denoiser = load_edm_model(ckpt_path, edm_lib_path=edm_lib_path)
        if mixed_precision:
            denoiser.use_fp16 = True
        inference_model = LightningDiffusion(model=denoiser, inference_config=inference_config)
    elif ckpt_type == "retrained":
        inference_model = load_retrained_model(ckpt_path, inference_config=inference_config)

    inference_runner = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        precision="16-mixed" if mixed_precision else "32",
        # NOTE: using inference_mode is not compatible with some ODE implementations that
        # internally use autograd to compute derivatives (e.g., see KarrasDiffEq).
        inference_mode=False,
    )

    # Run inference.
    samples_batches = inference_runner.predict(inference_model, noise_loader)

    # Save samples.
    os.makedirs(output_dir, exist_ok=True)
    start_index = 0
    for samples_batch in samples_batches:
        save_images(samples_batch, start_index=start_index, dir_path=output_dir)
        start_index += samples_batch.shape[0]


if __name__ == "__main__":
    main()

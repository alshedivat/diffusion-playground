"""Script for sampling from a pretrained diffusion model on CIFAR10 data."""
import functools
import os

import click
import lightning as L
import torch
from model import load_edm_model
from PIL import Image

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


@click.command()
@click.option("--ckpt_path", help="Path to the model checkpoint.")
@click.option("--output_dir", default="out", help="Output directory where images are saved.")
@click.option("--n_samples", default=50000, help="Number of samples to generate.")
@click.option("--n_steps", default=18, help="Number of denoising steps at inference time.")
@click.option("--batch_size", default=512, help="Batch size for inference.")
@click.option("--edm_lib_path", default="$HOME/edm/", help="Path to the EDM library.")
@click.option(
    "--ode_type",
    type=click.Choice(["karras_time", "karras_logsnr"]),
    default="karras_time",
    help="Type of the ODE to use for sampling.",
)
@click.option(
    "--solver_type",
    type=click.Choice(["karras_heun2", "dpmpp_m2"]),
    default="karras_heun2",
    help="Type of the solver to use for sampling.",
)
@click.option(
    "--noise_schedule_type",
    type=click.Choice(["karras", "linear_logsnr"]),
    default="karras",
    help="Type of the noise schedule to use for sampling.",
)
@click.option("--mixed_precision", is_flag=True, default=False, help="Enable mixed precision.")
@click.option("--n_gpus", default=2, help="Number of GPUs to run infernece on.")
@click.option("--seed", default=42, help="Random seed.")
def main(
    ckpt_path,
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
    L.seed_everything(seed, workers=True)
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

    # Load pretrained model.
    # TODO: add support for loading models trained in the playground.
    denoiser = load_edm_model(ckpt_path, edm_lib_path=edm_lib_path)
    if mixed_precision:
        denoiser.use_fp16 = True

    # Select ODE.
    if ode_type == "karras_time":
        ode_builder = functools.partial(
            KarrasDiffEq,
            t_to_sigma=lambda t: t,
            sigma_to_t=lambda sigma: sigma,
        )
    elif ode_type == "karras_logsnr":
        ode_builder = LogSnrDiffEq
    else:
        raise ValueError(f"Unknown ODE type: {ode_type}")

    # Select solver.
    if solver_type == "karras_heun2":
        solver = KarrasHeun2Solver()
    elif solver_type == "dpmpp_m2":
        solver = DPMppDiffEqSolver(order=2, multistep=True)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    # Select noise schedule.
    if noise_schedule_type == "karras":
        noise_schedule = KarrasNoiseSchedule(
            sigma_data=0.5, sigma_min=0.002, sigma_max=80.0, rho=7.0
        )
    elif noise_schedule_type == "logsnr_linear":
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
    inference_model = LightningDiffusion(model=denoiser, inference_config=inference_config)
    inference_runner = L.Trainer(
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

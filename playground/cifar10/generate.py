"""Script for sampling from a pretrained diffusion model on CIFAR10 data."""
import functools
import os

import click
import lightning as L
import torch
from model import load_edm_model
from PIL import Image

from diffusion.inference import (
    KarrasDiffEq,
    KarrasHeun2Solver,
    KarrasNoiseSchedule,
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
@click.option("--n_samples", default=50000, help="Number of samples to generate.")
@click.option("--n_steps", default=18, help="Number of denoising steps at inference time.")
@click.option("--batch_size", default=512, help="Batch size for inference.")
@click.option("--edm_lib_path", default="$HOME/edm/", help="Path to the EDM library.")
@click.option("--output_dir", default="out", help="Output directory where images are saved.")
@click.option("--n_gpus", default=2, help="Number of GPUs to run infernece on.")
@click.option("--seed", default=42, help="Random seed.")
def main(ckpt_path, n_samples, n_steps, batch_size, edm_lib_path, output_dir, n_gpus, seed):
    L.seed_everything(seed, workers=True)

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

    # Setup inference.
    # TODO: make ODE and solver configurable through script args.
    ode_builder = functools.partial(
        KarrasDiffEq,
        t_to_sigma=lambda t: t,
        sigma_to_t=lambda sigma: sigma,
    )
    inference_config = InferenceConfig(
        ode_builder=ode_builder,
        solver=KarrasHeun2Solver(),
        noise_schedule=KarrasNoiseSchedule(
            sigma_data=0.5, sigma_min=0.002, sigma_max=80.0, rho=7.0
        ),
        n_steps=n_steps,
        return_trajectory=False,
    )
    inference_model = LightningDiffusion(model=denoiser, inference_config=inference_config)
    inference_runner = L.Trainer(
        accelerator="gpu",
        devices=n_gpus,
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

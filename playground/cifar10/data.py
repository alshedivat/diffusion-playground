"""Loading and preprocessing CIFAR10 data."""
from typing import Any, Callable

import numpy as np
import torch
from torchvision import datasets, transforms

Tensor = torch.Tensor


def _get_transform():
    """Returns a torchvision transform for CIFAR10 data."""

    def _scale(image):
        return np.array(image) / 127.5 - 1.0

    def _to_tensor(image):
        return torch.from_numpy(image).permute(2, 0, 1).float()

    return transforms.Compose([_scale, _to_tensor])


class _DatasetWrapper(torch.utils.data.Dataset):
    """Wrapper for torchvision datasets that converts tuple elements to dicts."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return {"input": image, "class_labels": label}


def load_data(data_dir="."):
    """Loads CIFAR10 data from a given directory (download, if necessary)."""
    transform = _get_transform()
    train_dataset = _DatasetWrapper(
        datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    )
    test_dataset = _DatasetWrapper(
        datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    )
    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset,
    test_dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    train_collate_fn: Callable | None = None,
):
    """Creates Pytorch dataloaders for MNIST data."""
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    return train_loader, test_loader


def get_image_grid(trajectory: Tensor, step: int, nrows: int = 4, ncols: int = 4, padding: int = 2):
    """Returns a grid of images for the given step of the trajectory."""
    image_batch = (
        torch.permute(trajectory[step] * 127.5 + 128, [0, 2, 3, 1])
        .clip(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    # Pad images with white pixels.
    image_batch = np.pad(
        image_batch,
        constant_values=255,
        pad_width=((0, 0), (padding, padding), (padding, padding), (0, 0)),
    )
    # Sanity check.
    batch_size, height, width, channels = image_batch.shape
    assert batch_size == nrows * ncols
    # Reshape image batch into a grid.
    image_grid = image_batch.reshape(nrows, ncols, height, width, channels)
    image_grid = image_grid.swapaxes(1, 2).reshape(height * nrows, width * ncols, channels)
    return image_grid

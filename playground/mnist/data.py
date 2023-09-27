"""Loading and preprocessing MNIST data."""
import numpy as np
import torch
from torchvision import datasets, transforms

Tensor = torch.Tensor


def _get_transform():
    """Returns a torchvision transform for MNIST data."""

    def _scale(image):
        return (np.array(image) - 128.0) / 127.5

    def _to_tensor(image):
        return torch.from_numpy(image).unsqueeze(0).float()

    return transforms.Compose([_scale, _to_tensor])


def load_data(data_dir=".", seed=42):
    """Loads MNIST data from a given directory (download, if necessary)."""
    transform = _get_transform()
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    # Split training data into training and validation sets.
    rng = torch.Generator().manual_seed(seed)
    train_val_sizes = (50000, 10000)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, train_val_sizes, generator=rng
    )
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size: int = 256, num_workers: int = 4
):
    """Creates Pytorch dataloaders for MNIST data."""
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def get_image_grid(trajectory: Tensor, step: int, nrows: int = 4, ncols: int = 4, padding: int = 2):
    """Returns a grid of images for the given step of the trajectory."""
    image_batch = (trajectory[step] * 127.5 + 128).to(torch.uint8).squeeze().cpu().numpy()
    # Pad images with white pixels.
    image_batch = np.pad(
        image_batch, pad_width=((0, 0), (padding, padding), (padding, padding)), constant_values=255
    )
    # Sanity check.
    batch_size, height, width = image_batch.shape
    assert batch_size == nrows * ncols
    # Reshape image batch into a grid.
    image_grid = image_batch.reshape(nrows, ncols, height, width)
    image_grid = image_grid.swapaxes(1, 2).reshape(height * nrows, width * ncols)
    return image_grid

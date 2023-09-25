"""Various utility functions."""
import torch

Tensor = torch.Tensor


def expand_dims(x: Tensor, target_ndim):
    """Expands the dimensions of a tensor to match a target number of dimensions.

    Args:
        x: Input tensor of shape [N].
        target_ndim: Target number of dimensions.

    Returns:
        Tensor of shape [N, 1, ..., 1] with target_ndim dimensions and the same values as x.
    """
    return x.reshape(x.shape + (1,) * (target_ndim - x.ndim))


def sigma_to_logsnr(sigma: Tensor, sigma_data: float):
    return 2 * torch.log(sigma_data / sigma)


def logsnr_to_sigma(logsnr: Tensor, sigma_data: float):
    return sigma_data * torch.exp(-logsnr / 2)

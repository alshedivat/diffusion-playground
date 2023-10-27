"""Denoising models."""
import abc

import torch

from diffusion import utils

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Denoising models.
# -----------------------------------------------------------------------------
# Classes deined below are used to wrap trainable models, precondition inputs
# and outputs, and define a consistent API used at training and inference time.
# Preconditioning helps improve the dynamics of training (Karras et al., 2022).
# -----------------------------------------------------------------------------


class Denoiser(abc.ABC, torch.nn.Module):
    """Abstract base class for denoisers."""

    @abc.abstractmethod
    def forward(self, input, sigma, **kwargs):
        """Computes the denoised output for a given input and noise level.

        Must be implemented by subclasses.
        """


class SimpleDenoiser(Denoiser):
    """Simple denoiser that does not perform any preconditioning."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input, sigma, **kwargs):
        return self.model(input, sigma, **kwargs)


class KarrasDenoiser(Denoiser):
    """EDM denoiser from Karras et al. (2022) with preconditioned inputs and outputs.

    This denoiser wraps a trainable model and scales its inputs and outputs as follows:

        output = c_skip * input + c_out * model(c_in * input, c_noise)

    where Karras et al. (2022) originally defined c_skip, c_out, c_in, c_noise coefficients
    as functions of sigma and sigma_data as follows (see Table 1 in the paper):

        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
        c_noise = ln(sigma) / 4

    Note that these coefficients take a simpler form when computed from log-SNR instead of sigma,
    where log-SNR is deined as 2 log (sigma_data / sigma):

        c_skip = sigmoid(logsnr)
        c_out = sigma_data * sqrt(sigmoid(-logsnr))
        c_in = (1 / sigma_data) * sqrt(sigmoid(logsnr))
        c_noise = logsnr

    where sigmoid(x) = 1 / (1 + exp(-x)) is the sigmoid function. Note that definition of c_noise
    here is slightly different (logsnr is an affine transform of ln(sigma) / 4), but we keep it
    this way for simplicity, and it works equivalently well in practice.

    Reference: https://arxiv.org/abs/2206.00364.
    """

    def __init__(self, model, sigma_data):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def _c_skip(self, logsnr):
        return torch.sigmoid(logsnr)

    def _c_in(self, logsnr):
        return (1 / self.sigma_data) * torch.sqrt(torch.sigmoid(logsnr))

    def _c_out(self, logsnr):
        return self.sigma_data * torch.sqrt(torch.sigmoid(-logsnr))

    def forward(self, input, sigma, **kwargs):
        logsnr = utils.sigma_to_logsnr(sigma, sigma_data=self.sigma_data)
        c_in = utils.expand_dims(self._c_in(logsnr), input.ndim)
        c_out = utils.expand_dims(self._c_out(logsnr), input.ndim)
        c_skip = utils.expand_dims(self._c_skip(logsnr), input.ndim)
        return c_skip * input + c_out * self.model(c_in * input, logsnr, **kwargs)


class KarrasOptimalDenoiser(Denoiser):
    """Optimal denoiser that analytically minimizes denoising loss on the training data.

    The optimal denoiser depends on the training data and takes the following form:

        D(y, sigma) = [sum_i y_i Gau(y; y_i, sigma**2)] / [sum_i Gau(y; y_i, sigma**2)],
        where Gau(y; y_i, sigma) is Gaussian density with mean y_i and std sigma.

    For details see Appendix B.3 in Karras et al. (2022).

    Reference: https://arxiv.org/abs/2206.00364.
    """

    def __init__(self, train_data, sigma_data):
        super().__init__()
        self.train_data = train_data
        self.sigma_data = sigma_data

    @staticmethod
    def _normal_prob(x, mu, sigma):
        batch_size = x.shape[0]
        prob_per_dim = torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)
        return torch.prod(prob_per_dim.view(batch_size, -1), dim=-1, keepdim=True)

    @torch.no_grad()
    def forward(self, input, sigma, **kwargs):
        del kwargs  # Unused.
        sigma = utils.expand_dims(sigma, input.ndim)

        # Iterate over training data and compute the optimal denoised output for the given sigma.
        output = torch.zeros_like(input)
        normalization = torch.zeros_like(sigma)
        for (y_i,) in self.train_data:
            y_i = y_i.unsqueeze(0).to(input.device)
            p_i = self._normal_prob(input, y_i, sigma)
            output += y_i * p_i
            normalization += p_i
        output /= normalization

        return output

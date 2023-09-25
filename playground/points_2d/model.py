"""Diffusion model architecture for 2D points data."""
import math

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

# ----------------------------------------------------------------------------
# Step embedding.
# -----------------------------------------------------------------------------


class PositionalEmbedding(torch.nn.Module):
    """Computes harmonic embeddings of the steps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        device = x.device
        half_dim = self.dim // 2
        freq_coeff = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -freq_coeff)
        # Compuate t_i * freq_j matrix.
        time_x_freq = x[:, None] * freq[None, :]
        # Embedding is concatenation of sin and cos of t_i * freq_j.
        emb = torch.cat((time_x_freq.sin(), time_x_freq.cos()), dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, [0, 1])  # padding the last dimension
        return emb


# -----------------------------------------------------------------------------
# Point denoising model architecture.
# -----------------------------------------------------------------------------

NORM_CLS = torch.nn.LayerNorm
activation_fn = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)


class PointDenoisingMLP(torch.nn.Module):
    """A single block of the point denoising MLP model."""

    def __init__(self, input_dim: int, output_dim: int, time_emb_dim: int) -> None:
        super().__init__()

        # Coordinate transform layers.
        self.input_norm = NORM_CLS(input_dim)
        self.fc1 = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.hidden_norm = NORM_CLS(output_dim)
        self.fc2 = torch.nn.Linear(output_dim, output_dim, bias=False)
        self.skip_fc = torch.nn.Linear(input_dim, output_dim, bias=False)

        # Time transform layers.
        self.time_fc = torch.nn.Linear(time_emb_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.fc1(activation_fn(self.input_norm(x)))
        h += self.time_fc(t_emb)
        h = self.fc2(activation_fn(self.hidden_norm(h)))
        return h + self.skip_fc(x)


class PointDenoisingModel(torch.nn.Module):
    """Denoising diffusion model for 2D or 3D point data (each point is a datapoint).

    Architecture:
    - Input: (N, D) tensor of N points in D dimensions (D=2 or D=3).
    - Output: (N, D) tensor of N points in D dimensions (D=2 or D=3).
    - The model consists of multiple PointDenoisingMLP blocks.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

        # Input coordinates and time transforms.
        self.coord_transform = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.time_transform = torch.nn.Sequential(
            PositionalEmbedding(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
        )

        # Layers.
        self.blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(PointDenoisingMLP(hidden_dim, hidden_dim, hidden_dim))

        # Output tranform layers.
        self.output_norm = NORM_CLS(hidden_dim)
        self.output_fc = torch.nn.Linear(hidden_dim, input_dim)

        # Initialize output layer weights with zeros.
        torch.nn.init.zeros_(self.output_fc.weight)
        torch.nn.init.zeros_(self.output_fc.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.coord_transform(x)
        t_emb = self.time_transform(t)
        for block in self.blocks:
            x = block(x, t_emb)
        output = self.output_fc(activation_fn(self.output_norm(x)))
        return output

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_swiss_roll


def make_swiss_roll_dataframe(
    n_samples: int, n_dims: int = 2, noise: float = 0.0, scaling_factor: float = 10.0
) -> pd.DataFrame:
    coords_3d, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
    coords_3d /= scaling_factor
    if n_dims == 2:
        return pd.DataFrame({"x": coords_3d[:, 0], "y": coords_3d[:, 2]})
    else:
        assert n_dims == 3
        return pd.DataFrame({"x": coords_3d[:, 0], "y": coords_3d[:, 1], "z": coords_3d[:, 2]})


def create_train_val_datasets(
    n_train: int = 2**12, n_val: int = 2**9, n_dims: int = 2, noise: float = 0.0
):
    train_df = make_swiss_roll_dataframe(n_samples=n_train, n_dims=n_dims, noise=noise)
    train_data_tensor = torch.from_numpy(train_df.to_numpy().astype(np.float32))
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor)
    val_df = make_swiss_roll_dataframe(n_samples=n_val, n_dims=n_dims, noise=noise)
    val_data_tensor = torch.from_numpy(val_df.to_numpy().astype(np.float32))
    val_dataset = torch.utils.data.TensorDataset(val_data_tensor)
    return train_dataset, val_dataset

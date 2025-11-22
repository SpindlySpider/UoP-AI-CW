from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from typing import Optional, Tuple


class GaitDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, normalize: bool = True):
        # expect inputs and targets as numpy arrays of shape (N, features)
        if normalize:
            # reuse the project's normalization: (x + 50)/80
            inputs = (np.array(inputs) + 50.0) / 80.0
            targets = (np.array(targets) + 50.0) / 80.0

        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_dataloader(inputs: np.ndarray, targets: np.ndarray, batch_size: int = 16, shuffle: bool = True, normalize: bool = True, num_workers: int = 0) -> DataLoader:
    ds = GaitDataset(inputs, targets, normalize=normalize)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

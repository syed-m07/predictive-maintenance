"""PyTorch wrappers for the 30-cycle sensor windows (TCN, Bi-LSTM, uni-LSTM).

A Dataset answers two questions - "how many samples?" and "give me sample i" - and a
DataLoader batches and shuffles them. Windows come from feature_engineering.generate_sequences.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import SEED


class RULDataset(Dataset):
    """X: (n, seq_len, n_features) windows, y: (n,) RUL at each window's last cycle."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.y = None if y is None else torch.as_tensor(np.asarray(y, dtype=np.float32))
        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError("X and y lengths differ")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


def make_loader(X, y=None, batch_size: int = 256, shuffle: bool = False, seed: int = SEED) -> DataLoader:
    """Shuffled training loader uses a seeded generator, so batch order is reproducible."""
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(RULDataset(X, y), batch_size=batch_size, shuffle=shuffle, generator=gen)

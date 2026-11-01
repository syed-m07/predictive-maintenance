"""Temporal Convolutional Network (Bai, Kolter & Koltun, 2018) for RUL.

Three ideas:
1. CAUSAL convolution - the output at cycle t is computed only from cycles <= t. A normal
   Conv1d with padding p pads BOTH ends, so position t would also see t+1..t+p (the future).
   Fix: pad (k-1)*d, then CHOMP (cut) the extra (k-1)*d steps off the right end. Getting
   the chomp wrong silently leaks future steps - training still "works" - so it is tested
   explicitly (tests/test_models_torch.py).
2. DILATION - a dilated filter skips d-1 steps between taps. Stacking dilations 1, 2, 4 ...
   grows the receptive field (how many past cycles one output can see) exponentially.
3. RESIDUAL blocks - each block adds its input back to its output, which keeps deep stacks
   trainable.
Receptive field = 1 + 2*(k-1)*sum(dilations); default k=3, dilations 1,2,4 -> 29 cycles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from src.config import SEED
from src.models.base import RULModel, read_metadata, write_metadata
from src.models.utils import TrainConfig, get_device, predict_torch, train_torch_model

WEIGHTS_FILE = "model.pt"
DEFAULT_ARCH = {"channels": [32, 32, 64], "kernel_size": 3, "dropout": 0.2}


class Chomp1d(nn.Module):
    """Remove the last `chomp` time steps (the right-side padding that would see the future)."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    """[causal dilated conv -> ReLU -> dropout] x 2, plus a residual connection."""

    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(n_in, n_out, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(n_out, n_out, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        # 1x1 conv matches channel counts so input and output can be added
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class TCNNet(nn.Module):
    """Input (batch, time, features) -> RUL (batch,)."""

    def __init__(self, n_features: int, channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks, n_in = [], n_features
        for i, n_out in enumerate(channels):
            blocks.append(TemporalBlock(n_in, n_out, kernel_size, dilation=2 ** i, dropout=dropout))
            n_in = n_out
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(n_in, 1)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(2 ** i for i in range(len(channels)))

    def forward_sequence(self, x):
        """Per-timestep features (batch, channels, time) - used by the causality test."""
        return self.tcn(x.transpose(1, 2))           # Conv1d wants (batch, channels, time)

    def forward(self, x):
        h = self.forward_sequence(x)[:, :, -1]          # last time step summarises the window
        return self.head(h).squeeze(-1)


class TCNRUL(RULModel):
    model_type = "tcn"
    input_kind = "sequence"

    def __init__(self, arch: dict | None = None, train_cfg: dict | None = None, device: str | None = None):
        self.arch = {**DEFAULT_ARCH, **(arch or {})}
        self.train_cfg = TrainConfig(**(train_cfg or {}))
        self.device = get_device(device)
        self.net: TCNNet | None = None
        self.n_features: int | None = None
        self.history = None

    def _build(self, n_features: int) -> TCNNet:
        torch.manual_seed(self.train_cfg.seed)
        self.n_features = n_features
        return TCNNet(n_features, self.arch["channels"], self.arch["kernel_size"], self.arch["dropout"])

    def fit(self, X, y, X_val=None, y_val=None, verbose: bool = True, trial_callback=None) -> "TCNRUL":
        if X_val is None:
            raise ValueError("TCN training needs validation data for early stopping")
        self.net = self._build(X.shape[2])
        self.history = train_torch_model(self.net, X, y, X_val, y_val, self.train_cfg, self.device,
                                         verbose=verbose, trial_callback=trial_callback)
        self.net.to("cpu"); self.device = torch.device("cpu")   # serve/evaluate on CPU (V4)
        return self

    def predict(self, X) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[2] != self.n_features:
            raise ValueError(f"expected (n, seq_len, {self.n_features}), got {X.shape}")
        return predict_torch(self.net, X, self.device)

    def save(self, bundle_dir: Path | str, metadata: dict | None = None) -> Path:
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), bundle_dir / WEIGHTS_FILE)
        h = self.history
        write_metadata(bundle_dir, {
            "model_type": self.model_type, "input_kind": self.input_kind, "arch": self.arch,
            "train_cfg": self.train_cfg.__dict__, "n_features": self.n_features,
            "receptive_field": self.net.receptive_field,
            "history": None if h is None else {"train_loss": h.train_loss, "val_rmse": h.val_rmse,
                                               "lr": h.lr, "best_epoch": h.best_epoch, "seconds": h.seconds},
            **(metadata or {})})
        return bundle_dir

    @classmethod
    def load(cls, bundle_dir: Path | str) -> "TCNRUL":
        meta = read_metadata(bundle_dir)
        if meta["model_type"] != cls.model_type:
            raise ValueError(f"bundle holds a {meta['model_type']} model, not tcn")
        obj = cls(arch=meta["arch"], train_cfg=meta["train_cfg"], device="cpu")
        obj.net = obj._build(meta["n_features"])
        # weights_only=True: load tensors only - a tampered file cannot execute code
        obj.net.load_state_dict(torch.load(Path(bundle_dir) / WEIGHTS_FILE, map_location="cpu", weights_only=True))
        obj.net.eval()
        return obj

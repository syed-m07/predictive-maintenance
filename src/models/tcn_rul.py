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

from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from src.models.torch_base import TorchRULModel
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


class TCNRUL(TorchRULModel):
    """Fit/predict/save/load come from TorchRULModel; only the network is TCN-specific."""
    model_type = "tcn"
    DEFAULT_ARCH = DEFAULT_ARCH

    def _make_net(self, n_features: int) -> TCNNet:
        return TCNNet(n_features, self.arch["channels"], self.arch["kernel_size"], self.arch["dropout"])

    def _extra_metadata(self) -> dict:
        return {"receptive_field": self.net.receptive_field}

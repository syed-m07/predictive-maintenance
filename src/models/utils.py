"""Shared PyTorch training loop for the sequence models (TCN, Bi-LSTM, uni-LSTM).

One loop for all deep models, so they train under identical rules:
  forward -> MSE loss -> backward -> clip gradients -> optimizer step     (each batch)
  validation RMSE with no_grad -> LR scheduler -> early stopping            (each epoch)
The weights from the best validation epoch are restored at the end.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from src.config import SEED
from src.dataset import make_loader


def get_device(prefer: str | None = None) -> torch.device:
    """cuda on Colab's T4 if available, else cpu. Override with prefer='cpu'."""
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 8              # epochs without val improvement before stopping
    lr_patience: int = 3           # epochs without improvement before halving the LR
    lr_factor: float = 0.5
    grad_clip: float = 1.0         # max gradient norm - prevents exploding updates
    seed: int = SEED


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    val_rmse: list = field(default_factory=list)
    lr: list = field(default_factory=list)
    best_epoch: int = -1
    best_val_rmse: float = float("inf")
    seconds: float = 0.0


def _forward(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """Models may return (prediction, extras) - e.g. attention weights. Keep the prediction."""
    out = model(xb)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def predict_torch(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    preds = [_forward(model, xb.to(device)).cpu() for xb in make_loader(X, None, batch_size)]
    return torch.cat(preds).numpy().astype(np.float64)


def train_torch_model(model: nn.Module, X_train, y_train, X_val, y_val, cfg: TrainConfig,
                      device: torch.device, verbose: bool = True, trial_callback=None) -> TrainHistory:
    """Train with early stopping on validation RMSE; restore the best epoch's weights.

    trial_callback(epoch, val_rmse) lets Optuna prune hopeless trials early (raises to stop).
    """
    torch.manual_seed(cfg.seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True, seed=cfg.seed)
    y_val = np.asarray(y_val, dtype=np.float64)
    hist, best_state, bad_epochs = TrainHistory(), None, 0
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(_forward(model, xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(xb); n += len(xb)
        val_pred = np.clip(predict_torch(model, X_val, device), 0, None)
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
        sched.step(val_rmse)
        hist.train_loss.append(total / n); hist.val_rmse.append(val_rmse); hist.lr.append(opt.param_groups[0]["lr"])
        if val_rmse < hist.best_val_rmse - 1e-4:
            hist.best_val_rmse, hist.best_epoch, bad_epochs = val_rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        if verbose:
            print(f"  epoch {epoch + 1:3d}  train MSE {hist.train_loss[-1]:9.2f}  val RMSE {val_rmse:7.3f}"
                  f"  lr {hist.lr[-1]:.1e}{'  *' if bad_epochs == 0 else ''}", flush=True)
        if trial_callback is not None:
            trial_callback(epoch, val_rmse)
        if bad_epochs >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist.seconds = time.perf_counter() - t0
    return hist

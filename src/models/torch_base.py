"""Shared wrapper for the PyTorch sequence models (TCN, Bi-LSTM, uni-LSTM).

Subclasses only say how to build their network (`_make_net`) and what their default
architecture is; fitting, prediction, input checks and bundle save/load live here once, so
every deep model is trained, saved and loaded identically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.base import RULModel, read_metadata, write_metadata
from src.models.utils import TrainConfig, get_device, predict_torch, train_torch_model

WEIGHTS_FILE = "model.pt"


class TorchRULModel(RULModel):
    input_kind = "sequence"
    DEFAULT_ARCH: dict = {}

    def __init__(self, arch: dict | None = None, train_cfg: dict | None = None, device: str | None = None):
        self.arch = {**self.DEFAULT_ARCH, **(arch or {})}
        self.train_cfg = TrainConfig(**(train_cfg or {}))
        self.device = get_device(device)
        self.net: nn.Module | None = None
        self.n_features: int | None = None
        self.history = None

    # ---- subclasses implement
    def _make_net(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def _extra_metadata(self) -> dict:
        return {}

    # ---- shared
    def _build(self, n_features: int) -> nn.Module:
        torch.manual_seed(self.train_cfg.seed)
        self.n_features = n_features
        return self._make_net(n_features)

    def fit(self, X, y, X_val=None, y_val=None, verbose: bool = True, trial_callback=None):
        if X_val is None:
            raise ValueError(f"{self.model_type} training needs validation data for early stopping")
        self.net = self._build(X.shape[2])
        self.history = train_torch_model(self.net, X, y, X_val, y_val, self.train_cfg, self.device,
                                         verbose=verbose, trial_callback=trial_callback)
        self.net.to("cpu"); self.device = torch.device("cpu")   # evaluate/serve on CPU (V4)
        return self

    def _check(self, X) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[2] != self.n_features:
            raise ValueError(f"expected (n, seq_len, {self.n_features}), got {X.shape}")
        return X

    def predict(self, X) -> np.ndarray:
        return predict_torch(self.net, self._check(X), self.device)

    def save(self, bundle_dir: Path | str, metadata: dict | None = None) -> Path:
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), bundle_dir / WEIGHTS_FILE)
        h = self.history
        write_metadata(bundle_dir, {
            "model_type": self.model_type, "input_kind": self.input_kind, "arch": self.arch,
            "train_cfg": self.train_cfg.__dict__, "n_features": self.n_features,
            **self._extra_metadata(),
            "history": None if h is None else {"train_loss": h.train_loss, "val_rmse": h.val_rmse,
                                               "lr": h.lr, "best_epoch": h.best_epoch, "seconds": h.seconds},
            **(metadata or {})})
        return bundle_dir

    @classmethod
    def load(cls, bundle_dir: Path | str):
        meta = read_metadata(bundle_dir)
        if meta["model_type"] != cls.model_type:
            raise ValueError(f"bundle holds a {meta['model_type']} model, not {cls.model_type}")
        obj = cls(arch=meta["arch"], train_cfg=meta["train_cfg"], device="cpu")
        obj.net = obj._build(meta["n_features"])
        # weights_only=True: load tensors only - a tampered file cannot execute code
        obj.net.load_state_dict(torch.load(Path(bundle_dir) / WEIGHTS_FILE, map_location="cpu", weights_only=True))
        obj.net.eval()
        return obj

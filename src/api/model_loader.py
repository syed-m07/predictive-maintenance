"""Load the served models ONCE at startup and turn raw readings into predictions.

Each ServedModel keeps its model, the scaler it was trained with and its metadata together
(they come from the same bundle), so a model can never be paired with the wrong scaler or
column order. Preprocessing calls the SAME functions as training (src.feature_engineering);
tests/test_api.py checks that the API reproduces the official test predictions exactly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MAX_RUL, RAW_FEATURE_COLS, ROOT_DIR, SEQ_LEN
from src.feature_engineering import (apply_scaler, build_tabular_features, drop_low_variance,
                                     last_window_per_engine, load_scaler)
from src.models.base import SCALER_FILE
from src.models.loader import load_bundle, resolve_bundle_path
from src.registry import get_latest

DEFAULT_CONFIG = ROOT_DIR / "configs" / "serving.json"


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def load_config(path: str | Path | None = None) -> dict:
    """configs/serving.json, or the file named by PM_SERVING_CONFIG."""
    path = _abs(path or os.environ.get("PM_SERVING_CONFIG", DEFAULT_CONFIG))
    cfg = json.loads(path.read_text())
    if not cfg.get("models"):
        raise ValueError(f"{path}: 'models' is empty")
    if cfg.get("default") not in cfg["models"]:
        raise ValueError(f"{path}: default {cfg.get('default')!r} is not one of {cfg['models']}")
    return cfg


def readings_frame(readings: list[list[float]]) -> pd.DataFrame:
    """Raw rows -> the C-MAPSS frame layout (unit_id, cycle, op1..op3, s1..s21)."""
    X = np.asarray(readings, dtype=np.float64)
    df = pd.DataFrame(X, columns=RAW_FEATURE_COLS)
    df.insert(0, "cycle", np.arange(1, len(df) + 1))
    df.insert(0, "unit_id", 1)
    return df


@dataclass
class ServedModel:
    name: str
    version: str
    model: object
    scaler: object
    meta: dict
    registry_row: dict = field(default_factory=dict)

    @property
    def input_kind(self) -> str:
        return self.meta["input_kind"]

    def model_input(self, raw: pd.DataFrame):
        """Exactly the training/test pipeline: drop -> scale -> last feature row / last window."""
        scaled = apply_scaler(drop_low_variance(raw), self.scaler)
        if self.input_kind == "tabular":
            return build_tabular_features(scaled).iloc[[-1]]
        if self.input_kind == "sequence":
            X, _ = last_window_per_engine(scaled, int(self.meta.get("seq_len", SEQ_LEN)))
            return X
        raise ValueError(f"{self.name}: unknown input_kind {self.input_kind!r}")

    def predict_raw(self, raw: pd.DataFrame) -> float:
        pred = np.asarray(self.model.predict(self.model_input(raw)), dtype=np.float64).ravel()
        return float(np.clip(pred[0], 0.0, MAX_RUL))


def _plain(v):
    """Registry values come back as numpy scalars; JSON needs plain Python (NaN -> None)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (not isinstance(v, str) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def load_served_models(cfg: dict) -> dict[str, ServedModel]:
    """Latest registry version of every configured model, with its bundle scaler."""
    registry = _abs(cfg.get("registry", "trained_models/model_registry.csv"))
    served = {}
    for name in cfg["models"]:
        row = get_latest(name, registry)
        if row is None:
            raise LookupError(f"model {name!r} is not in the registry {registry}")
        bundle = resolve_bundle_path(row["path"])
        model, meta = load_bundle(bundle)
        served[name] = ServedModel(name=name, version=str(int(row["version"])), model=model,
                                   scaler=load_scaler(bundle / SCALER_FILE), meta=meta,
                                   registry_row={k: _plain(v) for k, v in row.items()})
    return served


def warm_up(served: dict[str, ServedModel], n_cycles: int = SEQ_LEN) -> None:
    """One throw-away prediction per model, so the first real request doesn't pay one-off costs
    (lazy initialisation, memory allocation)."""
    raw = readings_frame([[0.0] * len(RAW_FEATURE_COLS)] * n_cycles)
    for m in served.values():
        m.predict_raw(raw)

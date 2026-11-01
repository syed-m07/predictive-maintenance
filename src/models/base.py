"""Common interface for every RUL model (XGBoost, TCN, LSTMs, ROCKET).

Same five methods everywhere, so training scripts, the comparison script and the API can
treat models interchangeably (decision P4: any model must be servable):
    fit(X, y, X_val=None, y_val=None) -> self
    predict(X) -> np.ndarray                # raw model output (scoring clips to [0, 125])
    evaluate(X, y) -> dict                  # validation-style metrics via src.evaluation
    save(dir) / load(dir)                   # a model is a DIRECTORY ("bundle")

A bundle directory holds the model file(s), the scaler it was trained with, and
metadata.json (model type, input kind, feature order, params, data split). Keeping them
together means the API can never pair a model with the wrong scaler or column order.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.evaluation import evaluate_validation

METADATA_FILE = "metadata.json"
SCALER_FILE = "scaler.joblib"


class RULModel(ABC):
    model_type: str = "base"        # e.g. "xgboost", "tcn"
    input_kind: str = "tabular"     # "tabular" (one feature row) or "sequence" (window)

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None) -> "RULModel": ...

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...

    def evaluate(self, X, y) -> dict[str, float]:
        return evaluate_validation(y, self.predict(X))

    @abstractmethod
    def save(self, bundle_dir: Path | str, metadata: dict | None = None) -> Path: ...

    @classmethod
    @abstractmethod
    def load(cls, bundle_dir: Path | str) -> "RULModel": ...


def write_metadata(bundle_dir: Path, metadata: dict) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))


def read_metadata(bundle_dir: Path | str) -> dict:
    return json.loads((Path(bundle_dir) / METADATA_FILE).read_text())

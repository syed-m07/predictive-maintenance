"""Load any saved model bundle by reading its metadata.json (decision P4: any model is servable).

    model, meta = load_bundle("trained_models/lstm_uni_tuned_v1")

The bundle's `model_type` picks the class. Classes are imported only when needed, so a
process that serves only XGBoost never imports torch or sktime (the slim API image has
neither). Used by the comparison script, the SHAP analysis and the API.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from src.config import MODELS_DIR
from src.models.base import RULModel, read_metadata

# model_type -> "module:Class"
MODEL_CLASSES: dict[str, str] = {
    "xgboost": "src.models.xgboost_rul:XGBoostRUL",
    "tcn": "src.models.tcn_rul:TCNRUL",
    "lstm": "src.models.lstm_rul:LSTMRUL",
    "rocket": "src.models.rocket_rul:ROCKETRUL",
}


def model_class(model_type: str) -> type[RULModel]:
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"unknown model_type {model_type!r}; known: {sorted(MODEL_CLASSES)}")
    module, cls = MODEL_CLASSES[model_type].split(":")
    return getattr(importlib.import_module(module), cls)


def resolve_bundle_path(path: Path | str) -> Path:
    """Registry paths are relative to the models folder's PARENT, exactly as finish_run writes
    them (e.g. trained_models/tcn_v1 -> repo root). Resolving against MODELS_DIR.parent keeps
    this right when PM_MODELS_DIR redirects a smoke run to a scratch folder."""
    p = Path(path)
    return p if p.is_absolute() else MODELS_DIR.parent / p


def load_bundle(path: Path | str) -> tuple[RULModel, dict]:
    """Return (model, metadata) for the bundle directory at `path`."""
    bundle = resolve_bundle_path(path)
    if not bundle.is_dir():
        raise FileNotFoundError(f"no model bundle at {bundle}")
    meta = read_metadata(bundle)
    return model_class(meta["model_type"]).load(bundle), meta

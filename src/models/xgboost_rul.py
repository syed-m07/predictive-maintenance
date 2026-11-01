"""XGBoost regressor on the 238 tabular features.

Gradient boosting builds hundreds of small trees one after another; each new tree is fit to
the errors the previous trees still make. Early stopping watches the validation engines and
keeps the number of trees at which validation error was lowest - more trees after that point
would only memorise the training engines.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import SEED
from src.models.base import RULModel, read_metadata, write_metadata

MODEL_FILE = "model.json"   # XGBoost's native format: portable, and unlike pickle it cannot run code on load

DEFAULT_PARAMS = {          # HANDOFF section 10.5 starting config (the "baseline")
    "n_estimators": 1000,   # upper bound; early stopping picks the actual number
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "reg:squarederror",
}
EARLY_STOPPING_ROUNDS = 50


class XGBoostRUL(RULModel):
    model_type = "xgboost"
    input_kind = "tabular"

    def __init__(self, params: dict | None = None, early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
                 seed: int = SEED):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.feature_names: list[str] | None = None
        self.model: xgb.XGBRegressor | None = None

    def _check_columns(self, X) -> np.ndarray:
        """Inputs must arrive with the exact training column order - a silent column swap would
        produce confident nonsense, so a DataFrame is re-ordered and validated by name."""
        if isinstance(X, pd.DataFrame):
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"missing feature columns: {sorted(missing)[:5]} ...")
            return X[self.feature_names].to_numpy(dtype=np.float32)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected (n, {len(self.feature_names)}) features, got {X.shape}")
        return X

    def fit(self, X: pd.DataFrame, y, X_val: pd.DataFrame | None = None, y_val=None) -> "XGBoostRUL":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("fit expects a DataFrame so feature names are recorded")
        self.feature_names = list(X.columns)
        use_es = X_val is not None and self.early_stopping_rounds
        self.model = xgb.XGBRegressor(
            **self.params, random_state=self.seed, eval_metric="rmse",
            early_stopping_rounds=self.early_stopping_rounds if use_es else None)
        eval_set = [(self._check_columns(X_val), np.asarray(y_val))] if use_es else None
        self.model.fit(self._check_columns(X), np.asarray(y), eval_set=eval_set, verbose=False)
        return self

    @property
    def best_iteration(self) -> int | None:
        return getattr(self.model, "best_iteration", None)

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model.predict(self._check_columns(X))

    def feature_importance(self, kind: str = "gain") -> pd.Series:
        """Average loss reduction (gain) contributed by splits on each feature."""
        scores = self.model.get_booster().get_score(importance_type=kind)
        # booster names features f0..fN when trained from numpy; map back to real names
        named = {self.feature_names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: v
                 for k, v in scores.items()}
        return pd.Series(named).reindex(self.feature_names).fillna(0.0).sort_values(ascending=False)

    def save(self, bundle_dir: Path | str, metadata: dict | None = None) -> Path:
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(bundle_dir / MODEL_FILE)
        write_metadata(bundle_dir, {
            "model_type": self.model_type, "input_kind": self.input_kind,
            "feature_names": self.feature_names, "params": self.params,
            "early_stopping_rounds": self.early_stopping_rounds, "seed": self.seed,
            "best_iteration": self.best_iteration, **(metadata or {})})
        return bundle_dir

    @classmethod
    def load(cls, bundle_dir: Path | str) -> "XGBoostRUL":
        meta = read_metadata(bundle_dir)
        if meta["model_type"] != cls.model_type:
            raise ValueError(f"bundle holds a {meta['model_type']} model, not xgboost")
        obj = cls(params=meta["params"], early_stopping_rounds=meta["early_stopping_rounds"], seed=meta["seed"])
        obj.feature_names = meta["feature_names"]
        obj.model = xgb.XGBRegressor()
        obj.model.load_model(Path(bundle_dir) / MODEL_FILE)
        return obj

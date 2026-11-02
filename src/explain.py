"""SHAP explanations for the tabular (XGBoost) and sequence (PyTorch) models, and a per-sensor
comparison between them (planner w7t8).

SHAP in one paragraph: for one prediction, SHAP splits (prediction - base value) into one
contribution per input, so that the contributions add up exactly to the prediction. The base
value is the average prediction over a reference ("background") set. A positive contribution
pushed this engine's predicted RUL UP (more life left), a negative one pushed it DOWN.

  XGBoost   shap.TreeExplainer: exact for tree ensembles; one value per engineered feature.
  PyTorch   shap.GradientExplainer ("expected gradients"): averages gradients along paths from
            background windows to the input. One value per (time step, sensor). It is an
            approximation, so the contributions only roughly add up - we measure the gap.

Comparing the two: XGBoost sees 238 features built from 14 sensors, the deep model sees the
same 14 sensors over 30 cycles. Because SHAP values are additive, a sensor's total effect on
one prediction is the SUM of the values of everything derived from it (all its rolling stats,
lags, EMA - or all 30 time steps). Sensor importance = mean over engines of |that sum|.
If both models rank sensors similarly, the signal is robust rather than a model quirk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SENSORS


# --------------------------------------------------------------------------- tabular
def tree_shap(xgb_model, X: pd.DataFrame):
    """shap.Explanation for an XGBoostRUL on rows X (columns in training order)."""
    import shap
    X = X[xgb_model.feature_names]
    explainer = shap.TreeExplainer(xgb_model.model)
    exp = explainer(X)
    return exp


def tabular_sensor_of(feature: str) -> str:
    """'s11_rollstd_10' -> 's11'; 's11' -> 's11'."""
    return feature.split("_")[0]


def group_by_sensor_tabular(values: np.ndarray, feature_names: list[str],
                            sensors: list[str] = SENSORS) -> np.ndarray:
    """(n, n_features) SHAP values -> (n, n_sensors), summing each sensor's features."""
    values = np.asarray(values)
    owner = np.array([tabular_sensor_of(f) for f in feature_names])
    unknown = set(owner) - set(sensors)
    if unknown:
        raise ValueError(f"features from unknown sensors: {sorted(unknown)}")
    return np.stack([values[:, owner == s].sum(axis=1) for s in sensors], axis=1)


# --------------------------------------------------------------------------- sequence
def _prediction_only(net):
    """Wrap a net that returns (pred, extras) so SHAP sees a (batch, 1) output."""
    import torch

    class PredOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            out = out[0] if isinstance(out, tuple) else out
            return out.reshape(-1, 1)

    return PredOnly(net).eval()


def gradient_shap(torch_model, X_background: np.ndarray, X: np.ndarray, nsamples: int = 200,
                  seed: int = 0) -> np.ndarray:
    """(n, seq_len, n_features) expected-gradients SHAP values for a TorchRULModel."""
    import shap
    import torch
    torch.manual_seed(seed)
    wrapped = _prediction_only(torch_model.net)
    explainer = shap.GradientExplainer(wrapped, torch.as_tensor(np.asarray(X_background, dtype=np.float32)))
    vals = explainer.shap_values(torch.as_tensor(np.asarray(X, dtype=np.float32)), nsamples=nsamples,
                                 rseed=seed)
    vals = np.asarray(vals[0] if isinstance(vals, list) else vals)
    if vals.ndim == 4 and vals.shape[-1] == 1:          # (n, T, F, outputs=1)
        vals = vals[..., 0]
    if vals.shape != np.asarray(X).shape:
        raise AssertionError(f"SHAP values {vals.shape} do not match input {np.asarray(X).shape}")
    return vals


def group_by_sensor_sequence(values: np.ndarray) -> np.ndarray:
    """(n, seq_len, n_sensors) -> (n, n_sensors), summing over time steps."""
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError(f"expected (n, seq_len, n_sensors), got {values.shape}")
    return values.sum(axis=1)


# --------------------------------------------------------------------------- comparison
def sensor_importance(grouped: np.ndarray, sensors: list[str] = SENSORS) -> pd.Series:
    """Mean |per-sensor SHAP| over engines, as a share of the total (sums to 1)."""
    imp = np.abs(np.asarray(grouped)).mean(axis=0)
    if imp.shape != (len(sensors),):
        raise ValueError(f"{imp.shape[0]} importances for {len(sensors)} sensors")
    return pd.Series(imp / imp.sum(), index=sensors)


def compare_rankings(importances: dict[str, pd.Series], top: int = 5) -> pd.DataFrame:
    """Pairwise Spearman rank correlation and top-k overlap between models' sensor importances."""
    from scipy.stats import spearmanr
    names = list(importances)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = importances[a], importances[b].reindex(importances[a].index)
            rho = float(spearmanr(sa.to_numpy(), sb.to_numpy()).statistic)
            top_a, top_b = set(sa.nlargest(top).index), set(sb.nlargest(top).index)
            rows.append({"model_a": a, "model_b": b, "spearman_rho": rho,
                         f"top{top}_overlap": len(top_a & top_b), f"top{top}_shared": " ".join(sorted(top_a & top_b))})
    return pd.DataFrame(rows)

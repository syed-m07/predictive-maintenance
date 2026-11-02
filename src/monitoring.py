"""Data-drift detection: is the sensor data a model now sees still like the data it learned from?

Two tests per sensor, comparing a REFERENCE sample (the model's training engines) with a
CURRENT batch (new, unlabelled readings):

  KS test (Kolmogorov-Smirnov, scipy.stats.ks_2samp)
      ks_stat = the largest vertical gap between the two empirical CDFs (0 = identical,
      1 = no overlap). p_value = how likely a gap this large would be if both samples came
      from the same distribution. Caveat measured on FD001 (DECISIONS W2): with thousands of
      rows, and rows of one engine strongly correlated, tiny harmless differences already give
      p < 0.05, so KS alone over-alarms. It answers "is there a difference?", not "is it big?".

  PSI (Population Stability Index)
      Cut the reference into 10 equal-count bins (its deciles); compare the share of reference
      vs current values per bin:  PSI = sum (cur% - ref%) * ln(cur% / ref%).
      Industry rule of thumb: < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant shift.
      It is an EFFECT SIZE: it does not grow just because the sample is large.

Decision rule (owner-approved, DECISIONS W3): a sensor DRIFTS only if both agree
(KS p < 0.05 AND PSI > 0.2); the batch raises an ALARM if at least 3 sensors drift.

Both tests depend only on the ORDER of values, so they give the same answer on raw or
min-max-scaled sensors (any increasing per-sensor transform); raw values are used.

Vocabulary: covariate shift = P(X) changes while P(Y|X) stays (FD001 -> FD002: new operating
conditions). Concept drift = P(Y|X) itself changes. These tests see only X, so they detect a
change in the inputs - whether it hurts the model must be checked separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

KS_ALPHA = 0.05
PSI_THRESHOLD = 0.2
PSI_MODERATE = 0.1
MIN_FEATURES = 3
PSI_BINS = 10
PSI_EPS = 1e-4          # stands in for an empty bin's share, so ln() stays finite


def _clean(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError(f"{name} sample is empty")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} sample contains NaN or inf")
    return a


def compute_ks_drift(reference, current, alpha: float = KS_ALPHA) -> dict:
    """{'ks_stat', 'p_value', 'drift': p_value < alpha} for one feature."""
    r = ks_2samp(_clean(reference, "reference"), _clean(current, "current"))
    return {"ks_stat": float(r.statistic), "p_value": float(r.pvalue), "drift": bool(r.pvalue < alpha)}


def psi_bin_edges(reference, bins: int = PSI_BINS) -> np.ndarray:
    """Bin edges at the reference quantiles; outer edges open (-inf, +inf) so every current value
    lands in a bin. Repeated quantiles (discrete sensors) are merged, giving fewer bins."""
    edges = np.unique(np.quantile(_clean(reference, "reference"), np.linspace(0, 1, bins + 1))).astype(np.float64)
    if edges.size < 2:                      # constant reference c: bins (<c], (=c], (>c)
        c = edges[0]
        return np.array([-np.inf, np.nextafter(c, -np.inf), c, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def compute_psi(reference, current, bins: int = PSI_BINS, eps: float = PSI_EPS) -> float:
    """Population Stability Index of `current` against `reference` (0 = same shape)."""
    ref, cur = _clean(reference, "reference"), _clean(current, "current")
    edges = psi_bin_edges(ref, bins)
    # right-closed bins, so a value equal to an inner quantile stays with the lower bin
    e = np.bincount(np.searchsorted(edges[1:-1], ref, side="left"), minlength=len(edges) - 1) / ref.size
    a = np.bincount(np.searchsorted(edges[1:-1], cur, side="left"), minlength=len(edges) - 1) / cur.size
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_level(psi: float) -> str:
    return "significant" if psi > PSI_THRESHOLD else ("moderate" if psi > PSI_MODERATE else "stable")


def compute_feature_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, features: list[str],
                          ks_alpha: float = KS_ALPHA, psi_threshold: float = PSI_THRESHOLD) -> pd.DataFrame:
    """One row per feature: ks_stat, p_value, psi, psi_level, ks_drift, psi_drift, drift (both)."""
    missing = [f for f in features if f not in ref_df.columns or f not in cur_df.columns]
    if missing:
        raise KeyError(f"features missing from reference or current data: {missing}")
    rows = []
    for f in features:
        ks = compute_ks_drift(ref_df[f], cur_df[f], ks_alpha)
        psi = compute_psi(ref_df[f], cur_df[f])
        rows.append({"feature": f, "ks_stat": ks["ks_stat"], "p_value": ks["p_value"], "psi": psi,
                     "psi_level": psi_level(psi), "ks_drift": ks["drift"], "psi_drift": psi > psi_threshold,
                     "drift": ks["drift"] and psi > psi_threshold})
    return pd.DataFrame(rows).set_index("feature")


def flag_drift(drift_df: pd.DataFrame, ks_alpha: float = KS_ALPHA, psi_threshold: float = PSI_THRESHOLD,
               min_features: int = MIN_FEATURES) -> bool:
    """ALARM if at least `min_features` features have KS p < ks_alpha AND PSI > psi_threshold."""
    drifting = (drift_df["p_value"] < ks_alpha) & (drift_df["psi"] > psi_threshold)
    return bool(drifting.sum() >= min_features)


def drift_summary(drift_df: pd.DataFrame, **flag_kwargs) -> dict:
    """Batch-level numbers for logs, MLflow and the API's drift_score gauge."""
    ks_alpha = flag_kwargs.get("ks_alpha", KS_ALPHA)
    psi_threshold = flag_kwargs.get("psi_threshold", PSI_THRESHOLD)
    both = (drift_df["p_value"] < ks_alpha) & (drift_df["psi"] > psi_threshold)
    return {"n_features": int(len(drift_df)),
            "n_ks_drift": int((drift_df["p_value"] < ks_alpha).sum()),
            "n_psi_drift": int((drift_df["psi"] > psi_threshold).sum()),
            "n_drift": int(both.sum()),
            "median_psi": float(drift_df["psi"].median()), "max_psi": float(drift_df["psi"].max()),
            "drifted_features": " ".join(drift_df.index[both]),
            "alarm": flag_drift(drift_df, **flag_kwargs)}


# --------------------------------------------------------------------------- reference sample
def save_reference(df: pd.DataFrame, features: list[str], path: Path | str, meta: dict | None = None) -> Path:
    """Store the reference sample (raw feature values of the training engines) next to a model.

    npz, loaded with allow_pickle=False: plain numbers and strings only, nothing executable.
    The full sample is kept (not just summary stats) because KS needs the values themselves."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=df[features].to_numpy(dtype=np.float64),
                        features=np.array(features), meta=np.array(json.dumps(meta or {})))
    return path


def load_reference(path: Path | str) -> tuple[pd.DataFrame, dict]:
    with np.load(Path(path), allow_pickle=False) as z:
        df = pd.DataFrame(z["values"], columns=[str(f) for f in z["features"]])
        meta = json.loads(str(z["meta"]))
    return df, meta

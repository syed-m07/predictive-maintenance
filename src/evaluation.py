"""One scorer for every model, so the comparison table is apples-to-apples.

Metrics (lower is better except R^2):
  rmse   root mean squared error, in cycles - penalises big misses heavily
  mae    mean absolute error, in cycles - the "typical" miss
  r2     share of the variance in true RUL that the predictions explain (1 = perfect)
  nasa   NASA/PHM08 asymmetric score, SUMMED over engines - punishes LATE predictions
         (over-estimating remaining life) more than early ones

Test-set convention (docs/DECISIONS.md P3/P6): every test metric is reported twice -
  primary   vs RUL_FD00X.txt as given                     -> keys rmse, mae, r2, nasa
  clip125   vs the same truth clipped at MAX_RUL          -> keys rmse_clip125, ...
            ("literature-comparable": many papers clip the test truth too)

Predictions are clipped to [0, MAX_RUL] before scoring: RUL cannot be negative, and models
are trained on labels capped at MAX_RUL, so 125 is the largest meaningful prediction.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from src.config import MAX_RUL

# NASA score constants (Saxena et al., 2008): late errors are scaled by 10, early by 13,
# so e^(d/10) grows faster than e^(|d|/13) - being late costs more.
NASA_LATE_DIVISOR = 10.0
NASA_EARLY_DIVISOR = 13.0


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    if t.shape != p.shape:
        raise ValueError(f"y_true has {t.size} values but y_pred has {p.size}")
    if t.size == 0:
        raise ValueError("empty input")
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("inputs contain NaN or inf")
    return t, p


def clip_predictions(y_pred, max_rul: float = MAX_RUL) -> np.ndarray:
    """Clip predictions to the valid range [0, max_rul]."""
    return np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, max_rul)


def rmse(y_true, y_pred) -> float:
    t, p = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true, y_pred) -> float:
    t, p = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def r2(y_true, y_pred) -> float:
    """1 - SS_res / SS_tot. 1 = perfect, 0 = no better than predicting the mean, < 0 = worse."""
    t, p = _as_arrays(y_true, y_pred)
    ss_tot = np.sum((t - t.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")          # undefined when all true values are identical
    return float(1.0 - np.sum((t - p) ** 2) / ss_tot)


def nasa_score_per_engine(y_true, y_pred) -> np.ndarray:
    """Per-engine NASA penalty. d = predicted - true.

    d < 0 (early - maintenance happens too soon):  exp(-d / 13) - 1
    d >= 0 (late - the engine may fail first):     exp( d / 10) - 1
    Zero for a perfect prediction; always >= 0; grows exponentially with the error.
    """
    t, p = _as_arrays(y_true, y_pred)
    d = p - t
    return np.where(d < 0, np.exp(-d / NASA_EARLY_DIVISOR) - 1.0, np.exp(d / NASA_LATE_DIVISOR) - 1.0)


def nasa_score(y_true, y_pred) -> float:
    """NASA score summed over engines (the convention in the C-MAPSS literature).

    Because it is a SUM, it grows with the number of engines: compare scores only on the
    same test set (e.g. FD001's 100 engines), never across subsets of different size.
    """
    return float(np.sum(nasa_score_per_engine(y_true, y_pred)))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """rmse, mae, r2, nasa for already-clipped predictions."""
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
            "r2": r2(y_true, y_pred), "nasa": nasa_score(y_true, y_pred)}


def evaluate_test(y_true_raw, y_pred, max_rul: float = MAX_RUL) -> dict[str, float]:
    """Official test evaluation: one prediction per test engine vs RUL_FD00X.txt.

    Returns the primary metrics (truth as given) and the literature-comparable ones (truth
    clipped at max_rul, suffix _clip125). Predictions are clipped to [0, max_rul] for both.
    """
    pred = clip_predictions(y_pred, max_rul)
    truth = np.asarray(y_true_raw, dtype=np.float64)
    primary = regression_metrics(truth, pred)
    clipped = regression_metrics(np.minimum(truth, max_rul), pred)
    out = dict(primary)
    out.update({f"{k}_clip{int(max_rul)}": v for k, v in clipped.items()})
    return out


def evaluate_validation(y_true_clipped, y_pred, max_rul: float = MAX_RUL) -> dict[str, float]:
    """Validation evaluation: every row/window of the 20 held-out engines, clipped labels.

    This is what early stopping and Optuna optimise. Labels are already clipped, so there is
    only one version. Note: validation covers whole lifetimes (incl. the flat healthy phase)
    while test covers one truncated point per engine, so the two are not directly comparable.
    """
    return regression_metrics(y_true_clipped, clip_predictions(y_pred, max_rul))


def time_inference(predict_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray,
                   n_runs: int = 50, n_warmup: int = 5) -> dict[str, float]:
    """Measure prediction speed on CPU, two ways:

    latency_single_ms      median time to predict ONE engine (what an API caller waits)
    throughput_ms_per_item median time per engine when predicting the whole batch X at once

    Warm-up runs are excluded (first calls pay one-off costs: lazy initialisation, numba
    compilation, memory allocation). Median, not mean, so a stray OS hiccup doesn't skew it.
    """
    if len(X) == 0:
        raise ValueError("X is empty")
    single = X[:1]
    for _ in range(n_warmup):
        predict_fn(single)
        predict_fn(X)
    singles, batches = [], []
    for _ in range(n_runs):
        t0 = time.perf_counter(); predict_fn(single); singles.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); predict_fn(X); batches.append(time.perf_counter() - t0)
    return {"latency_single_ms": float(np.median(singles) * 1e3),
            "throughput_ms_per_item": float(np.median(batches) * 1e3 / len(X))}

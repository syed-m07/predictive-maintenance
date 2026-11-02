"""Uncertainty for the model comparison: bootstrap confidence intervals over test engines.

Why: the FD001 test set has only 100 engines, so a test RMSE is itself a noisy estimate.
If we had drawn a different 100 engines, every score would move. The bootstrap estimates
how much, using only the data we have:

  1. draw 100 engines WITH replacement from the 100 test engines (some appear twice, some
     not at all) - one "bootstrap test set";
  2. recompute every model's RMSE / NASA score on it;
  3. repeat many times (default 10,000). The middle 95 % of the recomputed scores is the
     95 % percentile confidence interval.

Paired comparison: every model is scored on the SAME resampled engines in each round, so
for two models we can look at the DIFFERENCE of their scores round by round. That is much
sharper than comparing two separate intervals, because both models find the same engines
easy or hard. If the 95 % interval of (model - reference) contains 0, the test set cannot
tell the two apart.

All predictions are clipped to [0, MAX_RUL] first, exactly as src.evaluation scores them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MAX_RUL, SEED
from src.evaluation import clip_predictions, nasa_score_per_engine

N_BOOT = 10_000
CI_LEVEL = 0.95


def bootstrap_indices(n_items: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(n_boot, n_items) engine indices, sampled with replacement. Seeded: same draws every run."""
    if n_items < 2:
        raise ValueError("need at least 2 items to bootstrap")
    return np.random.default_rng(seed).integers(0, n_items, size=(n_boot, n_items))


def _per_engine(truth, pred, max_rul: float) -> tuple[np.ndarray, np.ndarray]:
    """Squared error and NASA penalty per engine, after clipping the prediction."""
    truth = np.asarray(truth, dtype=np.float64)
    p = clip_predictions(pred, max_rul)
    if p.shape != truth.shape:
        raise ValueError(f"{p.size} predictions for {truth.size} engines")
    return (p - truth) ** 2, nasa_score_per_engine(truth, p)


def bootstrap_scores(truth, preds: dict[str, np.ndarray], idx: np.ndarray,
                     max_rul: float = MAX_RUL) -> dict[str, dict[str, np.ndarray]]:
    """For each model: {'rmse': (n_boot,), 'nasa': (n_boot,)} recomputed on every resample."""
    out = {}
    for name, pred in preds.items():
        sq, nasa = _per_engine(truth, pred, max_rul)
        out[name] = {"rmse": np.sqrt(sq[idx].mean(axis=1)), "nasa": nasa[idx].sum(axis=1)}
    return out


def _interval(samples: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(samples, [tail, 100.0 - tail])
    return float(lo), float(hi)


def confidence_intervals(truth, preds: dict[str, np.ndarray], n_boot: int = N_BOOT,
                         seed: int = SEED, level: float = CI_LEVEL,
                         max_rul: float = MAX_RUL) -> pd.DataFrame:
    """One row per model: rmse_ci_lo/hi and nasa_ci_lo/hi (percentile bootstrap)."""
    idx = bootstrap_indices(len(np.asarray(truth)), n_boot, seed)
    scores = bootstrap_scores(truth, preds, idx, max_rul)
    rows = []
    for name, s in scores.items():
        r_lo, r_hi = _interval(s["rmse"], level)
        n_lo, n_hi = _interval(s["nasa"], level)
        rows.append({"model": name, "rmse_ci_lo": r_lo, "rmse_ci_hi": r_hi,
                     "nasa_ci_lo": n_lo, "nasa_ci_hi": n_hi})
    return pd.DataFrame(rows).set_index("model")


def paired_differences(truth, preds: dict[str, np.ndarray], reference: str, n_boot: int = N_BOOT,
                       seed: int = SEED, level: float = CI_LEVEL,
                       max_rul: float = MAX_RUL) -> pd.DataFrame:
    """Each model vs `reference`, on identical resamples. Positive diff = model is WORSE.

    Columns per metric m in (rmse, nasa):
      m_diff                  point difference on the real test set (model - reference)
      m_diff_ci_lo / _hi      percentile interval of the per-resample differences
      m_p_ref_better          share of resamples where the reference scored strictly better
      m_distinguishable       True if the interval excludes 0
    """
    if reference not in preds:
        raise KeyError(f"reference model {reference!r} not in predictions")
    truth = np.asarray(truth, dtype=np.float64)
    idx = bootstrap_indices(len(truth), n_boot, seed)
    scores = bootstrap_scores(truth, preds, idx, max_rul)
    full = bootstrap_scores(truth, preds, np.arange(len(truth))[None, :], max_rul)
    rows = []
    for name in preds:
        if name == reference:
            continue
        row = {"model": name, "reference": reference}
        for m in ("rmse", "nasa"):
            d = scores[name][m] - scores[reference][m]
            lo, hi = _interval(d, level)
            row.update({f"{m}_diff": float(full[name][m][0] - full[reference][m][0]),
                        f"{m}_diff_ci_lo": lo, f"{m}_diff_ci_hi": hi,
                        f"{m}_p_ref_better": float(np.mean(d > 0)),
                        f"{m}_distinguishable": bool(lo > 0 or hi < 0)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------------------- model inputs
def test_inputs(d, seq_len: int) -> dict[str, object]:
    """The official test inputs, one per test engine, in RUL_FD00X.txt order:
    'tabular' (last feature row, DataFrame) and 'sequence' (last seq_len-cycle window)."""
    from src.feature_engineering import build_tabular_features, last_row_per_engine, last_window_per_engine
    tab = last_row_per_engine(build_tabular_features(d.test))
    X_seq, units = last_window_per_engine(d.test, seq_len)
    if tab["unit_id"].tolist() != units.tolist():
        raise AssertionError("tabular and sequence test rows are in different engine orders")
    return {"tabular": tab, "sequence": X_seq, "unit_id": units}


def pick_trajectory_engines(d, targets=(10, 50, 95), min_cycles: int = 150) -> list[int]:
    """Test engines for the multi-model plot: among engines with >= min_cycles of history, the
    one whose true final RUL is closest to each target (ties -> longer history). Near failure,
    mid-life and healthy-ish, each with a long trajectory to look at."""
    lengths = d.test.groupby("unit_id").size()
    truth = dict(zip(sorted(d.test["unit_id"].unique()), d.test_truth))
    cand = [u for u in lengths.index if lengths[u] >= min_cycles]
    picks = []
    for t in targets:
        u = min((u for u in cand if u not in picks), key=lambda u: (abs(truth[u] - t), -lengths[u]))
        picks.append(int(u))
    return picks


def engine_trajectory_inputs(d, unit: int, seq_len: int) -> dict[str, object]:
    """Inputs for every cycle from seq_len to the last recorded cycle of ONE test engine.

    True RUL at cycle c = (RUL after the last cycle, from RUL_FD00X.txt) + (last cycle - c):
    the engine loses one cycle of life per cycle. Only past data enters each input (windows
    end at c; tabular features are past-only), so this is what a live system would have seen.
    """
    from src.feature_engineering import build_tabular_features, generate_sequences
    units = sorted(d.test["unit_id"].unique())
    final_rul = float(d.test_truth[units.index(unit)])
    eng = d.test[d.test["unit_id"] == unit].sort_values("cycle")
    eng = eng.assign(rul=final_rul + eng["cycle"].max() - eng["cycle"])
    X_seq, truth, _ = generate_sequences(eng, seq_len)
    tab = build_tabular_features(eng)
    tab = tab[tab["cycle"] >= eng["cycle"].iloc[seq_len - 1]].reset_index(drop=True)
    cycles = eng["cycle"].to_numpy()[seq_len - 1:]
    if not (len(tab) == len(X_seq) == len(cycles)):
        raise AssertionError("trajectory inputs are misaligned")
    return {"cycle": cycles, "truth": truth.astype(np.float64), "tabular": tab, "sequence": X_seq}


def predict_kind(model, meta: dict, inputs: dict) -> np.ndarray:
    """Route the right input (tabular / sequence) to a model, by its bundle metadata."""
    kind = meta["input_kind"]
    if kind not in ("tabular", "sequence"):
        raise ValueError(f"unknown input_kind {kind!r}")
    return np.asarray(model.predict(inputs[kind]), dtype=np.float64)

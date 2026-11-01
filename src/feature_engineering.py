"""Turn raw C-MAPSS rows into model inputs - without leaking information.

Pipeline order (it matters):
  1. drop_low_variance      remove the 10 uninformative columns (EDA decision D6)
  2. compute_rul + clip_rul training labels, capped at MAX_RUL (training/val only)
  3. split_by_engine        hold out whole ENGINES for validation (never rows)
  4. fit_scaler             learn min/max from the TRAINING engines only, then apply the
                            same numbers to validation, test and API inputs
  5a. build_tabular_features  trailing rolling / lag / EMA features   -> XGBoost
  5b. generate_sequences      sliding 30-cycle windows of the 14 sensors -> TCN, LSTMs, ROCKET
  6. last_row_per_engine / last_window_per_engine   one input per TEST engine

Three leakage rules enforced here (and tested in tests/test_feature_pipeline.py):
  - no engine is in both train and validation;
  - the scaler never sees validation/test rows;
  - every feature at cycle t uses only cycles <= t of the SAME engine (past-only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (DROP_COLS, EMA_ALPHA, LAGS, MAX_RUL, ROLL_WINDOWS, SEED, SENSORS,
                        SEQ_LEN, VAL_FRACTION)

ROLL_STATS = ("mean", "std", "min", "max")


# --------------------------------------------------------------------------- cleaning & labels
def drop_low_variance(df: pd.DataFrame, cols: list[str] = DROP_COLS) -> pd.DataFrame:
    """Remove columns the EDA showed carry no information (see notebooks/01_EDA.ipynb)."""
    return df.drop(columns=[c for c in cols if c in df.columns])


def clip_rul(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """Piecewise-linear target: RUL above max_rul becomes max_rul. Returns a copy.

    Used for TRAINING/VALIDATION labels only. Test ground truth is not clipped for the
    primary metric (docs/DECISIONS.md P3/P6).
    """
    out = df.copy()
    out["rul"] = out["rul"].clip(upper=max_rul)
    return out


# --------------------------------------------------------------------------- split
def split_by_engine(df: pd.DataFrame, val_fraction: float = VAL_FRACTION,
                    seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a random set of WHOLE engines for validation.

    Why not a random row split: rows of the same engine are near-duplicates of their
    neighbours; a row split would put cycle 100 in train and cycle 101 in validation, so
    validation would measure memory, not generalisation to an unseen engine - which is
    exactly what the model faces in production and on the test set.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    units = np.sort(df["unit_id"].unique())
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(units) * val_fraction)))
    val_units = set(rng.choice(units, size=n_val, replace=False).tolist())
    is_val = df["unit_id"].isin(val_units)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


# --------------------------------------------------------------------------- scaling
def fit_scaler(train_df: pd.DataFrame, cols: list[str] = SENSORS) -> MinMaxScaler:
    """Fit ONE global MinMax scaler on training-engine rows only.

    Why global and train-only (HANDOFF section 7.1): a per-engine scaler would need each
    engine's full-lifetime min/max - future information that does not exist while an
    engine is still running - and test engines are different engines anyway. Validation
    and test values may fall slightly outside [0, 1]; that is expected and correct.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[cols].to_numpy())
    scaler.feature_names_ = list(cols)          # remember the column order it was fit on
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, cols: list[str] | None = None) -> pd.DataFrame:
    """Transform the sensor columns with an already-fitted scaler. Returns a copy."""
    cols = cols or scaler.feature_names_
    out = df.copy()
    out[cols] = scaler.transform(out[cols].to_numpy())
    return out


def save_scaler(scaler: MinMaxScaler, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path | str) -> MinMaxScaler:
    return joblib.load(path)


# --------------------------------------------------------------------------- tabular features
def tabular_feature_names(sensors: list[str] = SENSORS, windows: list[int] = ROLL_WINDOWS,
                          lags: list[int] = LAGS) -> list[str]:
    """Deterministic feature order: per sensor -> raw, rolling stats, lags, EMA.

    14 sensors x (1 raw + 3 windows x 4 stats + 3 lags + 1 EMA) = 238 features.
    Naming: {sensor}_{feature}_{param}, e.g. s11_rollstd_10.
    """
    names = []
    for s in sensors:
        names.append(s)
        names += [f"{s}_roll{stat}_{w}" for w in windows for stat in ROLL_STATS]
        names += [f"{s}_lag_{k}" for k in lags]
        names.append(f"{s}_ema")
    return names


def build_tabular_features(df: pd.DataFrame, sensors: list[str] = SENSORS,
                           windows: list[int] = ROLL_WINDOWS, lags: list[int] = LAGS,
                           alpha: float = EMA_ALPHA) -> pd.DataFrame:
    """Add past-only per-engine features for XGBoost. Returns a new DataFrame.

    - rolling mean/std/min/max over the last W cycles (min_periods=1, so early cycles use
      whatever history exists instead of producing NaN). The first row's std is undefined
      (one value) and is set to 0.
    - lag k: the sensor value k cycles ago; before cycle k+1 there is no such value, so the
      engine's first reading is used (the earliest real past value - never a future one).
    - EMA: exponentially weighted mean (alpha=0.3), a smoothed level that reacts faster than
      a plain rolling mean.
    Every statistic is computed inside groupby('unit_id'), so engines never mix.
    """
    df = df.reset_index(drop=True)
    g = df.groupby("unit_id", sort=False)[sensors]
    new = {}
    for w in windows:
        roll = g.rolling(w, min_periods=1)
        for stat in ROLL_STATS:
            vals = getattr(roll, stat)().reset_index(level=0, drop=True).sort_index()
            if stat == "std":
                vals = vals.fillna(0.0)
            for s in sensors:
                new[f"{s}_roll{stat}_{w}"] = vals[s]
    first = g.transform("first")
    for k in lags:
        shifted = g.shift(k).fillna(first)
        for s in sensors:
            new[f"{s}_lag_{k}"] = shifted[s]
    ema = g.ewm(alpha=alpha).mean().reset_index(level=0, drop=True).sort_index()
    for s in sensors:
        new[f"{s}_ema"] = ema[s]

    feats = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    ordered = tabular_feature_names(sensors, windows, lags)
    meta = [c for c in df.columns if c not in sensors]          # unit_id, cycle, rul, ...
    return feats[meta + ordered]


def last_row_per_engine(feat_df: pd.DataFrame) -> pd.DataFrame:
    """The final observed row of each engine (test protocol for tabular models).

    Each test engine has exactly one ground-truth value, valid at its last cycle.
    """
    return feat_df.groupby("unit_id", sort=True).tail(1).reset_index(drop=True)


# --------------------------------------------------------------------------- sequences
def generate_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN, feature_cols: list[str] = SENSORS,
                       target_col: str = "rul") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows (stride 1) inside each engine, for TCN / LSTMs / ROCKET.

    Window i covers cycles i .. i+seq_len-1 of ONE engine; its target is the RUL at the
    window's LAST cycle (the model predicts "now" from the recent past).
    Returns X (n, seq_len, n_features) float32, y (n,) float32, unit_ids (n,).
    Engines shorter than seq_len yield no training windows (none exist in FD001-FD004
    train sets: minimum 128 cycles).
    """
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit_id", sort=True):
        data = g[feature_cols].to_numpy(dtype=np.float32)
        n = len(data) - seq_len + 1
        if n <= 0:
            continue
        idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]   # (n, seq_len) row indices
        xs.append(data[idx])
        ys.append(g[target_col].to_numpy(dtype=np.float32)[seq_len - 1:])
        units.append(np.full(n, unit))
    if not xs:
        raise ValueError(f"no engine has >= {seq_len} cycles")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(units)


def pad_to_length(data: np.ndarray, seq_len: int) -> np.ndarray:
    """Left-pad a (t, f) array to (seq_len, f) by repeating its FIRST row.

    Used when an engine has fewer than seq_len observed cycles (FD002 min 21, FD004 min
    19). Repeating the earliest reading assumes the engine was in that state before
    observation began - it uses no future data. Arrays already long enough keep their
    last seq_len rows.
    """
    if len(data) >= seq_len:
        return data[-seq_len:]
    pad = np.repeat(data[:1], seq_len - len(data), axis=0)
    return np.concatenate([pad, data], axis=0)


def last_window_per_engine(df: pd.DataFrame, seq_len: int = SEQ_LEN,
                           feature_cols: list[str] = SENSORS) -> tuple[np.ndarray, np.ndarray]:
    """One (seq_len, n_features) window per engine: its last seq_len cycles (test protocol).

    Returns X (n_engines, seq_len, n_features) float32 and unit_ids in ascending order,
    which matches the line order of RUL_FD00X.txt.
    """
    xs, units = [], []
    for unit, g in df.groupby("unit_id", sort=True):
        xs.append(pad_to_length(g[feature_cols].to_numpy(dtype=np.float32), seq_len))
        units.append(unit)
    return np.stack(xs), np.asarray(units)


# --------------------------------------------------------------------------- one entry point
@dataclass
class PreparedData:
    """Scaled train/val/test frames + everything needed to evaluate on the official test set.

    train/val: dropped columns, CLIPPED rul labels, scaled sensors.
    test:      dropped columns, scaled sensors, no rul column (test engines did not fail).
    test_truth: RUL_FD00X.txt as given (unclipped), aligned to sorted test unit_ids.
    """
    subset: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    test_truth: np.ndarray
    scaler: MinMaxScaler
    train_units: list[int] = field(default_factory=list)
    val_units: list[int] = field(default_factory=list)


def prepare_data(subset: str = "FD001", val_fraction: float = VAL_FRACTION, seed: int = SEED,
                 drop_cols: list[str] = DROP_COLS, sensors: list[str] = SENSORS,
                 max_rul: int = MAX_RUL) -> PreparedData:
    """Steps 1-4 of the pipeline, in the leakage-safe order. Every training script uses this.

    load -> drop -> RUL labels -> clip -> split by engine -> fit scaler on TRAIN only ->
    apply the same scaler to val and test.
    """
    from src.data_loader import compute_rul, load_cmapss, load_rul_truth   # avoid import cycle

    train_full = clip_rul(compute_rul(drop_low_variance(load_cmapss(subset, "train"), drop_cols)), max_rul)
    train_df, val_df = split_by_engine(train_full, val_fraction, seed)
    scaler = fit_scaler(train_df, sensors)
    test_df = drop_low_variance(load_cmapss(subset, "test"), drop_cols)
    return PreparedData(
        subset=subset,
        train=apply_scaler(train_df, scaler),
        val=apply_scaler(val_df, scaler),
        test=apply_scaler(test_df, scaler),
        test_truth=load_rul_truth(subset),
        scaler=scaler,
        train_units=sorted(train_df["unit_id"].unique().tolist()),
        val_units=sorted(val_df["unit_id"].unique().tolist()),
    )

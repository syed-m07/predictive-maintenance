"""Data for the drift demo and the drift-triggered retraining (Block 12).

Kept apart from src/monitoring.py so the API can import the drift tests without the data
loaders. Every frame here holds RAW sensor values (the tests are scale-free, see monitoring).

Batches compared against the reference (FD001 training engines, the model's own data):
  FD001-val    the 20 FD001 validation engines: same population, full lifetimes
               -> the NO-DRIFT CONTROL; must not alarm (DV-004)
  FD001-test   the 100 FD001 test engines: same population, but cut off before failure, so
               mostly healthy early-life readings -> a harmless shift in health-state mix
  FD002-test   259 engines flying 6 operating conditions -> real covariate shift
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import MAX_RUL, SEED, SENSORS, VAL_FRACTION
from src.data_loader import compute_rul, load_cmapss, load_rul_truth
from src.feature_engineering import apply_scaler, clip_rul, drop_low_variance, fit_scaler, split_by_engine

BATCHES = ("FD001-val", "FD001-test", "FD002-test")
UNIT_OFFSET = 1000          # combined data: FD002 engine 7 becomes 1007, so ids never collide


def training_split(subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw (unscaled, unclipped-sensor) train/val frames of a subset, using THE project split."""
    full = clip_rul(compute_rul(drop_low_variance(load_cmapss(subset, "train"))), MAX_RUL)
    return split_by_engine(full, VAL_FRACTION, SEED)


def reference_frame(subset: str = "FD001") -> pd.DataFrame:
    """The reference sample: raw sensor readings of the subset's training engines."""
    return training_split(subset)[0]


def current_batch(name: str) -> pd.DataFrame:
    """Raw sensor readings of one current batch (see module docstring)."""
    if name == "FD001-val":
        return training_split("FD001")[1]
    if name in ("FD001-test", "FD002-test"):
        return drop_low_variance(load_cmapss(name.split("-")[0], "test"))
    raise ValueError(f"unknown batch {name!r}; choose from {BATCHES}")


def truncate_like_test(df: pd.DataFrame, test_truth: np.ndarray, rng: np.random.Generator,
                       min_cycles: int = 31) -> pd.DataFrame:
    """Cut every run-to-failure engine so its remaining life is a draw from the test-set truth,
    the way NASA's test engines were cut (used to explain why FD001-test shifts)."""
    parts = []
    for _, g in df.groupby("unit_id", sort=True):
        keep = len(g) - int(rng.choice(test_truth))
        if keep >= min_cycles:
            parts.append(g.iloc[:keep])
    return pd.concat(parts, ignore_index=True)


@dataclass
class CombinedData:
    """Training data pooled from several subsets for the retrained model.

    Each subset keeps its OWN 80/20 engine split (FD001 keeps exactly the project's validation
    engines), then the parts are pooled. One scaler is fit on the pooled training engines.
    `tests` maps subset -> (scaled test frame, unclipped truth)."""
    subsets: tuple[str, ...]
    train: pd.DataFrame
    val: pd.DataFrame
    tests: dict[str, tuple[pd.DataFrame, np.ndarray]]
    scaler: MinMaxScaler
    train_units: list[int] = field(default_factory=list)
    val_units: list[int] = field(default_factory=list)
    raw_train: pd.DataFrame | None = None          # unscaled, for the new drift reference


def prepare_combined(subsets=("FD001", "FD002"), sensors: list[str] = SENSORS) -> CombinedData:
    trains, vals, tests_raw = [], [], {}
    for k, s in enumerate(subsets):
        tr, va = training_split(s)
        trains.append(tr.assign(unit_id=tr["unit_id"] + k * UNIT_OFFSET))
        vals.append(va.assign(unit_id=va["unit_id"] + k * UNIT_OFFSET))
        te = drop_low_variance(load_cmapss(s, "test"))
        tests_raw[s] = (te, load_rul_truth(s))
    train, val = pd.concat(trains, ignore_index=True), pd.concat(vals, ignore_index=True)
    if set(train["unit_id"]) & set(val["unit_id"]):
        raise AssertionError("an engine is in both training and validation data")
    scaler = fit_scaler(train, sensors)
    return CombinedData(
        subsets=tuple(subsets), train=apply_scaler(train, scaler), val=apply_scaler(val, scaler),
        tests={s: (apply_scaler(te, scaler), truth) for s, (te, truth) in tests_raw.items()},
        scaler=scaler, train_units=sorted(train["unit_id"].unique().tolist()),
        val_units=sorted(val["unit_id"].unique().tolist()), raw_train=train)

"""Load NASA C-MAPSS files and build Remaining-Useful-Life (RUL) labels.

Three jobs, nothing else:
  1. load_cmapss      - read a train/test file into a correctly named DataFrame
  2. load_rul_truth   - read the official test answers (RUL_FD00X.txt)
  3. compute_rul      - label every TRAINING row with "cycles left until failure"

Feature engineering, scaling and splitting live in feature_engineering.py; keeping this
module tiny means every later stage starts from data whose shape is guaranteed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COLUMNS, DATA_RAW_DIR, SUBSETS

_SPLITS = ("train", "test")


def _check_args(subset: str, split: str | None = None) -> None:
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    if split is not None and split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")


def _validate_structure(df: pd.DataFrame, name: str) -> None:
    """Fail loudly if the file breaks assumptions the feature pipeline relies on.

    Rolling windows and lags are computed per engine assuming rows are ordered by cycle
    with no gaps. If a file violated that, features would be silently wrong - so we check.
    """
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{name}: expected {len(COLUMNS)} columns, got {df.shape[1]}")
    if df.isna().any().any():
        raise ValueError(f"{name}: contains NaN values")
    if not df["unit_id"].is_monotonic_increasing:
        raise ValueError(f"{name}: rows are not grouped/sorted by unit_id")
    # Within each engine the n-th row must be cycle n: starts at 1, no gaps, no duplicates,
    # in order. One vectorised comparison covers all four conditions.
    expected_cycle = df.groupby("unit_id").cumcount() + 1
    if not (df["cycle"] == expected_cycle).all():
        raise ValueError(f"{name}: cycles are not 1, 2, 3, ... within every engine")


def load_cmapss(subset: str, split: str, data_dir: Path | str = DATA_RAW_DIR) -> pd.DataFrame:
    """Read train_FD00X.txt or test_FD00X.txt.

    The files are whitespace-separated, have no header and end every line with trailing
    spaces. sep=r"\\s+" treats any run of whitespace as ONE separator; sep=" " would
    create two extra all-NaN columns (28 instead of 26).

    Returns a DataFrame with columns unit_id, cycle, op1-op3, s1-s21 (26 total),
    unit_id/cycle as int and everything else as float.
    """
    _check_args(subset, split)
    path = Path(data_dir) / f"{split}_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - see data/data_manifest.txt for download steps")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df = df.astype({"unit_id": "int64", "cycle": "int64"})
    _validate_structure(df, path.name)
    return df


def load_rul_truth(subset: str, data_dir: Path | str = DATA_RAW_DIR) -> np.ndarray:
    """Read RUL_FD00X.txt: the true remaining cycles of each TEST engine, in unit_id order.

    This is the ONLY ground truth for the test set. It is valid at the LAST observed cycle
    of each test engine - test engines were cut off before failure, so their RUL can never
    be derived from max(cycle).
    """
    _check_args(subset)
    path = Path(data_dir) / f"RUL_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - see data/data_manifest.txt")
    rul = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype="int64")
    if (rul < 0).any():
        raise ValueError(f"{path.name}: negative RUL values")
    return rul


def compute_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'rul' column to TRAINING data: rul = (engine's last cycle) - (current cycle).

    Valid only for training engines, which ran until failure, so their last cycle IS the
    failure cycle (rul = 0 there). This is the raw, unclipped label; clipping at MAX_RUL
    happens in feature_engineering.clip_rul so both versions stay available.
    Returns a copy; the input is not modified.
    """
    df = train_df.copy()
    df["rul"] = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]
    last_rows = df.groupby("unit_id").tail(1)
    assert (last_rows["rul"] == 0).all(), "last cycle of every engine must have RUL 0"
    assert (df["rul"] >= 0).all(), "RUL must be non-negative"
    return df

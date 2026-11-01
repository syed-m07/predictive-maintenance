"""Evidence for the sensor drop decision (used by notebooks/01_EDA.ipynb and its test).

Lives in src/ rather than the notebook so the decision is reproducible and testable:
`screen_columns` recomputes the statistics, `suggest_drop_cols` applies the rule, and
tests/test_eda.py asserts the rule still yields config.DROP_COLS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RAW_FEATURE_COLS


def _within_engine_trend(df: pd.DataFrame, col: str) -> float:
    """Mean |Spearman correlation| between a column and cycle, computed INSIDE each engine.

    Why within-engine: engines start from different baseline levels (manufacturing
    variation). Pooling all engines can hide a clear per-engine degradation trend - s9
    and s14 in FD001 are the example. Constant series have no defined correlation -> 0.
    """
    def one(g: pd.DataFrame) -> float:
        if g[col].nunique() < 2:
            return 0.0
        return abs(g[col].corr(g["cycle"], method="spearman"))

    return float(df.groupby("unit_id")[[col, "cycle"]].apply(one).mean())


def screen_columns(train_df: pd.DataFrame, cols: list[str] = RAW_FEATURE_COLS) -> pd.DataFrame:
    """Per-column statistics on TRAINING data (must contain an 'rul' column).

    Columns returned:
      n_unique             distinct values (1 = perfectly constant)
      std                  standard deviation
      spearman_rul         rank correlation with RUL, all engines pooled
      within_engine_trend  see _within_engine_trend
    """
    rows = []
    for c in cols:
        x = train_df[c]
        constant = x.nunique() < 2
        rows.append({
            "column": c,
            "n_unique": int(x.nunique()),
            "std": float(x.std()),
            "spearman_rul": 0.0 if constant else float(x.corr(train_df["rul"], method="spearman")),
            "within_engine_trend": _within_engine_trend(train_df, c),
        })
    return pd.DataFrame(rows).set_index("column")


def suggest_drop_cols(screen: pd.DataFrame, max_unique: int = 2, min_trend: float = 0.2) -> list[str]:
    """Drop rule: a column is uninformative if it is (near-)constant OR shows no trend.

    - n_unique <= max_unique  -> constant or a 2-value flicker (FD001: op3, s1, s5, s6,
      s10, s16, s18, s19).
    - within_engine_trend < min_trend -> varies, but not with engine age (FD001: op1,
      op2 - pure noise because FD001 has a single operating condition).
    Thresholds sit far from any borderline column in FD001 (the weakest kept sensor has
    trend 0.63; the strongest dropped one, s6, 0.09), so the result is not threshold-sensitive.
    """
    mask = (screen["n_unique"] <= max_unique) | (screen["within_engine_trend"] < min_trend)
    return [c for c in screen.index if mask[c]]


def lifetimes(train_df: pd.DataFrame) -> pd.Series:
    """Total cycles each training engine ran before failure."""
    return train_df.groupby("unit_id")["cycle"].max().rename("lifetime")


def align_to_failure(train_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with 'cycles_to_failure' = -rul, so every engine ends at x = 0.

    Plotting sensors against this aligns all engines at their failure point, which makes
    the shared degradation pattern visible despite different lifetimes.
    """
    out = train_df.copy()
    out["cycles_to_failure"] = -out["rul"].astype(np.int64)
    return out

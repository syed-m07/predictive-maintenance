"""Live drift score for the API's /metrics (planner w9t7).

Every prediction request contributes its NEWEST reading (last row) to a rolling window. Only the
newest row, because clients typically resend an engine's whole history on every request: adding
all rows would count old cycles over and over. Once the window holds `min_rows` readings, it is
compared with the reference sample (FD001 training engines) using the same KS + PSI rule as
scripts/retrain_if_drift.py (src/monitoring.py). Exposed as gauges:

  drift_score      median PSI across the 14 sensors (0 = like training data; > 0.2 = large shift)
  drift_alarm      1 if >= 3 sensors drift (KS p < 0.05 AND PSI > 0.2), else 0
  drift_window_rows how many readings the current score is based on

The window mixes engines and life stages as they arrive, so read it as a fleet-level signal,
not a per-engine one (and see EXPERIMENTS E8: an alarm says the inputs changed, not that the
model is wrong).
"""
from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_FEATURE_COLS, SENSORS
from src.monitoring import compute_feature_drift, drift_summary, load_reference

_SENSOR_IDX = [RAW_FEATURE_COLS.index(s) for s in SENSORS]


class DriftMonitor:
    def __init__(self, reference: pd.DataFrame, window: int = 1000, min_rows: int = 200):
        missing = [s for s in SENSORS if s not in reference.columns]
        if missing:
            raise ValueError(f"reference lacks sensors {missing}")
        self.reference = reference[SENSORS]
        self.min_rows = min_rows
        self._rows: deque = deque(maxlen=window)
        self._lock = threading.Lock()
        self.last: dict | None = None

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> "DriftMonitor":
        ref, _ = load_reference(path)
        return cls(ref, **kw)

    def add(self, newest_row: list[float]) -> dict | None:
        """Add one raw 24-value reading; return the drift summary once enough rows exist."""
        with self._lock:
            self._rows.append([newest_row[i] for i in _SENSOR_IDX])
            if len(self._rows) < self.min_rows:
                return None
            cur = pd.DataFrame(np.asarray(self._rows), columns=SENSORS)
        summary = drift_summary(compute_feature_drift(self.reference, cur, SENSORS))
        summary["window_rows"] = len(cur)
        self.last = summary
        return summary

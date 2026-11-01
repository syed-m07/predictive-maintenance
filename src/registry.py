"""Model registry: one CSV row per trained model (trained_models/model_registry.csv, committed).

Weights are gitignored, but this file is versioned: it is the permanent record of which
model, trained from which git commit and which MLflow run, produced which numbers.
"""
from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import REGISTRY_PATH, ROOT_DIR

COLUMNS = [
    "model_name", "version", "subset", "path",
    "val_rmse", "test_rmse", "test_mae", "test_r2", "nasa_score",
    "test_rmse_clip125", "nasa_score_clip125",
    "latency_single_ms", "throughput_ms_per_item", "train_time_s",
    "mlflow_run_id", "git_commit", "created_at", "notes",
]


def git_commit() -> str:
    """Short hash of HEAD, suffixed '-dirty' if there are uncommitted changes to tracked files."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT_DIR,
                               capture_output=True, text=True, check=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_registry(path: Path | str = REGISTRY_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def next_version(model_name: str, path: Path | str = REGISTRY_PATH) -> int:
    reg = read_registry(path)
    existing = reg.loc[reg["model_name"] == model_name, "version"]
    return int(existing.max()) + 1 if len(existing) else 1


def append_entry(entry: dict, path: Path | str = REGISTRY_PATH) -> dict:
    """Append one row. Unknown keys are rejected (typos would silently create empty columns)."""
    unknown = set(entry) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown registry columns: {sorted(unknown)}")
    row = {c: entry.get(c, "") for c in COLUMNS}
    row["created_at"] = row["created_at"] or datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:                                   # migrate an older/shorter header
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            old = read_registry(path).reindex(columns=COLUMNS)
            old.to_csv(path, index=False)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return row


def get_latest(model_name: str, path: Path | str = REGISTRY_PATH) -> pd.Series | None:
    reg = read_registry(path)
    rows = reg[reg["model_name"] == model_name]
    return None if rows.empty else rows.sort_values("version").iloc[-1]


def get_best(metric: str = "test_rmse", subset: str = "FD001",
             path: Path | str = REGISTRY_PATH) -> pd.Series | None:
    """Row with the lowest value of `metric` (all registry metrics are lower-is-better
    except test_r2, which is maximised)."""
    reg = read_registry(path)
    reg = reg[(reg["subset"] == subset) & reg[metric].notna()]
    if reg.empty:
        return None
    return reg.loc[reg[metric].idxmax()] if metric == "test_r2" else reg.loc[reg[metric].idxmin()]

"""Central configuration: every path, column list and hyperparameter default lives here.

Why one module: training scripts, the API, the drift scripts and the tests must all agree
on things like which sensors are used and how long a window is. If those values were
copied into each file they would drift apart silently; importing them from one place
makes that impossible.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# Silence MLflow's informational "agent hint" banner. Set here because config is imported
# before mlflow everywhere.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# ---------------------------------------------------------------- paths
# Resolved from this file's location, so scripts work no matter which folder you run them from.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "trained_models"
RESULTS_DIR = ROOT_DIR / "results"
CONFIGS_DIR = ROOT_DIR / "configs"
REGISTRY_PATH = MODELS_DIR / "model_registry.csv"

# MLflow 3.x put the plain-folder ("./mlruns") tracking store into maintenance mode and
# refuses it by default, so run metadata goes into a local SQLite file instead.
# Artifacts (plots, JSON) still land in ./mlruns/ next to it. Both are gitignored.
# Overridable with the MLFLOW_TRACKING_URI environment variable - used on Colab to log
# into Google Drive so runs survive the session (see docs/COLAB.md).
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
)
# Where run artifacts (plots, metadata) are stored. Pinned to an absolute path: with a SQLite
# backend MLflow otherwise defaults to ./mlruns relative to the CURRENT directory.
MLFLOW_ARTIFACT_ROOT = Path(os.environ.get("MLFLOW_ARTIFACT_ROOT", ROOT_DIR / "mlruns"))

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- dataset
SUBSET = "FD001"            # primary benchmark
DRIFT_SUBSET = "FD002"      # used for the covariate-shift / retraining demo
SUBSETS = ("FD001", "FD002", "FD003", "FD004")

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS_ALL = [f"s{i}" for i in range(1, 22)]                 # 21 sensors
COLUMNS = ["unit_id", "cycle"] + OP_COLS + SENSOR_COLS_ALL        # 26 columns in every file
RAW_FEATURE_COLS = OP_COLS + SENSOR_COLS_ALL                      # 24 values per cycle (API input)

# FD001 drop list - CONFIRMED by EDA (notebooks/01_EDA.ipynb, docs/DECISIONS.md D6):
# 7 constant (op3 s1 s5 s10 s16 s18 s19), 1 near-constant (s6), 2 trendless noise (op1 op2).
# tests/test_eda.py fails if the data-driven rule in src/eda.py ever disagrees with it.
DROP_COLS = ["op1", "op2", "op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SENSORS = [c for c in SENSOR_COLS_ALL if c not in DROP_COLS]      # 14 informative sensors

# ---------------------------------------------------------------- labels
# Piecewise-linear RUL: early-life cycles show no degradation signal, so every true RUL
# above 125 is treated as 125 for TRAINING labels. Test ground truth is NOT clipped
# (we report against NASA's RUL_FD00X.txt as given - see docs/DECISIONS.md).
MAX_RUL = 125

# ---------------------------------------------------------------- features
ROLL_WINDOWS = [5, 10, 30]  # trailing rolling mean/std/min/max
LAGS = [1, 5, 10]           # value k cycles ago
EMA_ALPHA = 0.3             # exponential moving average smoothing
SEQ_LEN = 30                # window length for TCN / Bi-LSTM / ROCKET

# ---------------------------------------------------------------- split
VAL_FRACTION = 0.2          # fraction of TRAIN ENGINES (not rows) held out for validation


def seed_everything(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch so runs are repeatable.

    torch is imported lazily: the slim API Docker image has no torch, and importing
    config there must not fail.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

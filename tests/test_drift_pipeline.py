"""Drift data helpers and the retrain_if_drift decision paths (no retraining here: that path
takes ~1 min and is exercised by the smoke run documented in EXPERIMENTS E8)."""
import os
import subprocess
import sys

import numpy as np
import pytest

from src.config import ROOT_DIR, SENSORS
from src.drift_data import (UNIT_OFFSET, current_batch, prepare_combined, reference_frame,
                            training_split, truncate_like_test)
from src.feature_engineering import prepare_data


def test_reference_is_exactly_the_project_training_engines():
    d = prepare_data("FD001")
    ref = reference_frame("FD001")
    assert sorted(ref["unit_id"].unique()) == d.train_units
    assert sorted(current_batch("FD001-val")["unit_id"].unique()) == d.val_units
    assert ref[SENSORS].max().max() > 1.5                    # raw values, not min-max scaled
    with pytest.raises(ValueError):
        current_batch("FD003-test")


def test_truncation_leaves_remaining_life_from_the_test_truth():
    val = training_split("FD001")[1]
    truth = np.array([10, 50, 90])
    cut = truncate_like_test(val, truth, np.random.default_rng(0))
    for u, g in cut.groupby("unit_id"):
        removed = len(val[val["unit_id"] == u]) - len(g)
        assert removed in truth and (g["cycle"].diff().dropna() == 1).all()


def test_combined_data_keeps_each_split_and_offsets_ids():
    c = prepare_combined(("FD001", "FD002"))
    d = prepare_data("FD001")
    fd001_val = [u for u in c.val_units if u < UNIT_OFFSET]
    assert fd001_val == d.val_units                          # FD001 keeps the project's validation engines
    assert not set(c.train_units) & set(c.val_units)
    assert max(c.train_units) > UNIT_OFFSET                   # FD002 engines present, shifted
    tr = c.train[SENSORS].to_numpy()
    assert tr.min() >= -1e-9 and tr.max() <= 1 + 1e-9         # scaler fit on pooled training engines
    assert set(c.tests) == {"FD001", "FD002"} and len(c.tests["FD002"][1]) == 259


def _run_script(tmp_path, *args):
    env = {**os.environ, "PM_MODELS_DIR": str(tmp_path / "models"), "PM_RESULTS_DIR": str(tmp_path / "results"),
           "MLFLOW_TRACKING_URI": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
           "MLFLOW_ARTIFACT_ROOT": str(tmp_path / "mlruns")}
    code = ("import json, sys; from scripts.retrain_if_drift import run; "
            f"out = run(*{list(args)!r}); print('ACTION=' + out['action'])")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


def test_control_batch_does_nothing(tmp_path):
    assert _run_script(tmp_path, "FD001-val") == "ACTION=none"
    assert (tmp_path / "models" / "reference_FD001.npz").exists()          # rebuilt when missing
    assert not (tmp_path / "models" / "model_registry.csv").exists()       # nothing registered


def test_same_subset_alarm_is_flagged_not_retrained(tmp_path):
    assert _run_script(tmp_path, "FD001-test") == "ACTION=flagged_for_review"
    assert not (tmp_path / "models" / "model_registry.csv").exists()


def test_dry_run_never_retrains(tmp_path):
    assert _run_script(tmp_path, "FD002-test", True) == "ACTION=dry_run"
    assert not (tmp_path / "models" / "model_registry.csv").exists()

"""The comparison script's own safety checks (scripts/compare_models.py)."""
import numpy as np
import pandas as pd
import pytest

from scripts.compare_models import check_protocol, latest_rows, tuning_trials
from src.feature_engineering import prepare_data, save_scaler
from src.models.base import SCALER_FILE


@pytest.fixture(scope="module")
def d():
    return prepare_data("FD001")


def bundle(tmp_path, d, scaler=None):
    b = tmp_path / "b"
    b.mkdir()
    save_scaler(scaler or d.scaler, b / SCALER_FILE)
    return b


def test_protocol_accepts_the_shared_split_and_scaler(tmp_path, d):
    meta = {"subset": "FD001", "data_split": {"train_units": d.train_units, "val_units": d.val_units}}
    check_protocol("ok", meta, bundle(tmp_path, d), d)


def test_protocol_rejects_other_validation_engines(tmp_path, d):
    meta = {"subset": "FD001", "data_split": {"train_units": d.train_units, "val_units": d.val_units[:-1] + [999]}}
    with pytest.raises(AssertionError, match="different train/validation engines"):
        check_protocol("bad", meta, bundle(tmp_path, d), d)


def test_protocol_rejects_another_scaler(tmp_path, d):
    import copy
    other = copy.deepcopy(d.scaler)
    other.data_max_ = other.data_max_ + 1.0
    meta = {"subset": "FD001", "data_split": {"train_units": d.train_units, "val_units": d.val_units}}
    with pytest.raises(AssertionError, match="scaler differs"):
        check_protocol("bad", meta, bundle(tmp_path, d, other), d)


def test_protocol_rejects_other_subset(tmp_path, d):
    with pytest.raises(AssertionError, match="trained on FD002"):
        check_protocol("bad", {"subset": "FD002", "data_split": {}}, bundle(tmp_path, d), d)


def test_latest_rows_keeps_newest_version_of_subset():
    reg = pd.DataFrame({"model_name": ["a", "a", "b", "c"], "version": [1, 2, 1, 1],
                        "subset": ["FD001", "FD001", "FD001", "FD001+FD002"]})
    out = latest_rows(reg, "FD001")
    assert sorted(zip(out.model_name, out.version)) == [("a", 2), ("b", 1)]   # c is another subset


def test_tuning_trials_parsed_from_notes():
    assert tuning_trials("optuna 50 trials, best #7") == 50
    assert tuning_trials(np.nan) == 0 and tuning_trials("") == 0

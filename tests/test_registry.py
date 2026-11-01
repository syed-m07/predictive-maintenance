import pandas as pd
import pytest

from src.registry import COLUMNS, append_entry, get_best, get_latest, next_version, read_registry


def test_append_versions_and_lookup(tmp_path):
    p = tmp_path / "reg.csv"
    assert read_registry(p).empty and next_version("xgb", p) == 1
    append_entry({"model_name": "xgb", "version": 1, "subset": "FD001", "test_rmse": 20.0, "test_r2": 0.7}, p)
    append_entry({"model_name": "xgb", "version": 2, "subset": "FD001", "test_rmse": 17.0, "test_r2": 0.8}, p)
    append_entry({"model_name": "tcn", "version": 1, "subset": "FD001", "test_rmse": 15.0, "test_r2": 0.9}, p)
    assert next_version("xgb", p) == 3 and next_version("tcn", p) == 2
    assert get_latest("xgb", p)["version"] == 2
    assert get_best("test_rmse", path=p)["model_name"] == "tcn"
    assert get_best("test_r2", path=p)["model_name"] == "tcn"          # r2: higher is better
    reg = read_registry(p)
    assert list(reg.columns) == COLUMNS and len(reg) == 3 and reg["created_at"].notna().all()


def test_rejects_unknown_columns(tmp_path):
    with pytest.raises(KeyError):
        append_entry({"model_name": "x", "tset_rmse": 1.0}, tmp_path / "reg.csv")


def test_migrates_old_header(tmp_path):
    p = tmp_path / "reg.csv"
    p.write_text("model_name,version,subset\nold,1,FD001\n")
    append_entry({"model_name": "new", "version": 1, "subset": "FD001"}, p)
    reg = read_registry(p)
    assert list(reg.columns) == COLUMNS and reg["model_name"].tolist() == ["old", "new"]

"""Tests for src/data_loader.py.

Expected numbers come from data/data_manifest.txt (verified against the raw files).
Tests that need the raw data are skipped automatically if data/raw/ is empty (e.g. CI).
"""
import numpy as np
import pandas as pd
import pytest

from src.config import COLUMNS, DATA_RAW_DIR
from src.data_loader import compute_rul, load_cmapss, load_rul_truth

needs_data = pytest.mark.skipif(
    not (DATA_RAW_DIR / "train_FD001.txt").exists(), reason="raw C-MAPSS data not downloaded"
)

# (subset, split) -> (rows, engines)   from data/data_manifest.txt
EXPECTED = {
    ("FD001", "train"): (20631, 100), ("FD001", "test"): (13096, 100),
    ("FD002", "train"): (53759, 260), ("FD002", "test"): (33991, 259),
    ("FD003", "train"): (24720, 100), ("FD003", "test"): (16596, 100),
    ("FD004", "train"): (61249, 249), ("FD004", "test"): (41214, 248),
}


@needs_data
@pytest.mark.parametrize("subset,split", list(EXPECTED))
def test_shapes_and_engine_counts(subset, split):
    df = load_cmapss(subset, split)
    rows, engines = EXPECTED[(subset, split)]
    assert df.shape == (rows, 26)
    assert df["unit_id"].nunique() == engines
    assert list(df.columns) == COLUMNS
    assert not df.isna().any().any()


@needs_data
@pytest.mark.parametrize("subset", ["FD001", "FD002", "FD003", "FD004"])
def test_rul_truth_has_one_value_per_test_engine(subset):
    truth = load_rul_truth(subset)
    assert len(truth) == load_cmapss(subset, "test")["unit_id"].nunique()
    assert (truth >= 0).all()


@needs_data
def test_compute_rul_on_fd001():
    df = compute_rul(load_cmapss("FD001", "train"))
    assert (df["rul"] >= 0).all()
    assert (df.groupby("unit_id").tail(1)["rul"] == 0).all()        # failure cycle
    first = df.groupby("unit_id").head(1)
    lifetimes = df.groupby("unit_id")["cycle"].max().to_numpy()
    assert (first["rul"].to_numpy() == lifetimes - 1).all()         # cycle 1 -> life - 1
    assert df["rul"].max() == 361                                   # longest engine: 362 cycles


def test_compute_rul_toy_example_and_no_mutation():
    toy = pd.DataFrame({"unit_id": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    out = compute_rul(toy)
    assert out["rul"].tolist() == [2, 1, 0, 1, 0]
    assert "rul" not in toy.columns                                  # input untouched


def _write(tmp_path, name, rows):
    """Write a fake C-MAPSS file with 26 numbers per line (+ trailing spaces like NASA's)."""
    lines = [" ".join(str(v) for v in r) + "  " for r in rows]
    (tmp_path / name).write_text("\n".join(lines) + "\n")


def _row(unit, cycle):
    return [unit, cycle] + [0.5] * 24


def test_trailing_spaces_parse_to_26_columns(tmp_path):
    _write(tmp_path, "train_FD001.txt", [_row(1, 1), _row(1, 2)])
    assert load_cmapss("FD001", "train", data_dir=tmp_path).shape == (2, 26)


def test_rejects_missing_cycle(tmp_path):
    _write(tmp_path, "train_FD001.txt", [_row(1, 1), _row(1, 3)])   # cycle 2 missing
    with pytest.raises(ValueError, match="cycles"):
        load_cmapss("FD001", "train", data_dir=tmp_path)


def test_rejects_unsorted_units(tmp_path):
    _write(tmp_path, "train_FD001.txt", [_row(2, 1), _row(1, 1)])
    with pytest.raises(ValueError, match="sorted"):
        load_cmapss("FD001", "train", data_dir=tmp_path)


def test_bad_arguments():
    with pytest.raises(ValueError):
        load_cmapss("FD009", "train")
    with pytest.raises(ValueError):
        load_cmapss("FD001", "validation")
    with pytest.raises(FileNotFoundError):
        load_cmapss("FD001", "train", data_dir="does/not/exist")

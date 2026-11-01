"""The sensor drop list in config must remain what the data says (src/eda.py rule)."""
import pandas as pd
import pytest

from src.config import DATA_RAW_DIR, DROP_COLS, SENSORS
from src.data_loader import compute_rul, load_cmapss
from src.eda import screen_columns, suggest_drop_cols

needs_data = pytest.mark.skipif(
    not (DATA_RAW_DIR / "train_FD001.txt").exists(), reason="raw C-MAPSS data not downloaded"
)


@needs_data
def test_drop_rule_reproduces_config_drop_list_on_fd001():
    screen = screen_columns(compute_rul(load_cmapss("FD001", "train")))
    assert sorted(suggest_drop_cols(screen)) == sorted(DROP_COLS)
    assert len(SENSORS) == 14


@needs_data
def test_s9_s14_hidden_trend_is_why_within_engine_metric_is_used():
    screen = screen_columns(compute_rul(load_cmapss("FD001", "train")), cols=["s9", "s14"])
    # weak when engines are pooled, strong inside each engine
    assert (screen["spearman_rul"].abs() < 0.4).all()
    assert (screen["within_engine_trend"] > 0.6).all()


def test_rule_on_toy_data():
    toy = pd.DataFrame({
        "unit_id": [1] * 4 + [2] * 4,
        "cycle": [1, 2, 3, 4] * 2,
        "const": [5.0] * 8,                       # constant -> drop
        "trend": [1, 2, 3, 4, 2, 3, 4, 5],         # rises with age -> keep
        "noise": [1, 2, 2, 1, 3, 4, 4, 3],         # symmetric: exactly zero trend -> drop
    })
    toy["rul"] = toy.groupby("unit_id")["cycle"].transform("max") - toy["cycle"]
    screen = screen_columns(toy, cols=["const", "trend", "noise"])
    assert suggest_drop_cols(screen) == ["const", "noise"]

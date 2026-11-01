"""Tests for src/evaluation.py - every number here is computed by hand."""
import math

import numpy as np
import pytest

from src.evaluation import (clip_predictions, evaluate_test, evaluate_validation, mae,
                            nasa_score, nasa_score_per_engine, r2, regression_metrics, rmse,
                            time_inference)


# ------------------------------------------------------------------ NASA score
def test_nasa_perfect_prediction_scores_zero():
    assert nasa_score([10], [10]) == 0.0
    assert nasa_score([5, 50, 120], [5, 50, 120]) == 0.0


def test_nasa_exact_values_late_and_early():
    assert nasa_score([10], [20]) == pytest.approx(math.e ** 1 - 1)            # +10 late  ~1.718
    assert nasa_score([20], [10]) == pytest.approx(math.e ** (10 / 13) - 1)    # -10 early ~1.159


def test_nasa_is_asymmetric_late_costs_more():
    for err in (1, 5, 10, 30, 60):
        late = nasa_score([100], [100 + err])
        early = nasa_score([100], [100 - err])
        assert late > early > 0, err


def test_nasa_grows_with_error_and_sums_over_engines():
    per = nasa_score_per_engine([0, 0, 0], [5, 10, 20])
    assert per[0] < per[1] < per[2]
    assert nasa_score([0, 0, 0], [5, 10, 20]) == pytest.approx(per.sum())


# ------------------------------------------------------------------ RMSE / MAE / R2
def test_rmse_mae_by_hand():
    t, p = [0, 0, 0, 0], [3, -3, 4, 0]
    assert rmse(t, p) == pytest.approx(math.sqrt((9 + 9 + 16 + 0) / 4))
    assert mae(t, p) == pytest.approx((3 + 3 + 4 + 0) / 4)


def test_r2_reference_points():
    t = [1.0, 2.0, 3.0, 4.0]
    assert r2(t, t) == pytest.approx(1.0)
    assert r2(t, [2.5] * 4) == pytest.approx(0.0)                               # predicting the mean
    assert r2(t, [4.0, 3.0, 2.0, 1.0]) < 0                                      # worse than the mean
    assert math.isnan(r2([5, 5], [5, 5]))


def test_input_validation():
    with pytest.raises(ValueError):
        rmse([1, 2], [1])
    with pytest.raises(ValueError):
        mae([], [])
    with pytest.raises(ValueError):
        nasa_score([1, np.nan], [1, 2])


# ------------------------------------------------------------------ evaluation conventions
def test_predictions_are_clipped_to_valid_range():
    np.testing.assert_array_equal(clip_predictions([-5, 60, 200]), [0, 60, 125])
    # a negative prediction is scored as 0, not as -5
    assert evaluate_validation([0], [-5])["rmse"] == 0.0


def test_evaluate_test_reports_primary_and_literature_comparable():
    """Truth 140 (> 125), prediction 125: primary error 15, clip125 error 0 (P6)."""
    m = evaluate_test([140], [125])
    assert m["rmse"] == pytest.approx(15.0) and m["mae"] == pytest.approx(15.0)
    assert m["nasa"] == pytest.approx(math.exp(15 / 13) - 1)                    # 15 early
    assert m["rmse_clip125"] == 0.0 and m["nasa_clip125"] == 0.0
    assert set(m) == {"rmse", "mae", "r2", "nasa", "rmse_clip125", "mae_clip125", "r2_clip125", "nasa_clip125"}


def test_evaluate_test_identical_when_truth_below_cap():
    truth, pred = [10, 50, 100], [12, 45, 110]
    m = evaluate_test(truth, pred)
    for k in ("rmse", "mae", "r2", "nasa"):
        assert m[k] == pytest.approx(m[f"{k}_clip125"])
    assert m == pytest.approx({**regression_metrics(truth, pred),
                               **{f"{k}_clip125": v for k, v in regression_metrics(truth, pred).items()}})


# ------------------------------------------------------------------ timing
def test_time_inference_reports_both_latencies():
    calls = []
    def predict(x):
        calls.append(len(x))
        return np.zeros(len(x))
    X = np.zeros((20, 30, 14))
    t = time_inference(predict, X, n_runs=3, n_warmup=2)
    assert t["latency_single_ms"] >= 0 and t["throughput_ms_per_item"] >= 0
    assert calls.count(1) == 5 and calls.count(20) == 5                         # warm-up + timed runs
    with pytest.raises(ValueError):
        time_inference(predict, X[:0])

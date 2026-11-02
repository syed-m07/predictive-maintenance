import numpy as np
import pytest

from src.comparison import (bootstrap_indices, bootstrap_scores, confidence_intervals,
                            paired_differences)
from src.evaluation import evaluate_test


def toy(n=100, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(5, 140, n)
    good = truth + rng.normal(0, 8, n)
    bad = truth + rng.normal(0, 25, n)
    return truth, {"good": good, "good_twin": good + rng.normal(0, 0.5, n), "bad": bad}


def test_indices_are_seeded_and_in_range():
    a, b = bootstrap_indices(50, 200, seed=1), bootstrap_indices(50, 200, seed=1)
    assert a.shape == (200, 50) and a.min() >= 0 and a.max() < 50
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, bootstrap_indices(50, 200, seed=2))
    with pytest.raises(ValueError):
        bootstrap_indices(1)


def test_identity_resample_equals_official_scorer():
    """Scoring on the un-resampled engines must reproduce src.evaluation exactly (incl. clipping)."""
    truth, preds = toy()
    preds["over_cap"] = np.full(100, 300.0)          # clipped to 125 by both paths
    s = bootstrap_scores(truth, preds, np.arange(100)[None, :])
    for name, p in preds.items():
        e = evaluate_test(truth, p)
        assert s[name]["rmse"][0] == pytest.approx(e["rmse"])
        assert s[name]["nasa"][0] == pytest.approx(e["nasa"])


def test_interval_contains_point_estimate_and_orders_models():
    truth, preds = toy()
    ci = confidence_intervals(truth, preds, n_boot=2000)
    for name, p in preds.items():
        e = evaluate_test(truth, p)
        assert ci.loc[name, "rmse_ci_lo"] < e["rmse"] < ci.loc[name, "rmse_ci_hi"]
        assert ci.loc[name, "nasa_ci_lo"] < e["nasa"] < ci.loc[name, "nasa_ci_hi"]
    assert ci.loc["good", "rmse_ci_hi"] < ci.loc["bad", "rmse_ci_lo"]


def test_paired_detects_real_gap_and_reports_tie():
    truth, preds = toy()
    p = paired_differences(truth, preds, reference="good", n_boot=2000)
    assert "good" not in p.index
    assert p.loc["bad", "rmse_distinguishable"] and p.loc["bad", "rmse_diff_ci_lo"] > 0
    assert p.loc["bad", "rmse_p_ref_better"] > 0.99
    assert not p.loc["good_twin", "rmse_distinguishable"]          # near-identical model = tie
    assert p.loc["good_twin", "rmse_diff_ci_lo"] < 0 < p.loc["good_twin", "rmse_diff_ci_hi"]
    e_bad, e_good = evaluate_test(truth, preds["bad"]), evaluate_test(truth, preds["good"])
    assert p.loc["bad", "rmse_diff"] == pytest.approx(e_bad["rmse"] - e_good["rmse"])


def test_paired_is_sharper_than_separate_intervals():
    """Why pairing matters: two models whose separate intervals overlap can still be told apart."""
    rng = np.random.default_rng(3)
    truth = rng.uniform(5, 140, 100)
    shared = rng.normal(0, 15, 100)                  # both models miss the same engines
    a, b = truth + shared, truth + shared * 1.15
    ci = confidence_intervals(truth, {"a": a, "b": b}, n_boot=2000)
    assert ci.loc["a", "rmse_ci_hi"] > ci.loc["b", "rmse_ci_lo"]       # separate intervals overlap
    p = paired_differences(truth, {"a": a, "b": b}, reference="a", n_boot=2000)
    assert p.loc["b", "rmse_distinguishable"]                          # paired test separates them


def test_errors():
    truth, preds = toy()
    with pytest.raises(KeyError):
        paired_differences(truth, preds, reference="missing", n_boot=10)
    with pytest.raises(ValueError):
        confidence_intervals(truth, {"short": np.zeros(99)}, n_boot=10)

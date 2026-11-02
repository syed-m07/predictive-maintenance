import numpy as np
import pandas as pd
import pytest

from src.explain import (compare_rankings, gradient_shap, group_by_sensor_sequence,
                         group_by_sensor_tabular, sensor_importance, tabular_sensor_of, tree_shap)

SENS = ["s2", "s3", "s11"]


def test_tabular_grouping_sums_each_sensors_features():
    names = ["s2", "s2_rollmean_5", "s3_lag_1", "s11_ema", "s11_rollstd_10"]
    v = np.array([[1.0, 2.0, -3.0, 4.0, -1.0],
                  [0.5, -0.5, 1.0, 0.0, 2.0]])
    g = group_by_sensor_tabular(v, names, SENS)
    np.testing.assert_allclose(g, [[3.0, -3.0, 3.0], [0.0, 1.0, 2.0]])
    np.testing.assert_allclose(g.sum(axis=1), v.sum(axis=1))      # additivity is preserved
    assert tabular_sensor_of("s11_rollstd_10") == "s11" and tabular_sensor_of("s4") == "s4"
    with pytest.raises(ValueError):
        group_by_sensor_tabular(v, names[:-1] + ["s99_ema"], SENS)


def test_signed_sum_before_abs():
    """Opposite contributions of one sensor's features cancel (they are one sensor's net effect)."""
    g = group_by_sensor_tabular(np.array([[5.0, -5.0, 1.0]]), ["s2", "s2_ema", "s3"], ["s2", "s3"])
    np.testing.assert_allclose(g, [[0.0, 1.0]])


def test_sequence_grouping_and_importance_shares():
    v = np.zeros((4, 30, 3))
    v[:, -1, 2] = 3.0          # s11 matters at the last step
    v[:, 0, 0] = -1.0          # s2 matters (negatively) at the first step
    imp = sensor_importance(group_by_sensor_sequence(v), SENS)
    assert imp.sum() == pytest.approx(1.0)
    assert imp["s11"] == pytest.approx(0.75) and imp["s2"] == pytest.approx(0.25) and imp["s3"] == 0
    with pytest.raises(ValueError):
        group_by_sensor_sequence(v[:, :, 0])
    with pytest.raises(ValueError):
        sensor_importance(np.ones((4, 2)), SENS)


def test_compare_rankings():
    a = pd.Series([0.5, 0.3, 0.2], index=SENS)
    out = compare_rankings({"a": a, "same": a.copy(), "rev": a[::-1].set_axis(SENS)}, top=2)
    same = out[(out.model_a == "a") & (out.model_b == "same")].iloc[0]
    rev = out[(out.model_a == "a") & (out.model_b == "rev")].iloc[0]
    assert same.spearman_rho == pytest.approx(1.0) and same.top2_overlap == 2
    assert rev.spearman_rho == pytest.approx(-1.0) and rev.top2_overlap == 1   # s3 is 2nd both ways


def test_tree_shap_is_exactly_additive():
    from src.models.xgboost_rul import XGBoostRUL
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["s2", "s3_ema", "s11"])
    y = 60 + 25 * X["s11"] - 10 * X["s2"]
    m = XGBoostRUL(params={"n_estimators": 50}).fit(X[:250], y[:250], X[250:], y[250:])
    exp = tree_shap(m, X[["s11", "s3_ema", "s2"]])                 # columns re-ordered by name
    np.testing.assert_allclose(exp.values.sum(axis=1) + exp.base_values, m.predict(X), atol=1e-3)
    imp = sensor_importance(group_by_sensor_tabular(exp.values, m.feature_names, SENS), SENS)
    assert imp.idxmax() == "s11" and imp.idxmin() == "s3"         # the unused feature ranks last


def test_gradient_shap_shape_and_finds_the_used_channel():
    from src.models.lstm_rul import LSTMRUL
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 10, 3)).astype(np.float32)
    y = 60 + 30 * X[:, -3:, 2].mean(axis=1)                         # only channel 2, recent steps
    m = LSTMRUL(arch={"hidden_size": 16, "bidirectional": False},
                train_cfg={"epochs": 15, "batch_size": 64, "lr": 1e-2, "seed": 0}, device="cpu")
    m.fit(X[:320], y[:320], X[320:], y[320:], verbose=False)
    vals = gradient_shap(m, X[:50], X[320:340], nsamples=100)
    assert vals.shape == (20, 10, 3)
    imp = sensor_importance(group_by_sensor_sequence(vals), SENS)
    assert imp.idxmax() == "s11"
    per_step = np.abs(vals[:, :, 2]).mean(axis=0)
    assert per_step[-3:].mean() > per_step[:5].mean()              # recent steps matter more

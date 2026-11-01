import numpy as np
import pandas as pd
import pytest

from src.models.base import read_metadata
from src.models.xgboost_rul import XGBoostRUL


def data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = 60 + 10 * X["a"] - 5 * X["b"] + rng.normal(size=n)      # stays inside [0, 125]
    return X, y


def test_fit_predict_learns_signal_and_early_stops():
    X, y = data()
    m = XGBoostRUL(params={"n_estimators": 500, "learning_rate": 0.1}, early_stopping_rounds=20)
    m.fit(X[:300], y[:300], X[300:], y[300:])
    assert m.best_iteration is not None and m.best_iteration < 499          # stopped early
    assert y.min() > 0 and y.max() < 125
    assert m.evaluate(X[300:], y[300:])["rmse"] < 0.4 * y.std()                # far better than the mean


def test_save_load_roundtrip_identical(tmp_path):
    X, y = data()
    m = XGBoostRUL(params={"n_estimators": 50}).fit(X[:300], y[:300], X[300:], y[300:])
    m.save(tmp_path / "bundle", metadata={"note": "x"})
    m2 = XGBoostRUL.load(tmp_path / "bundle")
    np.testing.assert_array_equal(m.predict(X), m2.predict(X))
    meta = read_metadata(tmp_path / "bundle")
    assert meta["feature_names"] == ["a", "b", "c", "d"] and meta["note"] == "x"


def test_column_order_is_enforced_by_name():
    X, y = data()
    m = XGBoostRUL(params={"n_estimators": 30}).fit(X, y)
    shuffled = X[["d", "c", "b", "a"]]
    np.testing.assert_array_equal(m.predict(X), m.predict(shuffled))       # re-ordered by name
    with pytest.raises(ValueError):
        m.predict(X.drop(columns="a"))
    with pytest.raises(ValueError):
        m.predict(np.zeros((3, 5)))


def test_load_rejects_wrong_model_type(tmp_path):
    X, y = data()
    m = XGBoostRUL(params={"n_estimators": 10}).fit(X, y)
    m.save(tmp_path / "b")
    import json
    meta = json.loads((tmp_path / "b" / "metadata.json").read_text())
    meta["model_type"] = "tcn"
    (tmp_path / "b" / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        XGBoostRUL.load(tmp_path / "b")

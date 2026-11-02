import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.models.rocket_rul import ROCKETRUL, ridge_path


def windows(n=300, seq_len=30, ch=3, seed=0):
    """Toy windows whose target depends on the trend of channel 0 inside the window."""
    rng = np.random.default_rng(seed)
    slope = rng.uniform(-1, 1, n)
    t = np.linspace(0, 1, seq_len)
    X = rng.normal(scale=0.1, size=(n, seq_len, ch))
    X[:, :, 0] += slope[:, None] * t[None, :]
    y = 60 + 40 * slope                                      # stays inside [0, 125]
    return X.astype(np.float32), y


def test_ridge_path_matches_sklearn():
    rng = np.random.default_rng(1)
    F = rng.normal(size=(200, 30)); F = (F - F.mean(0)) / F.std(0)
    y = F @ rng.normal(size=30) + 5 + rng.normal(size=200)
    coefs, intercept = ridge_path(F, y, [0.1, 10.0])
    for a, c in zip([0.1, 10.0], coefs):
        sk = Ridge(alpha=a).fit(F, y)
        np.testing.assert_allclose(c, sk.coef_, rtol=1e-6, atol=1e-8)
        assert intercept == pytest.approx(sk.intercept_)


def test_learns_signal_and_picks_alpha_on_validation():
    X, y = windows()
    m = ROCKETRUL(num_kernels=500).fit(X[:220], y[:220], X[220:], y[220:])
    assert set(m.alpha_scores) == set(m.alphas)
    assert m.alpha == min(m.alpha_scores, key=m.alpha_scores.get)
    assert m.evaluate(X[220:], y[220:])["rmse"] < 0.4 * y.std()
    assert m.timings["transform_s"] > 0 and m.timings["ridge_s"] > 0


def test_save_load_roundtrip_identical(tmp_path):
    X, y = windows()
    m = ROCKETRUL(num_kernels=500).fit(X[:220], y[:220], X[220:], y[220:])
    m.save(tmp_path / "b", metadata={"note": "x"})
    m2 = ROCKETRUL.load(tmp_path / "b")
    np.testing.assert_array_equal(m.predict(X), m2.predict(X))
    meta = json.loads((tmp_path / "b" / "metadata.json").read_text())
    assert meta["alpha"] == m.alpha and meta["note"] == "x" and meta["input_kind"] == "sequence"


def test_input_shape_is_checked_and_wrong_bundle_rejected(tmp_path):
    X, y = windows()
    m = ROCKETRUL(num_kernels=500).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :, :2])                               # wrong channel count
    with pytest.raises(ValueError):
        m.predict(X[:, 0, :])                                # 2-D input
    m.save(tmp_path / "b")
    meta = json.loads((tmp_path / "b" / "metadata.json").read_text())
    meta["model_type"] = "tcn"
    (tmp_path / "b" / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        ROCKETRUL.load(tmp_path / "b")

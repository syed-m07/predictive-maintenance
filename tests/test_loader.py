import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT_DIR
from src.models.loader import MODEL_CLASSES, load_bundle, model_class
from src.models.xgboost_rul import XGBoostRUL


def xgb_bundle(path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = 50 + 20 * X["a"]
    m = XGBoostRUL(params={"n_estimators": 30}).fit(X[:150], y[:150], X[150:], y[150:])
    m.save(path)
    return m, X


def test_every_registered_type_resolves_to_its_class():
    for t in MODEL_CLASSES:
        assert model_class(t).model_type == t
    with pytest.raises(ValueError):
        model_class("svm")


def test_load_bundle_dispatches_on_metadata(tmp_path):
    m, X = xgb_bundle(tmp_path / "b")
    loaded, meta = load_bundle(tmp_path / "b")
    assert isinstance(loaded, XGBoostRUL) and meta["model_type"] == "xgboost"
    np.testing.assert_allclose(loaded.predict(X), m.predict(X), rtol=1e-6)
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing")


def test_loading_xgboost_never_imports_torch(tmp_path):
    """The slim API image has no torch: loading an XGBoost bundle must not need it."""
    xgb_bundle(tmp_path / "b")
    code = ("import sys; from src.models.loader import load_bundle; "
            f"load_bundle(r'{tmp_path / 'b'}'); "
            "bad = [m for m in ('torch', 'sktime') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, check=True)

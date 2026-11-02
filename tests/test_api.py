"""API tests. Part 1 uses tiny models built on the fly (always runs, no trained weights needed).
Part 2 uses the real served bundles and is skipped when they are absent (weights are gitignored)."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.drift_monitor import DriftMonitor
from src.api.main import create_app
from src.api.model_loader import load_config, readings_frame
from src.config import RAW_FEATURE_COLS, ROOT_DIR, SENSORS
from src.feature_engineering import fit_scaler, save_scaler, tabular_feature_names
from src.models.base import SCALER_FILE
from src.monitoring import save_reference
from src.registry import append_entry

rng = np.random.default_rng(0)


def raw_rows(n, shift=0.0):
    """Plausible raw FD001-like rows: 24 values per cycle."""
    base = np.array([0.0, 0.0, 100.0, 518.67, 642.0, 1590.0, 1405.0, 14.62, 21.61, 554.0, 2388.0,
                     9060.0, 1.3, 47.5, 522.0, 2388.0, 8140.0, 8.44, 0.03, 392.0, 2388.0, 100.0, 38.8, 23.3])
    return (base + shift + rng.normal(0, 0.1, (n, 24))).round(4).tolist()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from src.models.lstm_rul import LSTMRUL
    from src.models.xgboost_rul import XGBoostRUL
    tmp = tmp_path_factory.mktemp("serve")
    names = tabular_feature_names()
    train = pd.DataFrame(rng.normal(size=(300, len(SENSORS))), columns=SENSORS)
    scaler = fit_scaler(train)
    reg = tmp / "registry.csv"

    xgb = XGBoostRUL(params={"n_estimators": 5})
    X = pd.DataFrame(rng.normal(size=(200, len(names))), columns=names)
    xgb.fit(X, rng.uniform(0, 125, 200))
    lstm = LSTMRUL(arch={"hidden_size": 8, "bidirectional": False},
                   train_cfg={"epochs": 1, "batch_size": 32, "seed": 0}, device="cpu")
    Xs = rng.normal(size=(64, 30, len(SENSORS))).astype(np.float32)
    lstm.fit(Xs, rng.uniform(0, 125, 64), Xs[:16], rng.uniform(0, 125, 16), verbose=False)
    for name, m in [("tiny_xgb", xgb), ("tiny_lstm", lstm)]:
        bundle = tmp / f"{name}_v1"
        m.save(bundle, metadata={"model_name": name, "version": 1, "subset": "FD001", "seq_len": 30})
        save_scaler(scaler, bundle / SCALER_FILE)
        append_entry({"model_name": name, "version": 1, "subset": "FD001", "path": str(bundle),
                      "test_rmse": 99.0}, reg)
    ref = pd.DataFrame(np.asarray(raw_rows(500)), columns=RAW_FEATURE_COLS)
    save_reference(ref, SENSORS, tmp / "ref.npz")
    cfg = tmp / "serving.json"
    cfg.write_text(json.dumps({"models": ["tiny_xgb", "tiny_lstm"], "default": "tiny_lstm",
                               "registry": str(reg), "reference": str(tmp / "ref.npz"),
                               "drift_window": 50, "drift_min_rows": 10}))
    with TestClient(create_app(cfg)) as c:
        yield c


def metric(c, line_start):
    for line in c.get("/metrics").text.splitlines():
        if line.startswith(line_start):
            return float(line.split()[-1])
    return None


def test_health_and_models(client):
    h = client.get("/health").json()
    assert h == {"status": "ok", "model_loaded": True, "models": ["tiny_lstm", "tiny_xgb"], "default_model": "tiny_lstm"}
    ms = {m["model_name"]: m for m in client.get("/models").json()}
    assert ms["tiny_lstm"]["is_default"] and not ms["tiny_xgb"]["is_default"]
    assert ms["tiny_xgb"]["input_kind"] == "tabular" and ms["tiny_lstm"]["input_kind"] == "sequence"
    assert ms["tiny_xgb"]["test_rmse"] == 99.0


@pytest.mark.parametrize("model", [None, "tiny_xgb", "tiny_lstm"])
def test_predict_returns_clipped_rul(client, model):
    url = "/predict" + (f"?model={model}" if model else "")
    r = client.post(url, json={"unit_id": "e1", "readings": raw_rows(40)})
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0.0 <= b["predicted_rul"] <= 125.0
    assert b["model_name"] == (model or "tiny_lstm") and b["model_version"] == "1"
    assert b["unit_id"] == "e1" and b["input_cycles"] == 40 and b["padded"] is False and b["latency_ms"] > 0


def test_short_history_is_padded_not_rejected(client):
    b = client.post("/predict?model=tiny_lstm", json={"readings": raw_rows(5)}).json()
    assert b["padded"] is True and b["input_cycles"] == 5


@pytest.mark.parametrize("readings, expect", [
    ([[1.0] * 23], "has 23 values"),
    ([], "at least 1"),
    ([[1.0] * 24, [1.0] * 25], "row 1 has 25"),
])
def test_malformed_input_is_422(client, readings, expect):
    r = client.post("/predict", json={"readings": readings})
    assert r.status_code == 422 and expect in json.dumps(r.json())


def test_nan_is_rejected(client):
    body = '{"readings": [[' + ", ".join(["NaN"] + ["1.0"] * 23) + "]]}"
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_unknown_model_is_404(client):
    r = client.post("/predict?model=rocket", json={"readings": raw_rows(30)})
    assert r.status_code == 404 and "tiny_xgb" in r.json()["detail"]
    assert client.get("/metadata?model=rocket").status_code == 404


def test_compare_equals_individual_predictions(client):
    rows = raw_rows(35)
    comp = {p["model_name"]: p["predicted_rul"] for p in
            client.post("/predict/compare", json={"readings": rows}).json()["predictions"]}
    assert set(comp) == {"tiny_xgb", "tiny_lstm"}
    for name, v in comp.items():
        assert client.post(f"/predict?model={name}", json={"readings": rows}).json()["predicted_rul"] == pytest.approx(v)


def test_metadata(client):
    m = client.get("/metadata").json()
    assert m["model_name"] == "tiny_lstm" and m["bundle"]["model_type"] == "lstm"
    assert "history" not in m["bundle"]


def test_inference_error_is_500_and_counted(client):
    served = client.app.state.models["tiny_xgb"]
    original = served.model.predict
    served.model.predict = lambda X: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        before = metric(client, 'prediction_errors_total{model="tiny_xgb"}') or 0
        r = client.post("/predict?model=tiny_xgb", json={"readings": raw_rows(30)})
        assert r.status_code == 500 and "boom" in r.json()["detail"]
        assert metric(client, 'prediction_errors_total{model="tiny_xgb"}') == before + 1
    finally:
        served.model.predict = original


def test_metrics_count_predictions_and_drift_turns_on(client):
    before = metric(client, 'prediction_count_total{model="tiny_xgb"}') or 0
    for _ in range(12):                                   # > drift_min_rows=10 newest readings
        client.post("/predict?model=tiny_xgb", json={"readings": raw_rows(30)})
    assert metric(client, 'prediction_count_total{model="tiny_xgb"}') == before + 12
    assert metric(client, "drift_score") >= 0             # -1 until enough readings
    assert metric(client, "drift_window_rows") >= 10
    assert metric(client, "drift_alarm") == 0             # same distribution as the reference


def test_drift_monitor_alarms_on_shifted_readings():
    ref = pd.DataFrame(np.asarray(raw_rows(2000)), columns=RAW_FEATURE_COLS)
    mon = DriftMonitor(ref, window=300, min_rows=200)
    assert mon.add(raw_rows(1)[0]) is None                # not enough rows yet
    for row in raw_rows(300):
        s = mon.add(row)
    assert s is not None and not s["alarm"]
    for row in raw_rows(300, shift=0.5):
        s = mon.add(row)
    assert s["alarm"] and s["n_drift"] >= 3 and s["window_rows"] == 300


def test_config_validation(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"models": ["a"], "default": "b"}))
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text(json.dumps({"models": [], "default": None}))
    with pytest.raises(ValueError):
        load_config(p)


def test_readings_frame_layout():
    df = readings_frame(raw_rows(3))
    assert list(df.columns[:2]) == ["unit_id", "cycle"] and df["cycle"].tolist() == [1, 2, 3]
    assert list(df.columns[2:]) == RAW_FEATURE_COLS


# --------------------------------------------------------------------------- real bundles
REAL_CFG = ROOT_DIR / "configs" / "serving.json"


def _real_available() -> bool:
    try:
        from src.api.model_loader import load_served_models
        load_served_models(load_config(REAL_CFG))
        return (ROOT_DIR / "results" / "test_predictions_all_models.csv").exists()
    except Exception:
        return False


@pytest.mark.skipif(not _real_available(), reason="served model bundles not present (weights are gitignored)")
def test_api_reproduces_official_test_predictions():
    """End to end: raw FD001 test readings through the API == the comparison script's predictions
    (results/test_predictions_all_models.csv) for every served model, all 100 engines."""
    from src.data_loader import load_cmapss
    expected = pd.read_csv(ROOT_DIR / "results" / "test_predictions_all_models.csv").set_index("unit_id")
    test = load_cmapss("FD001", "test")
    with TestClient(create_app(REAL_CFG)) as c:
        for unit, eng in test.groupby("unit_id"):
            rows = eng.sort_values("cycle")[RAW_FEATURE_COLS].values.tolist()
            preds = c.post("/predict/compare", json={"readings": rows}).json()["predictions"]
            for p in preds:
                assert p["predicted_rul"] == pytest.approx(expected.loc[unit, p["model_name"]], abs=2e-3), \
                    (unit, p["model_name"])

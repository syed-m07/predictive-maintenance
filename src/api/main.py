"""RUL prediction API (FastAPI). Run locally:

    uvicorn src.api.main:app --port 8000          # then open http://localhost:8000/docs

Endpoints
  GET  /health                 liveness + which models are loaded
  GET  /models                 the served shortlist (configs/serving.json) with registry metrics
  GET  /metadata?model=<name>  one model's registry row + bundle metadata (default model if omitted)
  POST /predict?model=<name>   one engine's raw readings -> predicted RUL (default model if omitted)
  POST /predict/compare        the same readings through EVERY served model
  GET  /metrics                Prometheus: request counts, latency, live drift score (per model)

Models, scalers and the drift reference are loaded ONCE at startup (lifespan), never per
request. Status codes: 404 unknown model, 422 malformed input (pydantic), 500 inference error.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.api.drift_monitor import DriftMonitor
from src.api.model_loader import ServedModel, _abs, load_config, load_served_models, readings_frame, warm_up
from src.api.schemas import (CompareResponse, HealthResponse, ModelInfo, PredictionResponse,
                             SensorWindow)
from src.config import SEQ_LEN

log = logging.getLogger("rul_api")

# Prometheus metrics: module-level, so they are registered once per process.
PREDICTIONS = Counter("prediction_count", "Predictions served", ["model"])
ERRORS = Counter("prediction_errors", "Failed predictions", ["model"])
LATENCY = Histogram("prediction_latency_seconds", "Preprocessing + prediction time", ["model"],
                    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
DRIFT_SCORE = Gauge("drift_score", "Median PSI of recent readings vs the training reference (-1 = not enough data yet)")
DRIFT_ALARM = Gauge("drift_alarm", "1 if >= 3 sensors drift (KS p<0.05 AND PSI>0.2)")
DRIFT_ROWS = Gauge("drift_window_rows", "Readings in the drift window")


def create_app(config_path=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = load_config(config_path)
        app.state.cfg = cfg
        app.state.models = load_served_models(cfg)
        app.state.default = cfg["default"]
        warm_up(app.state.models)
        ref = cfg.get("reference")
        app.state.drift = None
        if ref and _abs(ref).exists():
            app.state.drift = DriftMonitor.from_file(_abs(ref), window=cfg.get("drift_window", 1000),
                                                     min_rows=cfg.get("drift_min_rows", 200))
        else:
            log.warning("no drift reference at %s: drift metrics disabled", ref)
        DRIFT_SCORE.set(-1)
        log.info("serving %s (default %s)", list(app.state.models), app.state.default)
        yield
        app.state.models = {}

    app = FastAPI(title="Turbofan RUL API",
                  description="Remaining-useful-life predictions for NASA C-MAPSS engines (FD001-trained models).",
                  version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        """422 without echoing the input back: a rejected NaN cannot be written as JSON, and a
        5,000-row request should not come back in the error message."""
        errors = [{k: v for k, v in e.items() if k in ("type", "loc", "msg")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    def get_model(request: Request, name: str | None) -> ServedModel:
        models = request.app.state.models
        name = name or request.app.state.default
        if name not in models:
            raise HTTPException(404, f"unknown model {name!r}; served models: {sorted(models)}")
        return models[name]

    def predict_one(m: ServedModel, window: SensorWindow, raw) -> PredictionResponse:
        t0 = time.perf_counter()
        try:
            rul = m.predict_raw(raw)
        except Exception as e:                                   # noqa: BLE001 - reported to the client
            ERRORS.labels(m.name).inc()
            log.exception("inference failed for %s", m.name)
            raise HTTPException(500, f"inference failed for {m.name}: {type(e).__name__}: {e}") from e
        dt = time.perf_counter() - t0
        PREDICTIONS.labels(m.name).inc()
        LATENCY.labels(m.name).observe(dt)
        return PredictionResponse(predicted_rul=rul, model_name=m.name, model_version=m.version,
                                  timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  unit_id=window.unit_id, input_cycles=len(window.readings),
                                  padded=len(window.readings) < SEQ_LEN, latency_ms=dt * 1e3)

    def track_drift(request: Request, window: SensorWindow) -> None:
        monitor = request.app.state.drift
        if monitor is None:
            return
        s = monitor.add(window.readings[-1])
        DRIFT_ROWS.set(len(monitor._rows))
        if s is not None:
            DRIFT_SCORE.set(s["median_psi"])
            DRIFT_ALARM.set(int(s["alarm"]))

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        models = getattr(request.app.state, "models", {})
        return HealthResponse(status="ok", model_loaded=bool(models), models=sorted(models),
                              default_model=getattr(request.app.state, "default", None))

    @app.get("/models", response_model=list[ModelInfo])
    def models(request: Request):
        out = []
        for name, m in request.app.state.models.items():
            r = m.registry_row
            out.append(ModelInfo(model_name=name, model_version=m.version, model_type=m.meta["model_type"],
                                 input_kind=m.input_kind, is_default=name == request.app.state.default,
                                 trained_on=str(m.meta.get("subset", r.get("subset"))),
                                 test_rmse=r.get("test_rmse"), test_rmse_clip125=r.get("test_rmse_clip125"),
                                 nasa_score=r.get("nasa_score"), git_commit=r.get("git_commit"),
                                 created_at=r.get("created_at")))
        return out

    @app.get("/metadata")
    def metadata(request: Request, model: str | None = Query(None, description="model name; default if omitted")):
        m = get_model(request, model)
        meta = {k: v for k, v in m.meta.items() if k not in ("history", "data_split", "feature_names")}
        return {"model_name": m.name, "model_version": m.version, "registry": m.registry_row, "bundle": meta}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(window: SensorWindow, request: Request,
                model: str | None = Query(None, description="model name; default if omitted")):
        m = get_model(request, model)
        out = predict_one(m, window, readings_frame(window.readings))
        track_drift(request, window)
        return out

    @app.post("/predict/compare", response_model=CompareResponse)
    def predict_compare(window: SensorWindow, request: Request):
        raw = readings_frame(window.readings)
        preds = [predict_one(m, window, raw) for m in request.app.state.models.values()]
        track_drift(request, window)
        return CompareResponse(unit_id=window.unit_id, predictions=preds)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()

"""The shared end of every training run: score -> save bundle -> log to MLflow -> registry.

Every model goes through exactly this function, so all of them are scored, saved and recorded
identically (the comparison table depends on it).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.config import MAX_RUL, MODELS_DIR, RESULTS_DIR, SEED
from src.evaluation import evaluate_test
from src.feature_engineering import save_scaler
from src.models.base import SCALER_FILE, RULModel
from src.registry import append_entry, git_commit, next_version


def finish_run(*, model: RULModel, model_name: str, subset: str, scaler, data_split: dict,
               val_metrics: dict, test_truth, test_pred, timing: dict, train_time_s: float,
               figures: dict[str, plt.Figure] | None = None, notes: str = "",
               extra_metadata: dict | None = None) -> dict:
    """Must be called inside an active `mlflow.start_run()`. Returns the registry row."""
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("finish_run must be called inside mlflow.start_run()")
    test_metrics = evaluate_test(test_truth, test_pred, MAX_RUL)
    commit = git_commit()
    version = next_version(model_name)
    bundle = MODELS_DIR / f"{model_name}_v{version}"

    # ---- bundle: model + its scaler + metadata (everything the API needs, nothing else)
    model.save(bundle, metadata={
        "model_name": model_name, "version": version, "subset": subset, "seed": SEED,
        "max_rul": MAX_RUL, "data_split": data_split, "val_metrics": val_metrics,
        "test_metrics": test_metrics, "timing": timing, "train_time_s": train_time_s,
        "git_commit": commit, "mlflow_run_id": run.info.run_id, **(extra_metadata or {})})
    save_scaler(scaler, bundle / SCALER_FILE)

    # ---- MLflow
    mlflow.set_tags({"model_name": model_name, "version": version, "subset": subset,
                     "git_commit": commit, "notes": notes})
    mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    mlflow.log_metrics({**timing, "train_time_s": train_time_s})
    mlflow.log_artifact(str(bundle / "metadata.json"))
    preds = {"test_truth": np.asarray(test_truth).tolist(), "test_pred": np.asarray(test_pred).tolist()}
    mlflow.log_text(json.dumps(preds), "test_predictions.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in (figures or {}).items():
        path = RESULTS_DIR / f"{model_name}_{name}.png"
        fig.savefig(path)
        mlflow.log_artifact(str(path))
        plt.close(fig)

    # ---- registry (committed CSV)
    row = append_entry({
        "model_name": model_name, "version": version, "subset": subset,
        "path": bundle.relative_to(MODELS_DIR.parent).as_posix(),
        "val_rmse": round(val_metrics["rmse"], 4),
        "test_rmse": round(test_metrics["rmse"], 4), "test_mae": round(test_metrics["mae"], 4),
        "test_r2": round(test_metrics["r2"], 4), "nasa_score": round(test_metrics["nasa"], 2),
        "test_rmse_clip125": round(test_metrics["rmse_clip125"], 4),
        "nasa_score_clip125": round(test_metrics["nasa_clip125"], 2),
        "latency_single_ms": round(timing["latency_single_ms"], 4),
        "throughput_ms_per_item": round(timing["throughput_ms_per_item"], 5),
        "train_time_s": round(train_time_s, 2),
        "mlflow_run_id": run.info.run_id, "git_commit": commit, "notes": notes})
    return {**row, "test_metrics": test_metrics}

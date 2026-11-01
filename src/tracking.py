"""MLflow setup shared by all training scripts.

An *experiment* groups related runs (e.g. "rul-xgboost"); a *run* is one training job with
its params (inputs), metrics (results) and artifacts (files: plots, metadata). The UI
(`mlflow ui --backend-store-uri sqlite:///mlflow.db`) lets you compare runs side by side.
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")   # silence an informational banner

import mlflow  # noqa: E402

from src.config import MLFLOW_ARTIFACT_ROOT, MLFLOW_TRACKING_URI  # noqa: E402


def setup_mlflow(experiment: str) -> str:
    """Point MLflow at the project store and select (or create) an experiment.

    New experiments get an explicit artifact folder under MLFLOW_ARTIFACT_ROOT so artifacts
    never land in a stray ./mlruns of whatever directory a script was launched from.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        location = (MLFLOW_ARTIFACT_ROOT / experiment).resolve().as_uri()
        exp_id = mlflow.create_experiment(experiment, artifact_location=location)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment)
    return exp_id

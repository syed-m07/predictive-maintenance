"""Train, evaluate and register an XGBoost RUL model.

    python -m src.train_xgboost                                   # baseline (HANDOFF config)
    python -m src.train_xgboost --params-json configs/xgboost_best_params.json \
                                --model-name xgboost_tuned        # after Optuna (Block 6)

Steps: prepare_data (leakage-safe split + scaler) -> 238 tabular features -> fit with early
stopping on the 20 validation engines -> validation metrics -> ONE prediction per test engine
(its last row) scored against RUL_FD00X.txt -> CPU timing -> bundle + MLflow + registry.
"""
from __future__ import annotations

import argparse
import json
import time

import mlflow
import pandas as pd

from src.config import SEED, SUBSET, seed_everything
from src.evaluation import time_inference
from src.feature_engineering import (build_tabular_features, last_row_per_engine, prepare_data,
                                     tabular_feature_names)
from src.models.xgboost_rul import EARLY_STOPPING_ROUNDS, XGBoostRUL
from src.plotting import (apply_style, plot_feature_importance, plot_life_trajectories,
                          plot_test_predictions)
from src.registry import git_commit
from src.run_utils import finish_run
from src.tracking import setup_mlflow

EXPERIMENT = "rul-xgboost"


def build_xgb_inputs(subset: str = SUBSET):
    """Tabular train/val/test inputs. Returns (data, feats_train, feats_val, test_last_rows, names)."""
    d = prepare_data(subset)
    names = tabular_feature_names()
    f_train = build_tabular_features(d.train)
    f_val = build_tabular_features(d.val)
    f_test_last = last_row_per_engine(build_tabular_features(d.test))
    assert f_test_last["unit_id"].tolist() == sorted(d.test["unit_id"].unique()), "test order != truth order"
    return d, f_train, f_val, f_test_last, names


def run(subset: str = SUBSET, params: dict | None = None, model_name: str = "xgboost_baseline",
        notes: str = "") -> dict:
    commit = git_commit()          # the code version that actually runs (recorded in the registry)
    seed_everything(SEED)
    apply_style()
    d, f_train, f_val, f_test, names = build_xgb_inputs(subset)

    setup_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=model_name):
        model = XGBoostRUL(params=params)
        t0 = time.perf_counter()
        model.fit(f_train[names], f_train["rul"], f_val[names], f_val["rul"])
        train_time = time.perf_counter() - t0

        mlflow.log_params({**model.params, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                           "best_iteration": model.best_iteration, "n_features": len(names),
                           "subset": subset, "seed": SEED, "n_train_rows": len(f_train),
                           "n_val_rows": len(f_val), "n_test_engines": len(f_test)})

        val_metrics = model.evaluate(f_val[names], f_val["rul"])
        test_pred = model.predict(f_test[names])
        timing = time_inference(model.predict, f_test[names].to_numpy(dtype="float32"))

        val_traj = f_val[["unit_id", "cycle", "rul"]].assign(pred=model.predict(f_val[names]))
        figures = {
            "test_predictions": plot_test_predictions(d.test_truth, test_pred, f"{model_name}: test set ({subset}), one point per engine"),
            "val_trajectories": plot_life_trajectories(val_traj, f"{model_name}: predicted vs target RUL over validation engines' lives"),
            "feature_importance": plot_feature_importance(model.feature_importance(), f"{model_name}: top 20 features by gain"),
        }
        row = finish_run(
            model=model, model_name=model_name, subset=subset, scaler=d.scaler,
            data_split={"train_units": d.train_units, "val_units": d.val_units},
            val_metrics=val_metrics, test_truth=d.test_truth, test_pred=test_pred,
            timing=timing, train_time_s=train_time, figures=figures, notes=notes, commit=commit)

    tm = row["test_metrics"]
    print(f"\n{model_name} v{row['version']}  ({subset})  best_iteration={model.best_iteration}")
    print(f"  validation  RMSE {val_metrics['rmse']:.2f}  MAE {val_metrics['mae']:.2f}")
    print(f"  TEST        RMSE {tm['rmse']:.2f}  MAE {tm['mae']:.2f}  R2 {tm['r2']:.3f}  NASA {tm['nasa']:.1f}")
    print(f"  TEST clip125 RMSE {tm['rmse_clip125']:.2f}  NASA {tm['nasa_clip125']:.1f}   (literature-comparable)")
    print(f"  latency {timing['latency_single_ms']:.3f} ms/engine single, {timing['throughput_ms_per_item']:.4f} ms/engine batch; train {train_time:.1f}s")
    print(f"  bundle: {row['path']}   mlflow run: {row['mlflow_run_id']}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--params-json", help="JSON file with XGBoost params (e.g. Optuna's best)")
    ap.add_argument("--model-name", default="xgboost_baseline")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    params = json.loads(open(a.params_json).read()) if a.params_json else None
    run(a.subset, params, a.model_name, a.notes)


if __name__ == "__main__":
    main()

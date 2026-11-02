"""Block 10 (planner w7t6-w7t7): compare every registered model on the FD001 test set.

    python -m scripts.compare_models            # from the repo root

For every model in trained_models/model_registry.csv (latest version of each name):
  1. load the bundle and CHECK it follows the shared protocol: same validation engines and the
     same training-engine scaler as prepare_data() rebuilds (otherwise the comparison is void);
  2. re-score it on the 100 test engines and check the score matches the registry
     (catches a stale or mismatched bundle);
  3. re-time it on THIS machine's CPU (decision V4: Colab models were timed on Colab's CPU);
then add the no-skill mean predictor (decision V6), compute 95 % bootstrap intervals and
paired differences against the best model (owner-approved, 2026-11-02), and write:

  results/model_comparison.csv          the table (README / paper)
  results/model_comparison_paired.csv   every model vs the best, paired bootstrap
  results/test_predictions_all_models.csv  per-engine predictions of every model
  results/model_comparison_ci.png       scores with intervals
  results/model_comparison_paired.png   differences vs the best, with intervals
  results/multi_model_predictions.png   all models over 3 test engines' histories
"""
from __future__ import annotations

import os
import platform
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.comparison import (confidence_intervals, engine_trajectory_inputs, paired_differences,
                            pick_trajectory_engines, predict_kind, test_inputs)
from src.config import RESULTS_DIR, SEED, SEQ_LEN, SUBSET, seed_everything
from src.evaluation import clip_predictions, evaluate_test, time_inference
from src.feature_engineering import load_scaler, prepare_data
from src.models.base import SCALER_FILE
from src.models.loader import load_bundle, resolve_bundle_path
from src.plotting import (apply_style, model_family, plot_metric_intervals,
                          plot_multi_model_trajectories, plot_paired_differences, save_fig)
from src.registry import read_registry

MEAN_PREDICTOR = "mean_predictor (no skill)"
# Where each family was trained (docs/EXPERIMENTS.md E1-E5). Training times are NOT comparable
# across these two kinds of hardware; the table says which is which.
TRAIN_HARDWARE = {"xgboost": "local CPU", "rocket": "local CPU", "tcn": "Colab T4 GPU", "lstm": "Colab T4 GPU"}


def latest_rows(registry: pd.DataFrame, subset: str) -> pd.DataFrame:
    reg = registry[registry["subset"] == subset]
    return reg.sort_values("version").groupby("model_name", sort=False).tail(1).reset_index(drop=True)


def check_protocol(name: str, meta: dict, bundle, d) -> None:
    """Fail loudly if a bundle was trained on a different split or scaler than the others."""
    if meta.get("subset", SUBSET) != d.subset:
        raise AssertionError(f"{name}: trained on {meta.get('subset')}, comparing on {d.subset}")
    split = meta.get("data_split", {})
    if split.get("val_units") != d.val_units or split.get("train_units") != d.train_units:
        raise AssertionError(f"{name}: different train/validation engines than the shared protocol")
    sc = load_scaler(resolve_bundle_path(bundle) / SCALER_FILE)
    if not (np.allclose(sc.data_min_, d.scaler.data_min_) and np.allclose(sc.data_max_, d.scaler.data_max_)):
        raise AssertionError(f"{name}: bundle scaler differs from the training-engine scaler")


def model_size(model, meta: dict) -> tuple[float, float]:
    """(n_params, n_trees). Torch: trainable weights; ROCKET: ridge weights; XGBoost: trees used."""
    t = meta["model_type"]
    if t in ("tcn", "lstm"):
        return float(sum(p.numel() for p in model.net.parameters())), np.nan
    if t == "rocket":
        return float(model.coef.size + 1), np.nan
    if t == "xgboost":
        return np.nan, float(model.best_iteration + 1)
    return np.nan, np.nan


def tuning_trials(notes) -> float:
    m = re.search(r"optuna (\d+) trials", str(notes))
    return float(m.group(1)) if m else 0.0


def run(subset: str = SUBSET, results_dir=RESULTS_DIR) -> pd.DataFrame:
    seed_everything(SEED)
    apply_style()
    d = prepare_data(subset)
    inputs = test_inputs(d, SEQ_LEN)
    truth = d.test_truth
    rows, preds, models = [], {}, {}

    reg = latest_rows(read_registry(), subset)
    print(f"comparing {len(reg)} models on {subset} test ({len(truth)} engines)")
    for r in reg.itertuples():
        model, meta = load_bundle(r.path)
        check_protocol(r.model_name, meta, r.path, d)
        pred = predict_kind(model, meta, inputs)
        m = evaluate_test(truth, pred)
        # registry stores RMSE to 4 decimals and NASA to 2
        if abs(m["rmse"] - r.test_rmse) > 1e-3 or abs(m["nasa"] - r.nasa_score) > 0.011:
            raise AssertionError(f"{r.model_name}: re-scored RMSE {m['rmse']:.4f} / NASA {m['nasa']:.2f} "
                                 f"!= registry {r.test_rmse} / {r.nasa_score}")
        X = inputs[meta["input_kind"]]
        X = X[model.feature_names].to_numpy(dtype=np.float32) if meta["input_kind"] == "tabular" else X
        timing = time_inference(model.predict, X)
        n_params, n_trees = model_size(model, meta)
        preds[r.model_name] = pred
        models[r.model_name] = (model, meta)
        rows.append({"model": r.model_name, "family": model_family(r.model_name),
                     "tuning_trials": tuning_trials(r.notes),
                     "test_rmse": m["rmse"], "test_mae": m["mae"], "test_r2": m["r2"], "nasa": m["nasa"],
                     "test_rmse_clip125": m["rmse_clip125"], "nasa_clip125": m["nasa_clip125"],
                     "latency_single_ms": timing["latency_single_ms"],
                     "throughput_ms_per_item": timing["throughput_ms_per_item"],
                     "n_params": n_params, "n_trees": n_trees,
                     "train_time_s": r.train_time_s, "train_hardware": TRAIN_HARDWARE[meta["model_type"]]})
        print(f"  {r.model_name:<18} RMSE {m['rmse']:6.2f}  NASA {m['nasa']:8.1f}  "
              f"latency {timing['latency_single_ms']:7.3f} ms  (re-score matches registry)")

    # no-skill reference: always predict the mean clipped training label (decision V6)
    mean_label = float(d.train["rul"].mean())
    preds[MEAN_PREDICTOR] = np.full(len(truth), mean_label)
    m = evaluate_test(truth, preds[MEAN_PREDICTOR])
    rows.append({"model": MEAN_PREDICTOR, "family": "reference", "tuning_trials": 0.0,
                 "test_rmse": m["rmse"], "test_mae": m["mae"], "test_r2": m["r2"], "nasa": m["nasa"],
                 "test_rmse_clip125": m["rmse_clip125"], "nasa_clip125": m["nasa_clip125"]})

    table = pd.DataFrame(rows).set_index("model")
    best = table.drop(MEAN_PREDICTOR)["test_rmse"].idxmin()
    ci = confidence_intervals(truth, preds)
    paired = paired_differences(truth, preds, reference=best)
    table = table.join(ci).join(paired[["rmse_diff", "rmse_diff_ci_lo", "rmse_diff_ci_hi", "rmse_distinguishable",
                                        "nasa_diff", "nasa_diff_ci_lo", "nasa_diff_ci_hi", "nasa_distinguishable"]]
                                .add_suffix("_vs_best"))
    table = table.sort_values("test_rmse")
    cols = ["family", "tuning_trials", "test_rmse", "rmse_ci_lo", "rmse_ci_hi", "test_mae", "test_r2",
            "nasa", "nasa_ci_lo", "nasa_ci_hi", "test_rmse_clip125", "nasa_clip125",
            "rmse_diff_vs_best", "rmse_diff_ci_lo_vs_best", "rmse_diff_ci_hi_vs_best", "rmse_distinguishable_vs_best",
            "nasa_diff_vs_best", "nasa_diff_ci_lo_vs_best", "nasa_diff_ci_hi_vs_best", "nasa_distinguishable_vs_best",
            "latency_single_ms", "throughput_ms_per_item", "n_params", "n_trees", "train_time_s", "train_hardware"]
    table = table[cols]
    table.attrs["best"] = best

    # ---- files
    results_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(results_dir / "model_comparison.csv")
    paired.round(4).to_csv(results_dir / "model_comparison_paired.csv")
    pd.DataFrame({"unit_id": inputs["unit_id"], "true_rul": truth,
                  **{k: clip_predictions(v) for k, v in preds.items()}}).round(3) \
        .to_csv(results_dir / "test_predictions_all_models.csv", index=False)

    cpu = f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs"
    real = table.drop(MEAN_PREDICTOR)
    figs = {
        "model_comparison_ci": plot_metric_intervals(
            real, best, f"FD001 test set: scores with 95 % bootstrap intervals (10,000 resamples of the 100 engines)"),
        "model_comparison_paired": plot_paired_differences(
            paired.drop(MEAN_PREDICTOR), best, f"Paired bootstrap: each model vs {best} (95 % intervals)"),
    }
    trajectories = {}
    for u in pick_trajectory_engines(d):
        tin = engine_trajectory_inputs(d, u, SEQ_LEN)
        trajectories[u] = {"cycle": tin["cycle"], "truth": tin["truth"],
                           "preds": {n: predict_kind(mo, me, tin) for n, (mo, me) in models.items()}}
        # the last point of a trajectory IS the official test input: must equal the test prediction
        i = inputs["unit_id"].tolist().index(u)
        for n in models:
            if not np.isclose(trajectories[u]["preds"][n][-1], preds[n][i], atol=1e-3):
                raise AssertionError(f"{n}: trajectory end != test prediction for engine {u}")
    figs["multi_model_predictions"] = plot_multi_model_trajectories(
        trajectories, "All models on three FD001 test engines: predicted vs true RUL over each engine's history")
    for name, fig in figs.items():
        save_fig(fig, name, results_dir)
        plt.close(fig)

    print(f"\nbest by test RMSE: {best}.  Timing hardware: {cpu}")
    print(table[["test_rmse", "rmse_ci_lo", "rmse_ci_hi", "nasa", "rmse_diff_ci_lo_vs_best",
                 "rmse_diff_ci_hi_vs_best", "rmse_distinguishable_vs_best", "latency_single_ms"]].round(2).to_string())
    print(f"\nwrote {results_dir / 'model_comparison.csv'} (+ paired csv, predictions csv, 3 figures)")
    return table


if __name__ == "__main__":
    run()

"""Block 12 (planner w9t5): does the drift detector fire when it should, and only then?

    python -m scripts.simulate_drift            # from the repo root, ~1 min

Reference = raw sensor readings of the FD001 training engines (what every FD001 model learned
from); saved to trained_models/reference_FD001.npz for scripts/retrain_if_drift.py.
Current batches (src/drift_data.py):
  FD001-val    no-drift control: same population -> must NOT alarm (DV-004)
  FD001-test   same population, engines cut off before failure -> shifted health-state mix
  FD002-test   six operating conditions -> covariate shift
Plus a diagnostic: FD001-val engines cut off the way the test set was (20 seeded draws). If
the FD001-test alarm disappears, the cut-off explains it.

Then the FD001 models (xgboost_tuned, lstm_uni_tuned) are scored on FD001 test and FD002 test,
both through the FD001 scaler stored in their bundles - "what happens if we just deploy it".

Writes results/drift_report.csv, drift_summary.csv, drift_model_impact.csv,
drift_psi_by_sensor.png, drift_distributions.png; logs one run to MLflow 'drift-retraining'.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.comparison import predict_kind, test_inputs
from src.config import MODELS_DIR, RESULTS_DIR, SEED, SENSORS, SEQ_LEN, seed_everything
from src.data_loader import load_rul_truth
from src.drift_data import BATCHES, current_batch, reference_frame, truncate_like_test
from src.evaluation import evaluate_test
from src.feature_engineering import apply_scaler, load_scaler
from src.models.base import SCALER_FILE
from src.models.loader import load_bundle, resolve_bundle_path
from src.monitoring import compute_feature_drift, drift_summary, save_reference
from src.plotting import apply_style, plot_drift_distributions, plot_psi_by_sensor, save_fig
from src.registry import get_latest
from src.tracking import setup_mlflow

EXPERIMENT = "drift-retraining"
REFERENCE_PATH = MODELS_DIR / "reference_FD001.npz"
IMPACT_MODELS = ["xgboost_tuned", "lstm_uni_tuned"]
N_TRUNCATION_DRAWS = 20
PLOT_SENSORS = ["s4", "s11", "s9", "s3"]        # top SHAP sensors (E7)


def model_impact(models: list[str]) -> pd.DataFrame:
    """Score FD001-trained models on FD001 test and FD002 test, through their own FD001 scaler."""
    rows = []
    for name in models:
        row = get_latest(name)
        model, meta = load_bundle(row["path"])
        scaler = load_scaler(resolve_bundle_path(row["path"]) / SCALER_FILE)
        for subset in ("FD001", "FD002"):
            test = apply_scaler(current_batch(f"{subset}-test"), scaler)
            truth = load_rul_truth(subset)
            inputs = test_inputs(SimpleNamespace(test=test), SEQ_LEN)
            m = evaluate_test(truth, predict_kind(model, meta, inputs))
            scaled = test[SENSORS].to_numpy()
            rows.append({"model": name, "test_set": subset, "n_engines": len(truth),
                         "test_rmse": m["rmse"], "test_mae": m["mae"], "test_r2": m["r2"],
                         "nasa": m["nasa"], "nasa_per_engine": m["nasa"] / len(truth),
                         "share_inputs_outside_0_1": float(((scaled < 0) | (scaled > 1)).mean())})
    return pd.DataFrame(rows)


def run(results_dir=RESULTS_DIR) -> dict:
    seed_everything(SEED)
    apply_style()
    ref = reference_frame("FD001")
    save_reference(ref, SENSORS, REFERENCE_PATH,
                   meta={"subset": "FD001", "source": "training engines (project split)",
                         "units": sorted(ref["unit_id"].unique().tolist()), "n_rows": len(ref)})
    batches = {b: current_batch(b) for b in BATCHES}

    reports, summaries = [], []
    for b, cur in batches.items():
        rep = compute_feature_drift(ref, cur, SENSORS)
        reports.append(rep.reset_index().assign(batch=b))
        summaries.append({"batch": b, "n_ref_rows": len(ref), "n_cur_rows": len(cur),
                          "n_engines": cur["unit_id"].nunique(), **drift_summary(rep)})

    # diagnostic: cut the control engines off like the test set
    # Two questions, same seeded cut-offs: (a) does cutting the control engines off REPRODUCE the
    # shift vs the reference? (b) once cut off, do they MATCH the FD001 test engines?
    truth001 = load_rul_truth("FD001")
    rng = np.random.default_rng(SEED)
    cut = [truncate_like_test(batches["FD001-val"], truth001, rng) for _ in range(N_TRUNCATION_DRAWS)]
    for label, base in [("vs reference", ref), ("vs FD001-test", batches["FD001-test"])]:
        diag = [drift_summary(compute_feature_drift(base, c, SENSORS)) for c in cut]
        summaries.append({"batch": f"FD001-val cut off like test, {label} ({N_TRUNCATION_DRAWS} draws)",
                          "n_ref_rows": len(base), "n_engines": batches["FD001-val"]["unit_id"].nunique(),
                          "n_ks_drift": float(np.mean([d["n_ks_drift"] for d in diag])),
                          "n_psi_drift": float(np.mean([d["n_psi_drift"] for d in diag])),
                          "n_drift": float(np.mean([d["n_drift"] for d in diag])),
                          "median_psi": float(np.median([d["median_psi"] for d in diag])),
                          "max_psi": float(np.max([d["max_psi"] for d in diag])),
                          "alarm": f"{sum(d['alarm'] for d in diag)}/{N_TRUNCATION_DRAWS}"})

    report = pd.concat(reports, ignore_index=True)[["batch", "feature", "ks_stat", "p_value", "psi",
                                                     "psi_level", "ks_drift", "psi_drift", "drift"]]
    summary = pd.DataFrame(summaries)
    impact = model_impact(IMPACT_MODELS)

    results_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(results_dir / "drift_report.csv", index=False, float_format="%.6g")
    summary.to_csv(results_dir / "drift_summary.csv", index=False, float_format="%.4f")
    impact.to_csv(results_dir / "drift_model_impact.csv", index=False, float_format="%.4f")
    figs = {"drift_psi_by_sensor": plot_psi_by_sensor(report),
            "drift_distributions": plot_drift_distributions(ref, batches, PLOT_SENSORS)}
    paths = []
    for name, fig in figs.items():
        paths.append(save_fig(fig, name, results_dir))
        plt.close(fig)

    setup_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name="simulate_drift"):
        mlflow.set_tags({"stage": "simulation", "reference": "FD001 training engines"})
        for s in summaries[:len(BATCHES)]:
            mlflow.log_metrics({f"{s['batch']}_{k}": float(s[k]) for k in
                                ("n_drift", "n_ks_drift", "n_psi_drift", "median_psi", "max_psi", "alarm")})
        for r in impact.itertuples():
            mlflow.log_metric(f"{r.model}_{r.test_set}_test_rmse", r.test_rmse)
        for p in [results_dir / "drift_report.csv", results_dir / "drift_summary.csv",
                  results_dir / "drift_model_impact.csv", *paths]:
            mlflow.log_artifact(str(p))

    print(summary.drop(columns=["drifted_features"], errors="ignore").round(3).to_string(index=False))
    print("\n", impact.round(3).to_string(index=False))
    print(f"\nreference saved to {REFERENCE_PATH}")
    return {"report": report, "summary": summary, "impact": impact}


if __name__ == "__main__":
    run()

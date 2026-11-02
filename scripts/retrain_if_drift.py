"""Block 12 (planner w9t6): drift-triggered retraining. In production this runs on a schedule
(e.g. a daily cron job) against the latest batch of sensor readings.

    python -m scripts.retrain_if_drift --batch FD001-val     # control: no alarm, nothing happens
    python -m scripts.retrain_if_drift --batch FD002-test    # alarm -> retrain -> register
    python -m scripts.retrain_if_drift --batch FD002-test --dry-run   # detect only

Steps:
  1. load the reference sample (trained_models/reference_FD001.npz, written by
     scripts/simulate_drift.py; rebuilt from the FD001 training engines if missing);
  2. load the current batch: unlabelled sensor readings (src/drift_data.BATCHES);
  3. detect: KS + PSI per sensor, alarm if >= 3 sensors drift (DECISIONS W3);
  4. no alarm -> log and stop. Alarm -> log the alert, then
       - same subset as the reference (e.g. FD001-test): there is no new kind of data to learn
         from, so retraining would only refit the same data. Flag for review, don't retrain.
       - new subset (FD002): retrain XGBoost (the tuned FD001 settings) on FD001 + FD002
         training engines (labelled run-to-failure histories: the engines of the new regime
         that have since failed), evaluate on FD002 test AND FD001 test, register the model with
         notes "retrained_after_drift", and store a new reference inside its bundle;
  5. everything is logged to the MLflow experiment 'drift-retraining'.

Why drift-triggered: scheduled retraining wastes compute when nothing changed; performance-
triggered retraining needs true RUL labels, which in maintenance arrive only when an engine
fails - weeks or months late. Input drift is visible immediately.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import mlflow

from src.config import CONFIGS_DIR, SEED, SENSORS, seed_everything
from src.drift_data import BATCHES, current_batch, prepare_combined, reference_frame, training_split
from src.evaluation import evaluate_test, time_inference
from src.feature_engineering import build_tabular_features, last_row_per_engine, tabular_feature_names
from src.models.loader import resolve_bundle_path
from src.models.xgboost_rul import XGBoostRUL
from src.monitoring import compute_feature_drift, drift_summary, load_reference, save_reference
from src.plotting import apply_style, plot_test_predictions
from src.registry import get_latest, git_commit
from src.run_utils import finish_run
from src.tracking import setup_mlflow

from scripts.simulate_drift import EXPERIMENT, REFERENCE_PATH

log = logging.getLogger("retrain_if_drift")
MODEL_NAME = "xgboost_retrained"
BASE_MODEL = "xgboost_tuned"
REFERENCE_FILE = "reference.npz"


def load_or_build_reference(path=REFERENCE_PATH):
    if not path.exists():
        log.info("reference %s missing - rebuilding it from the FD001 training engines", path)
        ref = reference_frame("FD001")
        save_reference(ref, SENSORS, path, meta={"subset": "FD001", "source": "training engines (project split)",
                                                 "units": sorted(ref["unit_id"].unique().tolist()), "n_rows": len(ref)})
    return load_reference(path)


def tabular_last_rows(test_df):
    rows = last_row_per_engine(build_tabular_features(test_df))
    assert rows["unit_id"].tolist() == sorted(test_df["unit_id"].unique()), "test order != truth order"
    return rows


def retrain(new_subset: str, ref_subset: str, summary: dict, commit: str) -> dict:
    """Retrain XGBoost on reference + new subset; must run inside an MLflow run."""
    params = json.loads((CONFIGS_DIR / "xgboost_best_params.json").read_text())
    data = prepare_combined((ref_subset, new_subset))
    names = tabular_feature_names()
    f_train, f_val = build_tabular_features(data.train), build_tabular_features(data.val)
    model = XGBoostRUL(params=params)
    t0 = time.perf_counter()
    model.fit(f_train[names], f_train["rul"], f_val[names], f_val["rul"])
    train_time = time.perf_counter() - t0
    mlflow.log_params({**model.params, "best_iteration": model.best_iteration, "trained_on": "+".join(data.subsets),
                       "n_train_rows": len(f_train), "n_val_rows": len(f_val),
                       "n_train_engines": len(data.train_units), "n_val_engines": len(data.val_units)})

    tests = {}
    for s, (test_df, truth) in data.tests.items():
        rows = tabular_last_rows(test_df)
        tests[s] = (truth, model.predict(rows[names]), rows)
    truth_new, pred_new, rows_new = tests[new_subset]
    other = {s: evaluate_test(t, p) for s, (t, p, _) in tests.items()}
    for s, m in other.items():
        mlflow.log_metrics({f"{s}_test_{k}": v for k, v in m.items()})

    before = get_latest(BASE_MODEL)
    fig = plot_test_predictions(truth_new, pred_new, f"{MODEL_NAME}: {new_subset} test set, one point per engine")
    row = finish_run(
        model=model, model_name=MODEL_NAME, subset="+".join(data.subsets), scaler=data.scaler,
        data_split={"train_units": data.train_units, "val_units": data.val_units,
                    "unit_offset_note": "unit_id = original id + 1000 * index of subset in trained_on"},
        val_metrics=model.evaluate(f_val[names], f_val["rul"]), test_truth=truth_new, test_pred=pred_new,
        timing=time_inference(model.predict, rows_new[names].to_numpy(dtype="float32")),
        train_time_s=train_time, figures={"test_predictions": fig}, commit=commit,
        notes=f"retrained_after_drift; trigger={summary['batch']}; registry test metrics = {new_subset} test",
        extra_metadata={"test_set": new_subset, "trained_on": list(data.subsets),
                        "retrain": {"base_model": BASE_MODEL,
                                    "base_model_version": None if before is None else int(before["version"]),
                                    "trigger": summary, "test_metrics_by_subset": other}})
    # The new reference is the NEW regime's training engines - what production now sends.
    # (A pooled FD001+FD002 reference would keep alarming on FD002 batches -> retrain forever.)
    bundle = resolve_bundle_path(row["path"])
    new_ref = training_split(new_subset)[0]
    save_reference(new_ref, SENSORS, bundle / REFERENCE_FILE,
                   meta={"subset": new_subset, "source": f"{new_subset} training engines (regime now served)",
                         "units": sorted(new_ref["unit_id"].unique().tolist()), "n_rows": len(new_ref)})
    mlflow.log_artifact(str(bundle / REFERENCE_FILE))
    return {"row": row, "by_subset": other, "bundle": bundle}


def run(batch: str = "FD002-test", dry_run: bool = False, reference=None) -> dict:
    """`reference`: path to a reference .npz (default: the FD001 one). After a retrain, point it at
    the served model's bundle (<bundle>/reference.npz) for the next scheduled run."""
    commit = git_commit()
    seed_everything(SEED)
    apply_style()
    ref, ref_meta = load_reference(reference) if reference else load_or_build_reference()
    cur = current_batch(batch)
    report = compute_feature_drift(ref, cur, SENSORS)
    summary = {"batch": batch, **drift_summary(report)}
    ref_subset, new_subset = ref_meta["subset"], batch.split("-")[0]

    setup_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=f"retrain_if_drift[{batch}]"):
        mlflow.log_metrics({k: float(summary[k]) for k in
                            ("n_drift", "n_ks_drift", "n_psi_drift", "median_psi", "max_psi", "alarm")})
        mlflow.log_text(report.to_csv(), "drift_report.csv")
        mlflow.set_tags({"stage": "retrain_if_drift", "batch": batch, "reference": ref_meta.get("source", ref_subset),
                         "alarm": summary["alarm"], "git_commit": commit})
        if not summary["alarm"]:
            log.info("no drift: %d/%d sensors drifting (alarm needs >= 3). Nothing to do.",
                     summary["n_drift"], summary["n_features"])
            mlflow.set_tag("action", "none")
            return {"summary": summary, "action": "none"}

        log.warning("DRIFT ALARM on %s: %d/%d sensors drift (median PSI %.2f): %s", batch, summary["n_drift"],
                    summary["n_features"], summary["median_psi"], summary["drifted_features"])
        if new_subset == ref_subset:
            log.warning("batch comes from the reference's own subset (%s): no new labelled regime to learn from. "
                        "Flagged for human review; not retraining.", ref_subset)
            mlflow.set_tag("action", "flagged_for_review")
            return {"summary": summary, "action": "flagged_for_review"}
        if dry_run:
            mlflow.set_tag("action", "dry_run")
            return {"summary": summary, "action": "dry_run"}

        mlflow.set_tag("action", "retrained")
        out = retrain(new_subset, ref_subset, summary, commit)
        # close the loop: the same batch against the NEW model's reference
        new_ref, _ = load_reference(out["bundle"] / REFERENCE_FILE)
        after = drift_summary(compute_feature_drift(new_ref, cur, SENSORS))
        mlflow.log_metrics({f"after_{k}": float(after[k]) for k in ("n_drift", "median_psi", "max_psi", "alarm")})

    before = get_latest(BASE_MODEL)
    log.info("registered %s v%s (%s)", MODEL_NAME, out["row"]["version"], out["row"]["path"])
    for s, m in out["by_subset"].items():
        log.info("  %s test: RMSE %.2f  NASA %.1f  (NASA per engine %.2f)", s, m["rmse"], m["nasa"],
                 m["nasa"] / len(current_batch(f"{s}-test")["unit_id"].unique()))
    if before is not None:
        log.info("  before retraining, %s scored FD001 test RMSE %.2f (FD002: results/drift_model_impact.csv)",
                 BASE_MODEL, before["test_rmse"])
    log.info("  same batch vs the new reference: %d sensors drift, median PSI %.2f, alarm=%s",
             after["n_drift"], after["median_psi"], after["alarm"])
    return {"summary": summary, "action": "retrained", "after": after, **out}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="FD002-test", choices=BATCHES)
    ap.add_argument("--dry-run", action="store_true", help="detect and log only; never retrain")
    ap.add_argument("--reference", help="reference .npz (default trained_models/reference_FD001.npz)")
    a = ap.parse_args()
    run(a.batch, a.dry_run, a.reference)


if __name__ == "__main__":
    main()

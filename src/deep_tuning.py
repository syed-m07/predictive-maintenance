"""Optuna tuning shared by the deep sequence models (TCN now; Bi-LSTM / uni-LSTM in Block 8).

Same rules as the XGBoost search: objective = VALIDATION RMSE, test never touched, TPE
sampler with seeded random startup, resumable SQLite study, one nested MLflow run per trial.
Extra for neural nets: a MedianPruner stops a trial early when its validation RMSE at some
epoch is worse than the median of earlier trials at that same epoch - hopeless settings stop
burning GPU time, while promising ones always run to completion.

Storage defaults to ./optuna.db; on Colab set OPTUNA_STORAGE to a Google Drive path so the
study survives disconnects (docs/COLAB.md).
"""
from __future__ import annotations

import json
import os
import time

import mlflow
import optuna

from src.config import CONFIGS_DIR, RESULTS_DIR, ROOT_DIR, SEED, SUBSET, seed_everything
from src.evaluation import evaluate_validation
from src.tracking import setup_mlflow

STORAGE = os.environ.get("OPTUNA_STORAGE", f"sqlite:///{(ROOT_DIR / 'optuna.db').as_posix()}")
N_STARTUP_TRIALS = 10

TCN_CHANNELS = {"32-32-64": [32, 32, 64], "64-64-64": [64, 64, 64], "32-64-128": [32, 64, 128],
                "64-64-64-64": [64, 64, 64, 64]}


def suggest_tcn(trial: optuna.Trial) -> dict:
    """TCN search space (planner w5t8): channels, kernel_size, dropout, lr."""
    return {"channels": TCN_CHANNELS[trial.suggest_categorical("channels", list(TCN_CHANNELS))],
            "kernel_size": trial.suggest_int("kernel_size", 2, 5),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)}


def _family(name: str):
    """(model class, search-space fn, training entry point) per model family."""
    if name == "tcn":
        from src.models.tcn_rul import TCNRUL
        from src.train_tcn import EXPERIMENT, run
        return TCNRUL, suggest_tcn, EXPERIMENT, run, {"channels", "kernel_size", "dropout"}
    raise ValueError(f"unknown family {name!r}")


def tune(family: str, subset: str = SUBSET, n_trials: int = 50, epochs: int = 50,
         train_final_model: bool = True, device: str | None = None) -> dict:
    from src.train_tcn import build_seq_inputs
    seed_everything(SEED)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Model, suggest, experiment, run_final, arch_keys = _family(family)
    _, (X_tr, y_tr, _), (X_va, y_va, _), _ = build_seq_inputs(subset)

    study = optuna.create_study(
        study_name=f"{family}_{subset}", storage=STORAGE, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=5))
    finished = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    remaining = max(0, n_trials - len(finished))
    print(f"study {study.study_name}: {len(finished)} trials done, running {remaining} more  (storage {STORAGE})")

    def objective(trial: optuna.Trial) -> float:
        p = suggest(trial)
        arch = {k: v for k, v in p.items() if k in arch_keys}
        cfg = {k: v for k, v in p.items() if k not in arch_keys}
        model = Model(arch=arch, train_cfg={**cfg, "epochs": epochs}, device=device)

        def report(epoch, val_rmse):
            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        t0 = time.perf_counter()
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.log_params(trial.params)
            try:
                model.fit(X_tr, y_tr, X_va, y_va, verbose=False, trial_callback=report)
            except optuna.TrialPruned:
                mlflow.set_tag("state", "pruned")
                raise
            m = evaluate_validation(y_va, model.predict(X_va))
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
            mlflow.log_metrics({"best_epoch": model.history.best_epoch + 1, "fit_seconds": time.perf_counter() - t0})
        trial.set_user_attr("best_epoch", model.history.best_epoch + 1)
        return m["rmse"]

    def log_trial(s, t):
        state = "pruned" if t.state == optuna.trial.TrialState.PRUNED else f"val RMSE {t.value:7.3f}"
        best = s.best_value if any(x.state == optuna.trial.TrialState.COMPLETE for x in s.trials) else float("nan")
        print(f"  trial {t.number:3d}  {state:18s} best {best:7.3f}", flush=True)

    setup_mlflow(experiment)
    with mlflow.start_run(run_name=f"optuna_study_{subset}"):
        mlflow.set_tags({"study_name": study.study_name, "sampler": "TPE", "pruner": "median", "objective": "val_rmse"})
        if remaining:
            study.optimize(objective, n_trials=remaining, callbacks=[log_trial])
        best = study.best_trial
        best_params = {**best.params, "channels": TCN_CHANNELS[best.params["channels"]]} if family == "tcn" else dict(best.params)
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_val_rmse", best.value)
        from src.train_xgboost_optuna import plot_study
        for path in plot_study(study, f"{family}_optuna", N_STARTUP_TRIALS):
            mlflow.log_artifact(str(path))
        trials_csv = RESULTS_DIR / f"{family}_optuna_trials.csv"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(trials_csv, index=False)
        mlflow.log_artifact(str(trials_csv))

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIGS_DIR / f"{family}_best_params.json"
    out.write_text(json.dumps({**best_params, "epochs": epochs}, indent=2))
    print(f"\nbest trial #{best.number}: val RMSE {best.value:.3f}\n{json.dumps(best_params, indent=2)}\nsaved -> {out}")
    if train_final_model:
        return run_final(subset, {**best_params, "epochs": epochs}, f"{family}_tuned",
                         notes=f"optuna {len(study.trials)} trials, best #{best.number}", device=device)
    return {"best_params": best_params, "best_val_rmse": best.value}

"""Tune XGBoost with Optuna (TPE), then train and register the tuned model.

    python -m src.train_xgboost_optuna                  # 100 trials (resumes if interrupted)
    python -m src.train_xgboost_optuna --n-trials 5     # quick smoke run

How the search works: grid search tries every combination (cost explodes with the number of
settings); random search samples blindly. Optuna's TPE (Tree-structured Parzen Estimator)
splits finished trials into "good" and "bad" groups, models where each group's settings
lie, and proposes new settings that look like the good group - so later trials concentrate
on promising regions. The first trials are random (startup) to explore.

Rules:
- the objective is VALIDATION RMSE (20 held-out engines). The test set is never touched
  during tuning - it is scored once, for the final model only (DECISIONS.md V5).
- the study lives in optuna.db (SQLite): an interrupted search resumes where it stopped.
- every trial is a nested MLflow run under one parent run.
- final model = same 80/20 protocol as every other model (DEVIATIONS.txt DV-002).
"""
from __future__ import annotations

import argparse
import json
import time

import mlflow
import optuna

from src.config import CONFIGS_DIR, RESULTS_DIR, ROOT_DIR, SEED, SUBSET, seed_everything
from src.evaluation import evaluate_validation
from src.models.xgboost_rul import XGBoostRUL
from src.plotting import apply_style
from src.tracking import setup_mlflow
from src.train_xgboost import EXPERIMENT, build_xgb_inputs
from src.train_xgboost import run as train_final

STORAGE = f"sqlite:///{(ROOT_DIR / 'optuna.db').as_posix()}"
N_TRIALS = 100
N_STARTUP_TRIALS = 20          # random exploration before TPE starts modelling


def suggest_params(trial: optuna.Trial) -> dict:
    """Search space from HANDOFF section 10.5. n_estimators is an upper bound - early
    stopping (50 rounds on validation) still picks the actual number of trees."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }


def make_objective(f_train, f_val, names):
    X_tr, y_tr = f_train[names], f_train["rul"]
    X_va, y_va = f_val[names], f_val["rul"]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        t0 = time.perf_counter()
        model = XGBoostRUL(params=params).fit(X_tr, y_tr, X_va, y_va)
        metrics = evaluate_validation(y_va, model.predict(X_va))
        trial.set_user_attr("best_iteration", model.best_iteration)
        trial.set_user_attr("fit_seconds", time.perf_counter() - t0)
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.log_params({**params, "best_iteration": model.best_iteration})
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        return metrics["rmse"]

    return objective


def plot_study(study: optuna.Study, prefix: str, n_startup: int = N_STARTUP_TRIALS) -> list:
    """Optimisation history, hyperparameter importance, per-setting slice plot -> results/.
    Shared by the deep-model studies (src/deep_tuning.py)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from src.plotting import BLUE, MUTED, ORANGE, TEXT_SECONDARY

    apply_style()
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    nums = np.array([t.number for t in done]); vals = np.array([t.value for t in done])
    paths = []

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.scatter(nums, vals, s=14, color=MUTED, label="trial")
    ax.plot(nums, np.minimum.accumulate(vals), color=BLUE, label="best so far")
    ax.axvline(n_startup - 0.5, color=TEXT_SECONDARY, lw=1, ls="--")
    ax.text(n_startup, ax.get_ylim()[1] * 0.98, " TPE starts", va="top", color=TEXT_SECONDARY, fontsize=8.5)
    ax.set(title="Optuna: validation RMSE per trial (lower is better)", xlabel="trial", ylabel="validation RMSE")
    ax.legend(loc="upper right"); fig.tight_layout()
    paths.append(RESULTS_DIR / f"{prefix}_history.png"); fig.savefig(paths[-1]); plt.close(fig)

    if len(done) < 2:                       # importance / slice plots need >= 2 finished trials
        return paths
    imp = optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    keys = list(imp)[::-1]
    ax.barh(keys, [imp[k] for k in keys], color=ORANGE, height=0.65)
    ax.set(title="Which settings mattered? (fANOVA importance)", xlabel="share of validation-RMSE variance explained")
    ax.grid(axis="y", visible=False); fig.tight_layout()
    paths.append(RESULTS_DIR / f"{prefix}_param_importance.png"); fig.savefig(paths[-1]); plt.close(fig)

    # Slice plot: one panel per setting, validation RMSE vs that setting's value (a readable
    # replacement for parallel coordinates). How to read: each dot is a trial; where the dots
    # sit lowest is the good range for that setting; the orange dot is the best trial. The
    # y-axis is cut at the 90th percentile so a few disastrous trials don't flatten the rest.
    best = study.best_trial
    params = [k for k in imp]                          # ordered by importance
    ncol = min(4, len(params)); nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.6 * nrow), sharey=True, squeeze=False)
    ytop = np.percentile(vals, 90)
    for ax, p in zip(axes.flat, params):
        xs = [t.params[p] for t in done]
        ax.scatter(xs, vals, s=10, color=MUTED)
        ax.scatter([best.params[p]], [best.value], s=40, color=ORANGE, zorder=3)
        if isinstance(xs[0], str):
            ax.tick_params(axis="x", rotation=30)
        elif p in ("learning_rate", "lr"):
            ax.set_xscale("log")
        ax.set(title=f"{p}  ({imp[p]:.0%})", ylim=(vals.min() - 0.3, ytop))
    for ax in axes.flat[len(params):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("validation RMSE")
    fig.suptitle("Optuna: validation RMSE vs each setting (orange = best trial; % = importance)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    paths.append(RESULTS_DIR / f"{prefix}_slices.png"); fig.savefig(paths[-1]); plt.close(fig)
    return paths


def run(subset: str = SUBSET, n_trials: int = N_TRIALS, train_final_model: bool = True) -> dict:
    seed_everything(SEED)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d, f_train, f_val, _, names = build_xgb_inputs(subset)
    study = optuna.create_study(
        study_name=f"xgboost_{subset}", storage=STORAGE, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS))
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    print(f"study xgboost_{subset}: {done} trials already done, running {remaining} more")

    setup_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=f"optuna_study_{subset}"):
        mlflow.set_tags({"study_name": study.study_name, "sampler": "TPE", "objective": "val_rmse"})
        t0 = time.perf_counter()
        if remaining:
            study.optimize(make_objective(f_train, f_val, names), n_trials=remaining,
                           show_progress_bar=False, callbacks=[lambda s, t: print(
                               f"  trial {t.number:3d}  val RMSE {t.value:7.3f}   best {s.best_value:7.3f}", flush=True)])
        best = study.best_trial
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metrics({"best_val_rmse": best.value, "n_trials": len(study.trials),
                            "search_seconds": time.perf_counter() - t0})
        for p in plot_study(study, "xgboost_optuna"):
            mlflow.log_artifact(str(p))
        # every trial as CSV (committed): the full search history, inspectable without optuna.db
        trials_csv = RESULTS_DIR / "xgboost_optuna_trials.csv"
        study.trials_dataframe().to_csv(trials_csv, index=False)
        mlflow.log_artifact(str(trials_csv))

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIGS_DIR / "xgboost_best_params.json"
    out.write_text(json.dumps(best.params, indent=2))
    print(f"\nbest trial #{best.number}: val RMSE {best.value:.3f}  (best_iteration {best.user_attrs['best_iteration']})")
    print(json.dumps(best.params, indent=2), f"\nsaved -> {out}")

    if train_final_model:
        return train_final(subset, best.params, "xgboost_tuned",
                           notes=f"optuna {len(study.trials)} trials, best trial #{best.number}")
    return {"best_params": best.params, "best_val_rmse": best.value}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS, help="TOTAL trials in the study (resumes)")
    ap.add_argument("--no-final", action="store_true", help="only search; don't train the tuned model")
    a = ap.parse_args()
    run(a.subset, a.n_trials, not a.no_final)


if __name__ == "__main__":
    main()

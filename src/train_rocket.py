"""Train, evaluate and register a ROCKET (MiniRocket + ridge) RUL model, on CPU.

    python -m src.train_rocket                       # 10,000 kernels, alpha chosen on validation engines

Same inputs as the deep models (30x14 windows, src/train_sequence.py), same scoring and
registry (finish_run). ROCKET has no Optuna study: its one hyperparameter, the ridge penalty
alpha, is searched over a fixed grid on the validation engines inside fit (decisions K2-K3).
Training time is logged in two parts: the MiniRocket transform and the ridge fit + alpha search.
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.config import SEED, SEQ_LEN, SUBSET, seed_everything
from src.evaluation import time_inference
from src.models.rocket_rul import ROCKETRUL
from src.plotting import BLUE, ORANGE, apply_style, plot_life_trajectories, plot_test_predictions
from src.registry import git_commit
from src.run_utils import finish_run
from src.tracking import setup_mlflow
from src.train_sequence import build_seq_inputs, val_trajectory_frame

EXPERIMENT = "rul-rocket"


def plot_alpha_search(model: ROCKETRUL, title: str) -> plt.Figure:
    """How to read: validation RMSE (y) for each ridge penalty alpha (x, log scale). Small alpha
    = the linear layer follows the ~10,000 features closely (risk of overfitting); large alpha =
    coefficients shrunk toward zero (risk of underfitting). The marked point is the alpha kept."""
    a = np.array(sorted(model.alpha_scores)); r = np.array([model.alpha_scores[x] for x in a])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(a, r, color=BLUE, marker="o", ms=4)
    ax.scatter([model.alpha], [model.alpha_scores[model.alpha]], s=70, color=ORANGE, zorder=3,
               label=f"chosen alpha = {model.alpha:.3g}")
    ax.set_xscale("log"); ax.set(xlabel="ridge penalty alpha (log scale)", ylabel="validation RMSE (cycles)", title=title)
    ax.legend(loc="upper center", frameon=False); fig.tight_layout()
    return fig


def run(subset: str = SUBSET, num_kernels: int = 10_000, model_name: str = "rocket", notes: str = "") -> dict:
    commit = git_commit()          # the code version that actually runs (recorded in the registry)
    seed_everything(SEED)
    apply_style()
    d, (X_tr, y_tr, _), (X_va, y_va, u_va), (X_te, _) = build_seq_inputs(subset)

    setup_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=model_name):
        model = ROCKETRUL(num_kernels=num_kernels)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr, X_va, y_va)
        train_time = time.perf_counter() - t0
        mlflow.log_params({"num_kernels": num_kernels, "n_rocket_features": int(model.coef.shape[0]),
                           "alpha": model.alpha, "alpha_grid": "logspace(-3,6,10)", "subset": subset,
                           "seq_len": SEQ_LEN, "seed": SEED, "n_train_windows": len(X_tr),
                           "n_val_windows": len(X_va), "transform_time_s": round(model.timings["transform_s"], 2),
                           "ridge_time_s": round(model.timings["ridge_s"], 2)})

        val_pred = model.predict(X_va)
        val_metrics = model.evaluate(X_va, y_va)
        test_pred = model.predict(X_te)
        timing = time_inference(model.predict, X_te)
        figures = {
            "test_predictions": plot_test_predictions(d.test_truth, test_pred, f"{model_name}: test set ({subset}), one window per engine"),
            "val_trajectories": plot_life_trajectories(val_trajectory_frame(d, u_va, val_pred),
                                                       f"{model_name}: predicted vs target RUL over validation engines' lives"),
            "alpha_search": plot_alpha_search(model, f"{model_name}: ridge penalty chosen on validation engines"),
        }
        row = finish_run(model=model, model_name=model_name, subset=subset, scaler=d.scaler,
                         data_split={"train_units": d.train_units, "val_units": d.val_units},
                         val_metrics=val_metrics, test_truth=d.test_truth, test_pred=test_pred,
                         timing=timing, train_time_s=train_time, figures=figures, notes=notes, commit=commit,
                         extra_metadata={"timings": model.timings})

    tm = row["test_metrics"]
    print(f"\n{model_name} v{row['version']}  ({subset})  {model.coef.shape[0]} features, alpha {model.alpha:.3g}")
    print(f"  validation  RMSE {val_metrics['rmse']:.2f}  MAE {val_metrics['mae']:.2f}")
    print(f"  TEST        RMSE {tm['rmse']:.2f}  MAE {tm['mae']:.2f}  R2 {tm['r2']:.3f}  NASA {tm['nasa']:.1f}")
    print(f"  TEST clip125 RMSE {tm['rmse_clip125']:.2f}  NASA {tm['nasa_clip125']:.1f}   (literature-comparable)")
    print(f"  latency {timing['latency_single_ms']:.3f} ms/engine single, {timing['throughput_ms_per_item']:.4f} ms/engine batch")
    print(f"  train {train_time:.1f}s  (transform {model.timings['transform_s']:.1f}s + ridge/alpha search {model.timings['ridge_s']:.1f}s)")
    print(f"  bundle: {row['path']}   mlflow run: {row['mlflow_run_id']}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--num-kernels", type=int, default=10_000)
    ap.add_argument("--model-name", default="rocket")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    run(a.subset, a.num_kernels, a.model_name, a.notes)


if __name__ == "__main__":
    main()

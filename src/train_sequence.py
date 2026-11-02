"""Shared training run for every PyTorch sequence model (TCN, Bi-LSTM, uni-LSTM).

Steps: prepare_data (leakage-safe split + scaler) -> 30x14 windows -> train with early
stopping on the validation engines (best epoch restored) -> validation metrics -> ONE window
per test engine (its last 30 cycles) scored against RUL_FD00X.txt -> CPU timing -> figures ->
finish_run (bundle + MLflow + registry). Model-specific extras (receptive field, attention
heatmap) come in through two small hooks, so every deep model follows identical steps.
"""
from __future__ import annotations

import time
from typing import Callable

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.config import SEED, SEQ_LEN, SUBSET, seed_everything
from src.evaluation import time_inference
from src.feature_engineering import generate_sequences, last_window_per_engine, prepare_data
from src.models.torch_base import TorchRULModel
from src.plotting import BLUE, ORANGE, apply_style, plot_life_trajectories, plot_test_predictions
from src.registry import git_commit
from src.run_utils import finish_run
from src.tracking import setup_mlflow


def build_seq_inputs(subset: str = SUBSET, seq_len: int = SEQ_LEN):
    """Sequence inputs for every deep model / ROCKET. Returns (data, train, val, test) where
    train/val = (X, y, unit_ids) and test = (X, unit_ids)."""
    d = prepare_data(subset)
    train = generate_sequences(d.train, seq_len)
    val = generate_sequences(d.val, seq_len)
    X_test, test_units = last_window_per_engine(d.test, seq_len)
    assert test_units.tolist() == sorted(d.test["unit_id"].unique()), "test order != truth order"
    return d, train, val, (X_test, test_units)


def val_window_frame(d, val_units_per_window, seq_len: int = SEQ_LEN) -> pd.DataFrame:
    """Rows (unit_id, cycle, rul) aligned with the validation windows (window i ends at
    cycle seq_len + i of its engine)."""
    v = d.val.groupby("unit_id", sort=True).apply(lambda g: g.iloc[seq_len - 1:], include_groups=False)
    v = v.reset_index(level=0)[["unit_id", "cycle", "rul"]].reset_index(drop=True)
    assert (v["unit_id"].to_numpy() == val_units_per_window).all()
    return v


def val_trajectory_frame(d, val_units_per_window, preds, seq_len: int = SEQ_LEN) -> pd.DataFrame:
    v = val_window_frame(d, val_units_per_window, seq_len)
    assert len(v) == len(preds)
    return v.assign(pred=preds)


def plot_learning_curves(history, title: str) -> plt.Figure:
    """How to read: training loss (left, MSE) and validation RMSE (right) per epoch. Both should
    fall; validation flattening or rising while training keeps falling = overfitting. The
    marker shows the epoch whose weights were kept."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.3))
    ep = np.arange(1, len(history.train_loss) + 1)
    a.plot(ep, history.train_loss, color=BLUE); a.set(title="training loss (MSE)", xlabel="epoch")
    b.plot(ep, history.val_rmse, color=ORANGE)
    b.scatter([history.best_epoch + 1], [history.best_val_rmse], s=40, color=ORANGE, zorder=3)
    b.set(title=f"validation RMSE (best epoch {history.best_epoch + 1})", xlabel="epoch")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold"); fig.tight_layout()
    return fig


def run_sequence_model(model_cls: type[TorchRULModel], *, experiment: str, arch_keys: set[str],
                       subset: str = SUBSET, params: dict | None = None, model_name: str = "model",
                       notes: str = "", device: str | None = None, verbose: bool = True,
                       extra_params: Callable[[TorchRULModel], dict] | None = None,
                       extra_figures: Callable[..., dict] | None = None) -> dict:
    """extra_params(model) -> dict logged to MLflow after training.
    extra_figures(model, d, X_val, u_val, model_name) -> {name: figure}."""
    commit = git_commit()          # the code version that actually runs (recorded in the registry)
    seed_everything(SEED)
    apply_style()
    params = params or {}
    arch = {k: v for k, v in params.items() if k in arch_keys}
    train_cfg = {k: v for k, v in params.items() if k not in arch_keys}
    d, (X_tr, y_tr, _), (X_va, y_va, u_va), (X_te, _) = build_seq_inputs(subset)

    setup_mlflow(experiment)
    with mlflow.start_run(run_name=model_name):
        model = model_cls(arch=arch, train_cfg=train_cfg, device=device)
        mlflow.log_params({**model.arch, **model.train_cfg.__dict__, "subset": subset, "seq_len": SEQ_LEN,
                           "device": str(model.device), "n_train_windows": len(X_tr), "n_val_windows": len(X_va),
                           "torch_version": __import__("torch").__version__})
        train_device = str(model.device)                      # model moves to CPU after fit
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr, X_va, y_va, verbose=verbose)
        train_time = time.perf_counter() - t0
        h = model.history
        for i, (tl, vr) in enumerate(zip(h.train_loss, h.val_rmse)):
            mlflow.log_metrics({"epoch_train_mse": tl, "epoch_val_rmse": vr}, step=i)
        mlflow.log_params({"best_epoch": h.best_epoch + 1, "epochs_run": len(h.val_rmse),
                           **(extra_params(model) if extra_params else {})})

        val_pred = model.predict(X_va)
        val_metrics = model.evaluate(X_va, y_va)
        test_pred = model.predict(X_te)
        timing = time_inference(model.predict, X_te)
        figures = {
            "test_predictions": plot_test_predictions(d.test_truth, test_pred, f"{model_name}: test set ({subset}), one window per engine"),
            "val_trajectories": plot_life_trajectories(val_trajectory_frame(d, u_va, val_pred),
                                                       f"{model_name}: predicted vs target RUL over validation engines' lives"),
            "learning_curves": plot_learning_curves(h, f"{model_name}: learning curves"),
            **(extra_figures(model, d, X_va, u_va, model_name) if extra_figures else {}),
        }
        row = finish_run(model=model, model_name=model_name, subset=subset, scaler=d.scaler,
                         data_split={"train_units": d.train_units, "val_units": d.val_units},
                         val_metrics=val_metrics, test_truth=d.test_truth, test_pred=test_pred,
                         timing=timing, train_time_s=train_time, figures=figures, notes=notes, commit=commit,
                         extra_metadata={"seq_len": SEQ_LEN})

    tm = row["test_metrics"]
    print(f"\n{model_name} v{row['version']}  ({subset})  best epoch {h.best_epoch + 1}/{len(h.val_rmse)}  trained on {train_device}")
    print(f"  validation  RMSE {val_metrics['rmse']:.2f}  MAE {val_metrics['mae']:.2f}")
    print(f"  TEST        RMSE {tm['rmse']:.2f}  MAE {tm['mae']:.2f}  R2 {tm['r2']:.3f}  NASA {tm['nasa']:.1f}")
    print(f"  TEST clip125 RMSE {tm['rmse_clip125']:.2f}  NASA {tm['nasa_clip125']:.1f}   (literature-comparable)")
    print(f"  latency {timing['latency_single_ms']:.3f} ms/engine single, {timing['throughput_ms_per_item']:.4f} ms/engine batch; train {train_time:.1f}s")
    print(f"  bundle: {row['path']}   mlflow run: {row['mlflow_run_id']}")
    return row

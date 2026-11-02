"""Block 10 (planner w7t8): SHAP for XGBoost and the best deep models, and whether they agree.

    python -m scripts.shap_analysis            # from the repo root, ~6 min on a laptop CPU

Explained set: the 100 FD001 test engines' official inputs (their last row / last 30-cycle
window) - the same predictions the comparison table scores. This explains predictions; nothing
is tuned, so using the test inputs leaks nothing.

Models:
  xgboost_tuned    TreeExplainer (exact)
  lstm_uni_tuned   GradientExplainer - best deep model by test RMSE
  tcn_tuned        GradientExplainer - statistically tied with lstm_uni_tuned (paired
                   bootstrap, results/model_comparison_paired.csv), so it is explained too
Background for GradientExplainer: 200 training windows drawn with seed 42; 1,000 samples per
engine (checked: sensor ranking stable across seeds, rho 0.99; additivity gap ~1.5 cycles).

Writes:
  results/shap_xgboost_summary.png          beeswarm, top 20 engineered features
  results/shap_xgboost_waterfall.png        one engine's prediction, feature by feature
  results/shap_deep_time_sensor.png         where in the window each sensor mattered
  results/shap_sensor_agreement.png         per-sensor importance, all three models
  results/shap_sensor_importance.csv        the numbers behind the agreement figure
  results/shap_rank_agreement.csv           Spearman rho and top-5 overlap per model pair
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.comparison import test_inputs
from src.config import RESULTS_DIR, SEED, SENSORS, SEQ_LEN, SUBSET, seed_everything
from src.explain import (compare_rankings, gradient_shap, group_by_sensor_sequence,
                         group_by_sensor_tabular, sensor_importance, tree_shap)
from src.feature_engineering import generate_sequences, prepare_data
from src.models.loader import load_bundle
from src.plotting import (DIVERGING, apply_style, plot_sensor_agreement, plot_shap_time_sensor,
                          save_fig)
from src.registry import get_latest

TREE_MODEL = "xgboost_tuned"
DEEP_MODELS = ["lstm_uni_tuned", "tcn_tuned"]
WATERFALL_ENGINE = 76          # near failure (true RUL 10); also panel 1 of multi_model_predictions.png
N_BACKGROUND = 200
N_SAMPLES = 1000


def bundle(name: str, subset: str):
    row = get_latest(name)
    if row is None or row["subset"] != subset:
        raise LookupError(f"{name} ({subset}) is not in the registry")
    return load_bundle(row["path"])


def run(subset: str = SUBSET, results_dir=RESULTS_DIR) -> dict:
    import shap
    seed_everything(SEED)
    apply_style()
    d = prepare_data(subset)
    inp = test_inputs(d, SEQ_LEN)
    units = inp["unit_id"].tolist()
    figs, importances, gaps = {}, {}, {}

    # ---- XGBoost: exact tree SHAP
    xgb, _ = bundle(TREE_MODEL, subset)
    exp = tree_shap(xgb, inp["tabular"])
    pred = xgb.predict(inp["tabular"])
    gaps[TREE_MODEL] = float(np.abs(exp.values.sum(axis=1) + exp.base_values - pred).max())
    importances[TREE_MODEL] = sensor_importance(group_by_sensor_tabular(exp.values, xgb.feature_names))

    results_dir.mkdir(parents=True, exist_ok=True)
    # shap's plot functions draw on the CURRENT figure: give each its own and save it at once
    plt.figure()
    shap.plots.beeswarm(exp, max_display=20, show=False, color=DIVERGING, plot_size=(9, 7))
    fig = plt.gcf()
    fig.suptitle(f"{TREE_MODEL}: SHAP values of the top 20 features, 100 test engines",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    save_fig(fig, "shap_xgboost_summary", results_dir)
    plt.close(fig)

    i = units.index(WATERFALL_ENGINE)
    plt.figure()
    shap.plots.waterfall(exp[i], max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    lo, hi = fig.axes[0].get_xlim()
    for ax in fig.axes:                    # room for the leftmost value label; twin axes move together
        ax.set_xlim(lo - 0.08 * (hi - lo), hi)
    fig.suptitle(f"{TREE_MODEL}: why test engine {WATERFALL_ENGINE} gets a predicted RUL of {pred[i]:.1f} "
                 f"(true {d.test_truth[i]:.0f})", x=0.01, ha="left", fontsize=11, fontweight="bold")
    save_fig(fig, "shap_xgboost_waterfall", results_dir)
    plt.close(fig)

    # ---- deep models: expected gradients on the windows
    X_train, _, _ = generate_sequences(d.train, SEQ_LEN)
    bg = X_train[np.random.default_rng(SEED).choice(len(X_train), N_BACKGROUND, replace=False)]
    time_maps = {}
    for name in DEEP_MODELS:
        model, _ = bundle(name, subset)
        vals = gradient_shap(model, bg, inp["sequence"], nsamples=N_SAMPLES, seed=SEED)
        p = model.predict(inp["sequence"])
        gaps[name] = float(np.abs(model.predict(bg).mean() + vals.sum(axis=(1, 2)) - p).mean())
        importances[name] = sensor_importance(group_by_sensor_sequence(vals))
        time_maps[name] = np.abs(vals).mean(axis=0)                     # (seq_len, n_sensors)
        print(f"  {name}: SHAP done, mean additivity gap {gaps[name]:.2f} cycles")
    figs["shap_deep_time_sensor"] = plot_shap_time_sensor(
        time_maps, SENSORS, "Deep models: mean |SHAP| per sensor and time step (100 test windows, shared scale)")

    imp = pd.DataFrame(importances)[[TREE_MODEL] + DEEP_MODELS]
    agreement = compare_rankings(importances)
    rho = agreement.set_index(["model_a", "model_b"])["spearman_rho"]
    figs["shap_sensor_agreement"] = plot_sensor_agreement(
        imp, f"Per-sensor SHAP importance (Spearman rho: xgb-lstm {rho[(TREE_MODEL, DEEP_MODELS[0])]:.2f}, "
             f"xgb-tcn {rho[(TREE_MODEL, DEEP_MODELS[1])]:.2f}, lstm-tcn {rho[tuple(DEEP_MODELS)]:.2f})")

    imp.round(4).rename_axis("sensor").to_csv(results_dir / "shap_sensor_importance.csv")
    agreement.round(4).to_csv(results_dir / "shap_rank_agreement.csv", index=False)
    for name, fig in figs.items():
        save_fig(fig, name, results_dir)
        plt.close(fig)

    print(f"\nTreeExplainer max additivity gap: {gaps[TREE_MODEL]:.2e}")
    print("per-sensor share of importance:\n", imp.sort_values(TREE_MODEL, ascending=False).round(3).to_string())
    print("\n", agreement.round(3).to_string(index=False))
    return {"importance": imp, "agreement": agreement, "gaps": gaps}


if __name__ == "__main__":
    run()

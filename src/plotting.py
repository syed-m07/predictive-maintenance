"""Shared matplotlib style for every figure in notebooks/ and results/.

One place for colours and chart chrome so all figures read as one system:
- categorical colours are assigned in a FIXED order (slot 1, 2, 3 ...), never cycled;
- a series keeps its colour across figures (e.g. 'raw' vs 'clipped' RUL);
- text is drawn in neutral ink, never in a series colour;
- diverging maps use two hues with a neutral grey midpoint (no rainbow maps).
Palette: validated reference palette (light mode) from the dataviz guidelines.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.config import RESULTS_DIR

# ---- surfaces & ink
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED = "#b5b4ae"          # de-emphasised marks (e.g. dropped columns)

# ---- categorical slots, fixed order
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = CATEGORICAL[:5]

# ---- diverging: orange (negative) -> neutral grey -> blue (positive)
DIVERGING = LinearSegmentedColormap.from_list("pm_diverging", [ORANGE, "#efeeea", BLUE])


def apply_style() -> None:
    """Thin marks, recessive grid/axes, neutral text."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "axes.formatter.useoffset": False,   # show 2388.2, not "+2.388e3" offsets
        "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2.0,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "legend.frameon": False, "legend.fontsize": 8.5,
    })


def save_fig(fig: plt.Figure, name: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Save to results/<name>.png (committed: small, and used by the README)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.png"
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------- result figures
def plot_test_predictions(truth, pred, title: str, max_rul: int = 125) -> plt.Figure:
    """One point per test engine, engines sorted by true RUL.

    How to read: the blue line is the true remaining life; orange dots are predictions.
    Dots ABOVE the line are LATE (dangerous: the model thinks the engine has more life left
    than it does); dots below are early. The dashed line is the 125 cap - no prediction can
    go above it, so engines whose truth is above it carry an unavoidable error.
    """
    import numpy as np
    truth, pred = np.asarray(truth, float), np.clip(np.asarray(pred, float), 0, max_rul)
    order = np.argsort(truth)
    x = np.arange(len(truth))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(x, truth[order], color=BLUE, label="true RUL")
    ax.scatter(x, pred[order], s=16, color=ORANGE, zorder=3, label="predicted",
               edgecolors=SURFACE, linewidths=0.8)
    ax.axhline(max_rul, color=TEXT_SECONDARY, lw=1, ls="--")
    ax.text(len(truth) - 1, max_rul + 3, f"cap {max_rul}", color=TEXT_SECONDARY, fontsize=8.5, ha="right")
    ax.set(title=title, xlabel="test engines, sorted by true RUL", ylabel="RUL (cycles)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_life_trajectories(df, title: str, n: int = 4) -> plt.Figure:
    """Predicted vs true (clipped) RUL across the whole life of a few validation engines.

    `df` needs columns unit_id, cycle, rul, pred. How to read: each panel is one engine from
    new (left) to failure (right). A good model's orange line hugs the blue target: flat at 125
    while healthy, then descending to 0.
    """
    units = sorted(df["unit_id"].unique())
    picks = [units[i] for i in np.linspace(0, len(units) - 1, n).round().astype(int)]
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), sharey=True)
    for ax, u in zip(np.atleast_1d(axes), picks):
        e = df[df["unit_id"] == u]
        ax.plot(e["cycle"], e["rul"], color=BLUE, label="target (clipped)")
        ax.plot(e["cycle"], np.clip(e["pred"], 0, 125), color=ORANGE, lw=1.4, label="predicted")
        ax.set(title=f"engine {u}", xlabel="cycle")
    np.atleast_1d(axes)[0].set_ylabel("RUL (cycles)")
    np.atleast_1d(axes)[0].legend(loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance, title: str, top: int = 20) -> plt.Figure:
    """Horizontal bars, largest on top. How to read: longer bar = the model relied on that
    feature more (here: total loss reduction from splits on it). Not the same as SHAP."""
    s = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * top + 1.2))
    ax.barh(s.index, s.values, color=BLUE, height=0.7)
    ax.set(title=title, xlabel="importance (gain)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- model comparison
# One colour per model FAMILY (fixed slots); the tuned variant is solid, the default dashed.
FAMILY_COLORS = {"xgboost": BLUE, "tcn": ORANGE, "lstm_bi": AQUA, "lstm_uni": YELLOW, "rocket": MAGENTA}


def model_family(name: str) -> str:
    for fam in ("lstm_bi", "lstm_uni", "xgboost", "tcn", "rocket"):
        if name.startswith(fam):
            return fam
    return "other"


def model_style(name: str) -> dict:
    """Line style for a model: family colour; solid if tuned (or the only variant), else dashed."""
    fam = model_family(name)
    default_variant = fam != "rocket" and not name.endswith("_tuned")
    return {"color": FAMILY_COLORS.get(fam, MUTED), "ls": "--" if default_variant else "-",
            "lw": 1.3 if default_variant else 2.0}


def plot_metric_intervals(table, reference: str, title: str) -> plt.Figure:
    """Test RMSE and NASA score per model with 95 % bootstrap intervals.

    `table` is indexed by model with columns test_rmse, rmse_ci_lo/hi, nasa, nasa_ci_lo/hi,
    sorted best-first. How to read: the dot is the score on the real 100 test engines; the bar
    is the range it would plausibly take on a different draw of 100 engines. Overlapping bars
    hint that two models may not really differ - but the paired figure is the proper test.
    The dashed line marks the reference (best) model's score.
    """
    t = table.iloc[::-1]                                  # best on top
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(t) + 1.6), sharey=True)
    for ax, (m, lo, hi, label) in zip(axes, [("test_rmse", "rmse_ci_lo", "rmse_ci_hi", "test RMSE (cycles)"),
                                             ("nasa", "nasa_ci_lo", "nasa_ci_hi", "NASA score (sum over 100 engines)")]):
        colors = [model_style(n)["color"] for n in t.index]
        ax.hlines(y, t[lo], t[hi], color=colors, lw=2.2)
        ax.scatter(t[m], y, s=36, color=colors, zorder=3, edgecolors=SURFACE, linewidths=0.8)
        ax.axvline(table.loc[reference, m], color=TEXT_SECONDARY, lw=1, ls="--")
        ax.set(xlabel=label)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, t.index)
    axes[0].set_title("lower is better", loc="left", fontsize=9, fontweight="normal", color=TEXT_SECONDARY)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_paired_differences(paired, reference: str, title: str) -> plt.Figure:
    """Each model's test RMSE minus the reference's, with the 95 % paired-bootstrap interval.

    How to read: a bar entirely to the RIGHT of zero means the model is reliably worse than the
    reference on this test set. A bar that crosses zero means the 100 test engines cannot tell
    the two apart - calling either one "the winner" would be over-claiming.
    """
    p = paired.sort_values("rmse_diff", ascending=False)
    y = np.arange(len(p))
    colors = [model_style(n)["color"] for n in p.index]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(p) + 1.4))
    ax.axvline(0, color=TEXT_SECONDARY, lw=1)
    ax.hlines(y, p["rmse_diff_ci_lo"], p["rmse_diff_ci_hi"], color=colors, lw=2.2)
    ax.scatter(p["rmse_diff"], y, s=36, color=colors, zorder=3, edgecolors=SURFACE, linewidths=0.8)
    for yi, (lo, hi) in zip(y, zip(p["rmse_diff_ci_lo"], p["rmse_diff_ci_hi"])):
        verdict = "worse" if lo > 0 else ("better" if hi < 0 else "tie")
        ax.text(hi + 0.08, yi, verdict, va="center", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_yticks(y, p.index)
    ax.set(title=title, xlabel=f"test RMSE minus {reference}'s (cycles); > 0 = worse than {reference}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_multi_model_trajectories(trajectories: dict, title: str, max_rul: int = 125) -> plt.Figure:
    """Predicted vs true RUL over the recorded history of a few TEST engines, all models.

    `trajectories` maps unit_id -> {"cycle": (n,), "truth": (n,), "preds": {model: (n,)}}.
    How to read: each panel is one test engine; x is the cycle at which a prediction is made
    (from cycle 30, when the first full window exists, to the last recorded cycle). The grey
    line is the true remaining life (it keeps rising to the left, past the 125 cap, because the
    truth is not clipped). The point at the far right is the one the test score uses. Lines
    that sit ABOVE the grey line near the end are late predictions - the costly kind.
    Colour = model family; solid = tuned, dashed = default settings.
    """
    units = list(trajectories)
    fig, axes = plt.subplots(1, len(units), figsize=(4.6 * len(units), 4.0), sharey=True)
    for ax, u in zip(np.atleast_1d(axes), units):
        tr = trajectories[u]
        ax.plot(tr["cycle"], tr["truth"], color=TEXT_SECONDARY, lw=2.6, label="true RUL", zorder=1)
        for name, pred in tr["preds"].items():
            ax.plot(tr["cycle"], np.clip(pred, 0, max_rul), label=name, zorder=2, **model_style(name))
        ax.axhline(max_rul, color=MUTED, lw=1, ls=":")
        ax.set(title=f"test engine {u}: true RUL at last cycle = {tr['truth'][-1]:.0f}",
               xlabel="cycle (prediction made here)")
        ax.set_ylim(0, max(max_rul, float(np.max(tr["truth"]))) + 8)
    np.atleast_1d(axes)[0].set_ylabel("RUL (cycles)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- SHAP
SEQUENTIAL = LinearSegmentedColormap.from_list("pm_sequential", [SURFACE, BLUE])


def plot_shap_time_sensor(maps: dict, sensors: list[str], title: str) -> plt.Figure:
    """Mean |SHAP| per (sensor, time step) for sequence models; one panel each, SHARED colour scale.

    `maps` is {model: (seq_len, n_sensors) array}. How to read: x is the position inside the
    30-cycle window (30 = the most recent cycle), y the sensor; darker = that sensor at that
    moment moved predictions more (in cycles of RUL, either direction). A dark right edge
    means the model relies on the newest readings.
    """
    vmax = max(float(m.max()) for m in maps.values())
    fig, axes = plt.subplots(1, len(maps), figsize=(5.6 * len(maps) + 0.8, 4.4), sharey=True)
    for ax, (name, m) in zip(np.atleast_1d(axes), maps.items()):
        im = ax.imshow(m.T, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=vmax,
                       extent=(0.5, m.shape[0] + 0.5, len(sensors) - 0.5, -0.5))
        ax.set_yticks(range(len(sensors)), sensors)
        ax.set(title=name, xlabel="time step in window (30 = most recent cycle)")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("sensor")
    fig.colorbar(im, ax=axes, label="mean |SHAP| (cycles of RUL)", shrink=0.85)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    return fig


def plot_sensor_agreement(importance, title: str) -> plt.Figure:
    """Per-sensor share of SHAP importance, one bar per model, sensors sorted by average share.

    `importance` is a DataFrame (sensors x models), each column summing to 1. How to read:
    long bars for the same sensor across ALL models = a robust degradation signal, not a quirk
    of one model. Shares compare how a model spreads its attention, not absolute effect sizes.
    """
    imp = importance.loc[importance.mean(axis=1).sort_values().index]
    n = imp.shape[1]
    h = 0.8 / n
    y = np.arange(len(imp))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(imp) + 1.4))
    for k, name in enumerate(imp.columns):
        ax.barh(y + (k - (n - 1) / 2) * h, imp[name], height=h, label=name,
                color=model_style(name)["color"])
    ax.set_yticks(y, imp.index)
    ax.set(title=title, xlabel="share of total mean |SHAP| (per model, sums to 1)")
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right")
    fig.tight_layout()
    return fig

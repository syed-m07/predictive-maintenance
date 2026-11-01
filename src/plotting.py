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

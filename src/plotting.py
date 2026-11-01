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

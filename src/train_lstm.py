"""Train, evaluate and register an LSTM-with-attention RUL model (Bi-LSTM or uni-LSTM, P7).

    python -m src.train_lstm --variant bi                                # default config, Bi-LSTM
    python -m src.train_lstm --variant uni                               # the ablation
    python -m src.train_lstm --variant bi --params-json configs/lstm_bi_best_params.json --model-name lstm_bi_tuned

Same steps as the TCN (src/train_sequence.py), plus the attention heatmap.
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.config import SEQ_LEN, SUBSET
from src.models.lstm_rul import LSTMRUL
from src.plotting import BLUE
from src.train_sequence import run_sequence_model, val_window_frame

EXPERIMENT = "rul-lstm"
ARCH_KEYS = {"hidden_size", "num_layers", "dropout", "bidirectional", "attn_dim"}
LIFE_POINTS = (150, 100, 75, 50, 25, 0)          # cycles before failure shown in the heatmap


def plot_attention(model: LSTMRUL, d, X_val, u_val, model_name: str) -> dict:
    """Attention heatmap for the longest-lived validation engine.

    How to read: each ROW is one prediction moment in the engine's life (top = early, bottom
    = at failure); each COLUMN is one cycle inside that 30-cycle input window (left = oldest,
    right = newest). Darker = more attention weight (each row sums to 1; uniform would be
    1/30 = 0.033 per cell). The label on the right is the predicted / target RUL.
    Question it answers: which recent cycles drive the prediction, and does that change as the
    engine approaches failure?
    """
    frame = val_window_frame(d, u_val)
    engine = d.val.groupby("unit_id")["cycle"].max().idxmax()
    rows = np.where(frame["unit_id"].to_numpy() == engine)[0]
    last_cycle = frame.loc[rows, "cycle"].max()
    picks = []
    for before in LIFE_POINTS:
        target_cycle = last_cycle - before
        if target_cycle >= SEQ_LEN:
            picks.append(rows[np.argmin(np.abs(frame.loc[rows, "cycle"].to_numpy() - target_cycle))])
    W = model.attention_weights(X_val[picks])
    preds = np.clip(model.predict(X_val[picks]), 0, 125)

    cmap = LinearSegmentedColormap.from_list("attn", ["#eef4fc", BLUE, "#123c6e"])
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(picks) + 1.8))
    im = ax.imshow(W, aspect="auto", cmap=cmap, vmin=0, vmax=max(W.max(), 2 / SEQ_LEN))
    ax.set_yticks(range(len(picks)),
                  [f"{last_cycle - frame.loc[i, 'cycle']} cycles before failure" for i in picks])
    ticks = list(range(0, SEQ_LEN - 1, 5)) + [SEQ_LEN - 1]
    ax.set_xticks(ticks, [f"t-{SEQ_LEN - 1 - c}" if c < SEQ_LEN - 1 else "now" for c in ticks])
    ax.set_xlabel("cycle inside the 30-cycle input window (oldest -> newest)")
    right = ax.twinx()                                  # predicted / target RUL per row, right side
    right.set_ylim(ax.get_ylim()); right.set_yticks(range(len(picks)),
        [f"pred {preds[r]:.0f} / target {frame.loc[i, 'rul']:.0f}" for r, i in enumerate(picks)])
    right.tick_params(length=0); right.grid(False)
    for s in right.spines.values():
        s.set_visible(False)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=[ax, right], fraction=0.025, pad=0.13,
                        label="attention weight (black line = uniform, 1/30)")
    cbar.ax.axhline(1 / SEQ_LEN, color="#0b0b0b", lw=1.5)        # uniform attention reference
    ax.set_title(f"{model_name}: attention over the input window, validation engine {engine}")
    return {"attention": fig}


def run(subset: str = SUBSET, params: dict | None = None, model_name: str | None = None,
        notes: str = "", device: str | None = None, verbose: bool = True, variant: str | None = None) -> dict:
    params = dict(params or {})
    if variant is not None:
        params["bidirectional"] = variant == "bi"
    params.setdefault("bidirectional", True)
    model_name = model_name or ("lstm_bi" if params["bidirectional"] else "lstm_uni")
    return run_sequence_model(LSTMRUL, experiment=EXPERIMENT, arch_keys=ARCH_KEYS, subset=subset,
                              params=params, model_name=model_name, notes=notes, device=device,
                              verbose=verbose, extra_figures=plot_attention,
                              extra_params=lambda m: {"n_parameters": sum(p.numel() for p in m.net.parameters())})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["bi", "uni"], required=True)
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--params-json", help="JSON with arch (hidden_size, num_layers, dropout) and training params (lr, ...)")
    ap.add_argument("--model-name", default=None, help="default: lstm_bi / lstm_uni")
    ap.add_argument("--device", default=None)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    params = json.loads(open(a.params_json).read()) if a.params_json else None
    run(a.subset, params, a.model_name, a.notes, a.device, variant=a.variant)


if __name__ == "__main__":
    main()

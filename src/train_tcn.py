"""Train, evaluate and register a TCN RUL model (CPU locally, or GPU on Colab).

    python -m src.train_tcn                                              # default config
    python -m src.train_tcn --params-json configs/tcn_best_params.json --model-name tcn_tuned

The steps are shared with the LSTMs (src/train_sequence.py); TCN-specific: the receptive
field is logged.
"""
from __future__ import annotations

import argparse
import json

from src.config import SUBSET
from src.models.tcn_rul import TCNRUL
from src.train_sequence import build_seq_inputs, run_sequence_model  # noqa: F401  (re-exported)

EXPERIMENT = "rul-tcn"
ARCH_KEYS = {"channels", "kernel_size", "dropout"}


def run(subset: str = SUBSET, params: dict | None = None, model_name: str = "tcn",
        notes: str = "", device: str | None = None, verbose: bool = True) -> dict:
    return run_sequence_model(TCNRUL, experiment=EXPERIMENT, arch_keys=ARCH_KEYS, subset=subset,
                              params=params, model_name=model_name, notes=notes, device=device,
                              verbose=verbose,
                              extra_params=lambda m: {"receptive_field": m.net.receptive_field})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--params-json", help="JSON with arch (channels, kernel_size, dropout) and training params (lr, ...)")
    ap.add_argument("--model-name", default="tcn")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    params = json.loads(open(a.params_json).read()) if a.params_json else None
    run(a.subset, params, a.model_name, a.notes, a.device)


if __name__ == "__main__":
    main()

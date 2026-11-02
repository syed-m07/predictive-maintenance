"""Tune an LSTM-with-attention variant with Optuna (50 trials on Colab's T4), then train and
register <variant>_tuned. Both variants get the identical search (decision P7).

    python -m src.train_lstm_optuna --variant bi      # -> configs/lstm_bi_best_params.json, lstm_bi_tuned
    python -m src.train_lstm_optuna --variant uni     # -> configs/lstm_uni_best_params.json, lstm_uni_tuned

See src/deep_tuning.py for the rules (validation-only objective, median pruning, resumable).
"""
import argparse

from src.config import SUBSET
from src.deep_tuning import tune


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["bi", "uni"], required=True)
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--n-trials", type=int, default=50, help="TOTAL trials in the study (resumes)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-final", action="store_true")
    a = ap.parse_args()
    tune(f"lstm_{a.variant}", a.subset, a.n_trials, a.epochs, not a.no_final, a.device)


if __name__ == "__main__":
    main()

"""Tune the TCN with Optuna (50 trials on Colab's T4), then train and register tcn_tuned.

    python -m src.train_tcn_optuna                     # 50 trials (resumes if interrupted)
    python -m src.train_tcn_optuna --n-trials 2 --epochs 2 --no-final   # smoke test

See src/deep_tuning.py for the rules (validation-only objective, median pruning, resumable).
"""
import argparse

from src.config import SUBSET
from src.deep_tuning import tune


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--n-trials", type=int, default=50, help="TOTAL trials in the study (resumes)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-final", action="store_true")
    a = ap.parse_args()
    tune("tcn", a.subset, a.n_trials, a.epochs, not a.no_final, a.device)


if __name__ == "__main__":
    main()

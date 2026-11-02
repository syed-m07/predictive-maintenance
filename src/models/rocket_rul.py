"""ROCKET (MiniRocket) + ridge regression on the 30x14 sensor windows.

How it works:
  1. MiniRocket (Dempster et al. 2021, the fast variant of ROCKET 2020) convolves every window
     with ~10,000 fixed kernels. The kernels are not learned: their weights come from a small
     fixed set, and their dilations and biases are set once from the training windows. Each
     kernel gives one number per window, the "proportion of positive values" (PPV): how
     often the convolution output exceeded the bias. The result is ~10,000 features per window.
  2. A ridge regression (a linear model with an L2 penalty `alpha`) maps those features to RUL.
     The only thing that is "trained" is that linear layer, which is why ROCKET trains in a
     fraction of the time a deep model needs.

Choice of alpha (decisions K2-K3 in docs/DECISIONS.md): the penalty is chosen from
np.logspace(-3, 6, 10) by RMSE on the 20 VALIDATION ENGINES, the same data every other model
uses for early stopping / Optuna. We do not use RidgeCV's default (leave-one-out over training
windows): neighbouring windows of one engine overlap by 29 of 30 cycles, so leaving one window
out leaks its neighbours and favours too little regularisation.

X^T X is computed once and each alpha is one Cholesky solve, so the 10-alpha search costs
little more than a single fit. The fitted coefficients are stored as plain numbers (npz); only the
MiniRocket transformer itself is stored with joblib (like the scaler; trusted local files only).
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.config import MAX_RUL, SEED
from src.models.base import RULModel, read_metadata, write_metadata

TRANSFORMER_FILE = "minirocket.joblib"
RIDGE_FILE = "ridge.npz"
DEFAULT_ALPHAS = tuple(float(a) for a in np.logspace(-3, 6, 10))   # one per decade; see DECISIONS K3 for why not (-3, 3)


def ridge_path(F: np.ndarray, y: np.ndarray, alphas) -> tuple[np.ndarray, float]:
    """Ridge solutions for every alpha on standardised, centred features F (n, p).

    Returns (coefs (n_alphas, p), intercept). Centering F and y makes the intercept mean(y),
    unpenalised (as in sklearn). X^T X is formed once; each alpha is then one Cholesky solve
    of (X^T X + alpha I) c = X^T y (about 10x cheaper than an eigendecomposition here)."""
    y_mean = float(y.mean())
    G = F.T @ F                                   # (p, p), the expensive product, done once
    Fy = F.T @ (y - y_mean)
    diag = np.arange(G.shape[0])
    coefs = []
    for a in alphas:
        G[diag, diag] += a
        coefs.append(cho_solve(cho_factor(G, lower=True, check_finite=False), Fy, check_finite=False))
        G[diag, diag] -= a
    return np.stack(coefs), y_mean


class ROCKETRUL(RULModel):
    model_type = "rocket"
    input_kind = "sequence"

    def __init__(self, num_kernels: int = 10_000, alphas=DEFAULT_ALPHAS, seed: int = SEED):
        self.num_kernels = num_kernels
        self.alphas = tuple(float(a) for a in alphas)
        self.seed = seed
        self.transformer = None
        self.n_features: int | None = None
        self.seq_len: int | None = None
        self.feat_mean = self.feat_scale = self.coef = None
        self.intercept: float | None = None
        self.alpha: float | None = None
        self.alpha_scores: dict[float, float] = {}      # alpha -> validation RMSE
        self.timings: dict[str, float] = {}             # seconds: transform_s, ridge_s

    # ---- helpers
    def _check(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"expected (n, seq_len, n_features) windows, got {X.shape}")
        if self.n_features is not None and X.shape[1:] != (self.seq_len, self.n_features):
            raise ValueError(f"expected (n, {self.seq_len}, {self.n_features}), got {X.shape}")
        return X

    def _features(self, X: np.ndarray) -> np.ndarray:
        """(n, seq_len, channels) -> sktime's (n, channels, timepoints) -> standardised MiniRocket features."""
        F = np.asarray(self.transformer.transform(np.ascontiguousarray(X.transpose(0, 2, 1))), dtype=np.float64)
        return (F - self.feat_mean) / self.feat_scale

    # ---- RULModel interface
    def fit(self, X, y, X_val=None, y_val=None) -> "ROCKETRUL":
        from sktime.transformations.panel.rocket import MiniRocketMultivariate   # heavy import, only to train
        X = self._check(X)
        y = np.asarray(y, dtype=np.float64)
        self.seq_len, self.n_features = X.shape[1], X.shape[2]
        t0 = time.perf_counter()
        self.transformer = MiniRocketMultivariate(num_kernels=self.num_kernels, random_state=self.seed)
        F = np.asarray(self.transformer.fit_transform(np.ascontiguousarray(X.transpose(0, 2, 1))), dtype=np.float64)
        self.feat_mean = F.mean(axis=0)
        sd = F.std(axis=0)
        self.feat_scale = np.where(sd > 1e-12, sd, 1.0)          # constant features stay 0
        F = (F - self.feat_mean) / self.feat_scale
        self.timings["transform_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        coefs, self.intercept = ridge_path(F, y, self.alphas)
        del F

        if X_val is not None:
            F_val = self._features(self._check(X_val))
            y_val = np.asarray(y_val, dtype=np.float64)
            for a, c in zip(self.alphas, coefs):
                pred = np.clip(F_val @ c + self.intercept, 0, MAX_RUL)   # scored the way the test is
                self.alpha_scores[a] = float(np.sqrt(np.mean((pred - y_val) ** 2)))
            best = min(self.alpha_scores, key=self.alpha_scores.get)
        else:
            best = self.alphas[len(self.alphas) // 2]               # no validation data: middle of the grid
        self.alpha = best
        self.coef = coefs[self.alphas.index(best)]
        self.timings["ridge_s"] = time.perf_counter() - t0   # includes the validation transform
        return self

    def predict(self, X) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("model is not fitted")
        return (self._features(self._check(X)) @ self.coef + self.intercept).astype(np.float32)

    def save(self, bundle_dir: Path | str, metadata: dict | None = None) -> Path:
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.transformer, bundle_dir / TRANSFORMER_FILE)
        np.savez(bundle_dir / RIDGE_FILE, coef=self.coef, feat_mean=self.feat_mean,
                 feat_scale=self.feat_scale, intercept=np.array(self.intercept))
        write_metadata(bundle_dir, {
            "model_type": self.model_type, "input_kind": self.input_kind,
            "num_kernels": self.num_kernels, "n_rocket_features": int(self.coef.shape[0]),
            "alphas": list(self.alphas), "alpha": self.alpha,
            "alpha_val_rmse": {str(k): v for k, v in self.alpha_scores.items()},
            "seq_len": self.seq_len, "n_features": self.n_features, "seed": self.seed,
            **(metadata or {})})
        return bundle_dir

    @classmethod
    def load(cls, bundle_dir: Path | str) -> "ROCKETRUL":
        bundle_dir = Path(bundle_dir)
        meta = read_metadata(bundle_dir)
        if meta["model_type"] != cls.model_type:
            raise ValueError(f"bundle holds a {meta['model_type']} model, not rocket")
        obj = cls(num_kernels=meta["num_kernels"], alphas=meta["alphas"], seed=meta["seed"])
        obj.transformer = joblib.load(bundle_dir / TRANSFORMER_FILE)
        r = np.load(bundle_dir / RIDGE_FILE, allow_pickle=False)
        obj.coef, obj.feat_mean, obj.feat_scale = r["coef"], r["feat_mean"], r["feat_scale"]
        obj.intercept = float(r["intercept"])
        obj.alpha = meta["alpha"]
        obj.alpha_scores = {float(k): v for k, v in meta["alpha_val_rmse"].items()}
        obj.seq_len, obj.n_features = meta["seq_len"], meta["n_features"]
        return obj

# Decision Log

Every non-obvious choice in this project, **why** it was made, and what was deliberately *not* done.
Newest entries at the bottom of each section. Shortcuts and scope changes are tracked separately
in [`DEVIATIONS.txt`](DEVIATIONS.txt) and need owner approval first.

Sources of truth, in order: this file → [`HANDOFF.md`](HANDOFF.md) → [`pm_planner_v4.html`](pm_planner_v4.html)
(v4 = corrected copy of `pm_planner_fixed.html`; same task IDs, so browser ticks carry over).

---

## 1. Project-level decisions (owner, 2026-11-01)

| # | Decision | Why |
|---|---|---|
| P1 | Code is written directly in `src/`. Notebooks only *import* from `src/` and render showcase plots. | Avoids the copy-paste drift between Colab and the repo; `src/` is the single source of truth. |
| P2 | **Heavy training (TCN/LSTM Optuna, FD002–FD004 deep models) runs on Google Colab's T4 GPU**, using the same `src/` code cloned from GitHub. See [`COLAB.md`](COLAB.md). | No local NVIDIA GPU. Colab lets us keep the original trial counts instead of cutting them. |
| P3 | Test metrics are reported against NASA's `RUL_FD00X.txt` **as given (unclipped)**. Only *training/validation labels* are clipped at 125. | Stays faithful to the benchmark's true goal. The README states the convention explicitly, because many papers clip the test truth too, and those numbers aren't directly comparable. |
| P4 | **All four models (XGBoost, TCN, Bi-LSTM+attention, ROCKET) are evaluated end to end and all can be served by the API.** Which model(s) stay in the Docker image is decided from the final comparison table, and unused ones are removed *then* (logged here). | Pick the serving model on evidence, not up front. Consequence: the API's model loader must be model-agnostic, and the first Docker image includes CPU torch. |
| P5 | No shortcuts. Original scope is kept (100 XGBoost trials, 50 TCN trials, 50 LSTM trials, Bi-LSTM, SHAP on XGBoost **and** the best deep model, Optuna plots, multi-model plot, demo, paper outline). | Resume-grade project. See DEVIATIONS.txt DV-001. |

## 2. Environment decisions

| # | Decision | Why |
|---|---|---|
| E1 | **Python 3.12** (not the machine's default 3.15). The venv lives at `predictive-maintenance/venv/`. | Verified with pip's resolver: `optuna` and `mlflow` have no installable dependency chain on 3.15 yet. Every package resolves on 3.12.10. |
| E2 | Exact version pins in `requirements.txt`, verified by `pip check` plus a smoke test (each library runs a tiny real computation). | Reproducibility. "Pin major versions" (original plan) is too loose to reproduce results. |
| E3 | `torch` CPU build locally (via the PyTorch CPU index). | No NVIDIA GPU here; the CUDA build is about 2.5 GB of unused code. Colab provides its own CUDA torch. |
| E4 | Added `numba==0.68.1`. | Found by the smoke test: sktime's `MiniRocketMultivariate` refuses to run without numba, and sktime doesn't install it. Compatible with numpy 2.5.4 (no downgrade). |
| E5 | MLflow backend = **SQLite** (`mlflow.db`), artifacts in `./mlruns/`. Overridable via the `MLFLOW_TRACKING_URI` env var. | MLflow 3.17 refuses the plain-folder `./mlruns` store ("maintenance mode"). SQLite is the officially recommended local backend. The handoff already listed it as an option. |
| E6 | Separate `requirements-serve.txt` for the Docker image. | The API needs about 10 packages, not about 25. A smaller image builds faster and has a smaller attack surface. Torch is added to it if a deep model is served (see P4). |
| E7 | Constants in `src/config.py` (not `configs/config.yaml`). | Importable, type-checked by the editor, one file. `configs/` holds only *generated* artifacts (best-params JSON). |

## 3. Data decisions

| # | Decision | Why |
|---|---|---|
| D1 | Data comes from Kaggle `behrad3d/nasa-cmaps` v1 via kagglehub; SHA-256 checksums are in `data/sha256sums.txt`. | The NASA dspace URL in the old manifest is dead. Checksums let anyone confirm they have byte-identical files. |
| D2 | `x.txt` and the PDF from the Kaggle folder are **not** copied to `data/raw/`. | `x.txt` (218 numbers) isn't part of NASA's dataset and matches no RUL file. The PDF is the reference paper (copyrighted). |
| D3 | Read files with `sep=r"\s+"`. | Verified: `sep=" "` produces 28 columns (2 all-NaN) because of trailing spaces. |
| D4 | Trust file contents over NASA's readme for engine counts. | Verified: FD004 has 249 train / 248 test engines; the readme (and HANDOFF.md §4) says the reverse. |

## 4. Corrections to the original plan (not shortcuts: the original was wrong)

Full list in HANDOFF.md §7; all applied to planner v4.
- Per-engine normalisation → **one global scaler fit on training engines only** (per-engine is leaky and impossible at test time).
- `train_FD001` shape is `(20631, 26)`, not `(17731, 26)` (verified).
- Rolling features use `min_periods=1` (no forward-fill); all features are trailing / past-only.
- Test protocol: **last window / last row per test engine** vs `RUL_FD00X.txt`.
- Bi-LSTM caveat restated correctly (fine on a past window, not usable for strict streaming).
- FastAPI `lifespan` instead of the deprecated `@app.on_event("startup")`.
- Docker base `python:3.12-slim` (matches E1), not 3.10.
- Planner's NASA-score "≈3× penalty" wording replaced with exact values.

## 5. Repository hygiene

| # | Decision | Why |
|---|---|---|
| R1 | Untracked the empty placeholder files `trained_models/xgboost_v1.pkl`, `trained_models/lstm_v1.pt` and `mlruns/.gitkeep`. | They were committed before `.gitignore` existed, so git kept tracking them; the ignore rules never applied. Model files are build outputs, and MLflow creates `mlruns/` itself. |
| R2 | `.gitignore`: `trained_models/*` except `model_registry.csv`; fixed `.ipynb_checkpoints/` (the dot was missing); added `mlflow.db`, `.pytest_cache/`. | The registry CSV *is* the versioned record of which model produced which numbers; weights are reproducible. |
| R3 | Commits: conventional prefixes, and the body explains what changed and why. No AI co-author trailer. | The commit history is part of the portfolio and the debugging trail. |

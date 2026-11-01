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
| P6 | **Test metrics are reported twice:** (a) **primary** = against `RUL_FD00X.txt` as given (P3); (b) **secondary, "literature-comparable"** = the same predictions scored against the truth clipped at 125. Both columns go in `model_comparison.csv` and the README, clearly labelled. Implemented in `src/evaluation.py` (Block 4). | Many published C-MAPSS results clip the test truth too, so (b) lets reviewers compare our numbers to papers like for like. It's purely additive: no metric is replaced. Approved by owner 2026-11-02. |
| P7 | **Add a unidirectional LSTM (with the same attention head) as an ablation of the Bi-LSTM.** Same code path (`bidirectional=False`), same data, same Optuna budget, same tuning protocol, same scoring. Both appear in the comparison table. Approved by owner 2026-11-02. | Bidirectionality was an assumption in the original plan, not a proven benefit. Unidirectional LSTMs are the default for streaming / forecasting. Testing it turns "we used Bi-LSTM" into "we measured whether bidirectionality helps on a past window: Δ = X". Low cost: one flag, one extra Colab study. A uni-LSTM can also serve a streaming setting, which the Bi-LSTM can't. |
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
| E8 | Added `nbconvert==7.17.2` and `nbformat==5.11.1`. | To re-execute notebooks top to bottom from the CLI (`jupyter nbconvert --execute`), which proves every committed notebook runs and its outputs are current. |
| E9 | `pytest.ini` sets `pythonpath = .` | So plain `pytest` from the repo root can import `src.*`, with no install step or `sys.path` hacks. |

## 3. Data decisions

| # | Decision | Why |
|---|---|---|
| D1 | Data comes from Kaggle `behrad3d/nasa-cmaps` v1 via kagglehub; SHA-256 checksums are in `data/sha256sums.txt`. | The NASA dspace URL in the old manifest is dead. Checksums let anyone confirm they have byte-identical files. |
| D2 | `x.txt` and the PDF from the Kaggle folder are **not** copied to `data/raw/`. | `x.txt` (218 numbers) isn't part of NASA's dataset and matches no RUL file. The PDF is the reference paper (copyrighted). |
| D3 | Read files with `sep=r"\s+"`. | Verified: `sep=" "` produces 28 columns (2 all-NaN) because of trailing spaces. |
| D4 | Trust file contents over NASA's readme for engine counts. | Verified: FD004 has 249 train / 248 test engines; the readme (and HANDOFF.md §4) says the reverse. |
| D5 | The loader **validates structure on every load**: 26 columns, no NaN, unit_id sorted, and cycles exactly 1, 2, 3 … inside every engine. | Rolling/lag features assume ordered, gap-free cycles. A violating file would produce silently wrong features, so it fails loudly instead. All 8 files pass (verified). |
| D6 | **Drop list decided by an explicit rule, not by eye** (`src/eda.suggest_drop_cols`): ≤ 2 distinct values **or** within-engine trend < 0.2. On FD001 it yields exactly the planned 10 columns. `tests/test_eda.py` asserts that. | Reproducible and testable. The rule isn't threshold-sensitive: the weakest kept sensor has trend 0.63; the strongest dropped one (s6) 0.09. |
| D7 | The screen uses **within-engine** trend, not just pooled correlation with RUL. | s9/s14 have weak pooled correlation (−0.32/−0.20) but strong per-engine trends (0.66/0.70). Engines start at different levels and, for s9/s14, move in different directions, so pooling hides the signal. A correlation-only screen would have wrongly dropped them. |
| D8 | **s9 and s14 kept, with a recorded caveat.** | They carry real per-engine information and are part of the benchmark's standard 14-sensor set. Caveat: they are 96 % correlated with each other, and their trend direction differs between engines (some rise sharply near failure, some drift down). Watch how SHAP weights them. |
| D9 | **Known consequence of P3 (unclipped test truth):** 11 of the 100 FD001 test engines have true RUL > 125 (max 145). Models trained on clipped labels predict ≤ 125, so up to 20 cycles of error on those engines is unavoidable. | Not a bug; documented so the RMSE isn't misread. The README states it next to the results. |

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
| R4 | Removed the empty `notebooks/Week1_EDA … Week5_LSTM.ipynb` placeholders. Notebooks follow HANDOFF §9 naming (`01_EDA`, `02_Results_and_Comparison`, `03_SHAP`, `04_Attention`). | All five were 0 bytes. They were slots for the old Colab-lab workflow, which P1 replaced. |
| R5 | Two `src/` modules beyond the HANDOFF layout: `src/eda.py` (drop-rule statistics) and `src/plotting.py` (one shared figure style). | Keeps notebooks free of logic (P1), makes the drop decision testable, and gives every figure the same colours and chrome. |
| R6 | Figures are saved to `results/*.png` and committed. Notebooks are committed **with outputs**. | The README and GitHub viewers need the images without re-running anything. Notebooks are re-executed with nbconvert before each commit, so the outputs match the code. |

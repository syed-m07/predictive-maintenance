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

## 3b. Feature-pipeline decisions (Block 3)

| # | Decision | Why |
|---|---|---|
| F1 | **One entry point, `prepare_data(subset)`**, runs load → drop → RUL → clip → split → fit scaler (train only) → apply. Every training script calls it. | Leakage usually creeps in when each script re-implements the order. One function = one order, tested once. |
| F2 | Validation = **20 random engines** (seed 42), not ids 81–100. FD001 val engines: 8 9 13 18 37 46 48 50 55 62 64 69 72 74 76 84 85 87 90 95. | Random by seed avoids any accidental ordering effect in engine ids, and stays reproducible. |
| F3 | **MinMax scaler** (not Standard). | Bounded [0, 1] inputs suit the neural nets. Data has no extreme outliers (verified in EDA). Irrelevant for XGBoost (trees don't care about scale). |
| F4 | **No absolute `cycle` feature.** All 238 features depend only on the last ≤ 30 cycles (EMA: negligible older weight, 0.7³⁰ ≈ 2·10⁻⁵). | The API receives a recent window of readings, not the engine's age. A test proves features from the last 30 rows equal features from the full history, so serving computes exactly what training computed. |
| F5 | Lags before an engine's (k+1)-th cycle use the engine's **first** reading. Rolling windows use `min_periods=1`; first-row rolling std = 0. | No NaN, and no value from the future (back-filling would use later cycles). |
| F6 | Short engines (< 30 cycles) are **left-padded by repeating their first row** at test time. Training uses only full windows (every train engine has ≥ 128 cycles). | Assumes the engine was in its earliest observed state before observation began. Uses no future data. Needed for FD002 (min 21) and FD004 (min 19). |
| F7 | **Leakage tests were mutation-checked:** the code was deliberately broken three ways (centred rolling window, scaler fit on train+val, row-level split) and the suite had to fail each time. The first run exposed a gap: the scaler-on-train+val mutation passed. A pipeline-level test was added; all three are now caught. | A test that has never failed proves nothing. |
| F8 | **Finding: rolling-std features are the weakest family** (best \|ρ\| 0.47 vs raw 0.72, rolling mean / EMA 0.76). The original plan expected rolling std to rise toward failure. In FD001 the sensor *level* drifts, but its noise does not grow. The features are **kept** (no evidence-free pruning); SHAP will show whether XGBoost uses them. | Reported honestly in `02_Feature_Engineering.ipynb` instead of repeating the planner's expectation. |

## 3c. Evaluation decisions (Block 4)

| # | Decision | Why |
|---|---|---|
| V1 | **One scorer, `src/evaluation.py`, for every model.** Tested against hand-computed values and mutation-checked (swapped late/early penalties, flipped error sign, removed clipping: each one breaks tests). | The comparison table is only fair if every model is graded by identical, verified code. |
| V2 | Predictions are **clipped to [0, 125]** before scoring. | RUL can't be negative; models are trained on labels ≤ 125, so 125 is the largest meaningful output. |
| V3 | NASA score = **sum** over engines (literature convention). It's compared only within the same test set. | It grows with the number of engines, so FD001 (100) and FD002 (259) sums aren't comparable. |
| V4 | Speed measured **two ways, on CPU, median of 50 runs after 5 warm-ups:** `latency_single_ms` (one engine, what an API caller waits) and `throughput_ms_per_item` (whole batch). Colab GPU timings are never used in the table. | Deep models are often slow per request but fast in batches; one number would hide that. The same hardware for all models keeps it fair. |
| V5 | Validation metrics (every window of the 20 held-out engines, clipped labels) drive early stopping and Optuna. **Test metrics are computed once per final model, never used for tuning.** | Tuning on test would leak test information into model selection. The two aren't directly comparable: validation covers whole lifetimes, test one truncated point per engine. |
| V6 | **Reference rows** on the FD001 test set (verified with the scorer): predict the training mean → RMSE 43.0, NASA 32,772; always predict 125 → RMSE 64.6, NASA 1,502,475; **perfect-but-capped-at-125 oracle → RMSE 3.74, NASA 14.6**. The mean predictor appears as a "no-skill" row in the comparison table. | Gives a "no skill" floor that real models must beat. It also **quantifies D9:** under the unclipped-truth convention, RMSE ≈ 3.7 is the best any clipped-label model can achieve. |

## 3d. Deep-model decisions (Blocks 7–8)

| # | Decision | Why |
|---|---|---|
| M1 | **One training loop and one model base for all deep models** (`src/models/utils.py`, `src/models/torch_base.py`, `src/train_sequence.py`). TCN, Bi-LSTM and uni-LSTM differ only in the network. | Identical optimiser, early stopping, clipping, scheduler, scoring and bundle format, so differences in results come from the architectures. The refactor was verified: the Colab TCN bundles still reproduce 16.6694 / 14.3022 exactly. |
| M2 | Deep models train on **raw scaled 30×14 windows**, not the 238 engineered features. | The point of TCN/LSTM is to learn temporal patterns themselves; the engineered features belong to XGBoost. This is what the comparison measures. |
| M3 | TCN = Bai et al. 2018 (weight-normed causal dilated convs + residuals). **Causality is tested and mutation-checked** (no chomp / chomp one step short both fail). | The chomp bug silently leaks future steps while training still "works". |
| M4 | LSTM = `nn.LSTM` → **Bahdanau additive attention** over all 30 hidden states → head on [context, query]. The query = final hidden state (Bi-LSTM: forward state at the last step ⊕ backward state at the first step, which is tested). One class, a `bidirectional` flag (P7). | Attention gives the interpretability figure; one class makes the bi/uni comparison a single-variable ablation. |
| M5 | Optuna for deep models: 50 trials, TPE (10 random startup), **MedianPruner**. Pruned trials end in MLflow as **KILLED** (with tag `state=pruned`), not FAILED. Fixed after the first TCN study showed 28 "FAILED" runs. | Pruning saves GPU time on hopeless settings; the status should say what happened. |
| M6 | **Pruning warm-up: 5 epochs for TCN, 10 for the LSTMs.** | Measured on FD001: the Bi-LSTM sits on a plateau (val RMSE ≈ 40, i.e. predicting the mean) for ~3 epochs, then breaks through (40.1 at epoch 3 → 18.8 at epoch 6). A 5-epoch warm-up could prune slow-but-good LSTM trials. The TCN study (already run) used 5 and isn't affected. |
| M7 | LSTM search space (planner w6t7): hidden 64/128/192/256, layers 1–3, dropout 0.1–0.5, lr 1e-4–1e-2 (log). **Identical for both variants.** | A fair ablation needs the same search budget and space. |

## 3e. ROCKET decisions (Block 9)

| # | Decision | Why |
|---|---|---|
| K1 | ROCKET = sktime 1.2.0 `MiniRocketMultivariate` (10,000 kernels → 9,996 PPV features, `random_state=42`) on the **same 30×14 windows** as the deep models (transposed to sktime's `(n, 14, 30)`), then a **ridge regression** on the standardised features. API checked against the installed version first (planner w7t3). | Same input as TCN/LSTM, so the comparison isolates the model. MiniRocket is the fast, deterministic successor of ROCKET (Dempster et al. 2021) and sktime's recommended variant. |
| K2 | **Ridge penalty chosen on the 20 validation engines**, not with `RidgeCV`'s default leave-one-out over training windows. | Neighbouring windows of one engine overlap by 29 of 30 cycles, so leave-one-window-out leaks the left-out window's neighbours and favours too little regularisation. Using the validation engines matches how every other model is selected (early stopping / Optuna). |
| K3 | **Alpha grid widened from the planner's `logspace(-3, 3, 10)` to `logspace(-3, 6, 10)`** (one value per decade). **Approved by the owner, 2026-11-02 (DV-003).** | Measured: with the planner grid, validation RMSE fell monotonically to the largest value (α=10⁻³: 20.80 → α=10³: 18.91), so the best α was outside the grid and ROCKET would be under-rated. With ~10,000 features and ~14,000 highly correlated windows, strong regularisation is expected. **Result:** optimum α = 10⁵, inside the new grid (10⁶ is worse); test RMSE 17.92 → 16.39 (EXPERIMENTS E5). |
| K4 | Ridge solved by hand: `XᵀX` formed once, one Cholesky solve per α (`ridge_path`, unit-tested against sklearn `Ridge`). Coefficients stored as plain numbers (`ridge.npz`); only the MiniRocket transformer uses joblib. | A prototype with one sklearn `Ridge` fit per α had not finished after ~20 min on the 4-core laptop; an eigendecomposition of XᵀX alone took 220 s. The Cholesky path is ~10× cheaper. npz cannot execute code on load, unlike pickle. |
| K5 | No Optuna study for ROCKET. Training time is reported in two parts: transform and ridge/α search. | Its only real hyperparameter (α) is searched exhaustively in fit. The HANDOFF says "log training time, it's the story"; separating the parts shows where the time goes. |

## 3f. Comparison and explainability decisions (Block 10)

| # | Decision | Why |
|---|---|---|
| C1 | `scripts/compare_models.py` **re-loads every bundle** through one loader (`src/models/loader.py`), **checks the protocol** (same train/validation engines, bundle scaler equal to the training-engine scaler), **re-scores** the test set and stops if the score differs from the registry, then **re-times every model on the local CPU**. | A comparison table is only meaningful if every row was produced under the same rules. Re-scoring catches stale or mismatched bundles, and re-timing puts all latencies on one machine (V4). |
| C2 | **95 % percentile-bootstrap intervals** over the 100 test engines: 10,000 resamples, seed 42, predictions clipped exactly as in the scorer. **Paired** differences against the best model, on identical resamples. **Approved by the owner, 2026-11-02.** | With only 100 test engines, a 0.05-cycle gap is noise. Pairing removes the shared "easy/hard engine" variation, so it can separate models whose separate intervals overlap (unit-tested). |
| C3 | "Best" = lowest test RMSE, but **any model whose paired 95 % interval contains 0 is reported as tied with it**. No single winner is claimed when several models tie. | Avoids over-reading noise in the README / paper (E4 finding 3). |
| C4 | No-skill row = always predict the **mean clipped training label (86.56)**, reproducing V6 (RMSE 43.0, NASA 32,772). | The floor every model must clear. |
| C5 | Latency covers **`model.predict` on the prepared input only**. It does not include building XGBoost's 238 features or cutting windows. Training times are reported with their hardware (`train_hardware`: local CPU vs Colab T4), because they are not comparable across the two. | Measures the models themselves, the same way for all. The feature-building cost matters for the API and is measured there (Block 11). |
| C6 | Multi-model plot (planner w7t7): **all 9 models**, colour = family, solid = tuned, dashed = default. Engines chosen by a fixed rule (≥ 150 cycles of history; true final RUL closest to 10 / 50 / 95 → engines 76, 17, 13). Each model predicts at every cycle of the engine's recorded history. True RUL there = final RUL + cycles remaining to the last record. The script asserts that the last point equals the official test prediction. | Shows how models behave over time rather than at one point. Test data is only displayed here, never used to select anything. |
| C7 | SHAP explained set = the **100 official test inputs**. XGBoost: `TreeExplainer` (exact, additivity gap < 1e-4). Deep: `GradientExplainer` with 200 training windows (seed 42) as background, **1,000 samples** per input. `tcn_tuned` is explained alongside `lstm_uni_tuned` because they are statistically tied (C3). | 1,000 samples halves the additivity gap vs 200 (~3 → ~1.3 cycles). The sensor ranking is stable across seeds (Spearman 0.99), so it is not sampling noise. Explaining both tied deep models tests whether "the best deep model" story depends on which one we picked. |
| C8 | Models are compared **per sensor**: a sensor's SHAP for one prediction = the **signed sum** over all its derived inputs (XGBoost: raw + rolling + lags + EMA; deep: all 30 time steps). Importance = mean of its absolute value, as a share of the total. | SHAP values are additive, so the sum is the sensor's net contribution. Summing before the absolute value lets opposite pushes from one sensor's features cancel, as they do in the prediction. It is the only common unit between 238 features and 30×14 windows. |
| C9 | `src/models/loader.py` imports each model class **only when its bundle type is loaded** (tested: loading XGBoost never imports torch or sktime). | The API (Block 11) serves a shortlist chosen by the owner. An image without torch must still load the tabular model. |

## 3g. Drift and retraining decisions (Block 12)

| # | Decision | Why |
|---|---|---|
| W1 | **Reference = raw sensor readings of the FD001 training engines** (the model's own data), stored as the full sample in `trained_models/reference_FD001.npz` (no pickle). Raw values, not scaled. | KS needs the values, not summary stats. Both tests depend only on the order of values, so min-max scaling changes nothing (unit-tested). |
| W2 | **KS alone is not used to raise alarms.** Measured: between two random halves of the FD001 training engines (training vs validation engines) KS flags 5/14 sensors at p < 0.05. | With ~4,000–30,000 rows and strongly correlated rows within an engine, KS detects differences far too small to matter. It answers "is there a difference?", not "is it big?". |
| W3 | **Rule: a sensor drifts if KS p < 0.05 AND PSI > 0.2; alarm if ≥ 3 sensors drift.** **Approved by the owner, 2026-11-02 (with DV-004).** | KS says the difference is real; PSI (an effect size that doesn't grow with n) says it is large. Fits the planned `flag_drift(ks_alpha, psi_threshold, min_features=3)` signature. Unit test: a tiny shift with 200,000 rows is KS-significant but does not alarm. |
| W4 | **No-drift control = FD001 training vs FD001 validation engines (DV-004).** FD001 validation vs test is reported as a harmless shift, with a cut-off diagnostic (20 seeded draws). | The planned pair alarms because test engines are cut off before failure (mostly healthy readings). The diagnostic shows this is the whole explanation (DV-004). |
| W5 | Current batch = **unlabelled** sensor readings (a subset's test split). Retraining uses the new regime's **labelled run-to-failure training engines**. | That is the realistic situation: live readings come without labels; labels exist only for engines that have since failed. |
| W6 | Retraining = **XGBoost with the tuned FD001 settings** on FD001 + FD002 training engines. Each subset keeps its own 80/20 engine split (FD001 keeps the project's validation engines), with one scaler on the pooled training engines. Same 14 sensors / 238 features, so the API input does not change. **No operating-condition normalisation** (future work). | XGBoost trains on a CPU in about a minute, which a scheduled job can afford; the deep models need a GPU. Keeping the input schema means the retrained model can be served by the same API. Condition normalisation would change the feature pipeline (a plan change), so it isn't done here. |
| W7 | An alarm on a batch from the reference's **own** subset (e.g. FD001-test) is **flagged for review, not retrained**. | There is no new regime to learn from; refitting the same data changes nothing, and an alarm isn't proof of harm (W4). |
| W8 | The retrained model's reference = **the new regime's training engines** (saved in its bundle as `reference.npz`). Next runs monitor against it with `--reference`. | Found in the smoke run: a pooled FD001+FD002 reference still alarms on FD002 batches (13 sensors, median PSI 0.29), so the job would retrain forever. Against FD002's own training engines, FD002 test gives median PSI 0.01 and no alarm. |
| W9 | The retrained model is registered with `subset = "FD001+FD002"` and **FD002 test** metrics (`notes` says so). The FD001 test metrics are in its metadata and in MLflow. | The comparison table filters `subset == "FD001"`, so it is unaffected. The registry stays one row per model with its primary target. |

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
| R4 | Removed the empty `notebooks/Week1_EDA … Week5_LSTM.ipynb` placeholders. Notebooks: `01_EDA`, `02_Feature_Engineering` (added: the planner's Week-2 feature plots), then `03_Results_and_Comparison`, `04_SHAP`, `05_Attention` (HANDOFF §9 numbering shifted by one). Every figure carries a "How to read this figure" note. | All five were 0 bytes. They were slots for the old Colab-lab workflow, which P1 replaced. |
| R5 | Two `src/` modules beyond the HANDOFF layout: `src/eda.py` (drop-rule statistics) and `src/plotting.py` (one shared figure style). | Keeps notebooks free of logic (P1), makes the drop decision testable, and gives every figure the same colours and chrome. |
| R6 | Figures are saved to `results/*.png` and committed. Notebooks are committed **with outputs**. | The README and GitHub viewers need the images without re-running anything. Notebooks are re-executed with nbconvert before each commit, so the outputs match the code. |

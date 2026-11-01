# HANDOFF — Predictive Maintenance Intelligence System (NASA C-MAPSS, RUL Prediction)

**Owner:** Syed Maaz Ahmad
**Handoff date:** 2026-11-01
**Status:** Week 1 of an original 10-week plan — repo scaffolded, dataset downloaded and understood. **No models, no pipeline code yet.**
**New constraint:** The full project must now be completed in **1–2 days**. This document compresses the 10-week plan into a prioritized 2-day build without dropping the project's core story.

> **Redacted public copy:** personal career notes removed; technical content unchanged. Some facts below were corrected later: see `docs/DECISIONS.md` (e.g. FD004 engine counts, Python 3.12, MLflow SQLite backend).
>
> **To the coding agent:** read Sections 1–6 before writing any code. Section 7 lists bugs found in the original plan — do not reproduce them. Section 8 is the build order. Sections 9–15 are specs. Inspect the actual repo first (Section 5.3) because some state below is unverified.

---

## 0. Rules of engagement for the coding agent

1. **Inspect before building.** Run `git log --oneline`, `tree -L 2`, and check whether `src/data_loader.py`, `data/data_manifest.txt`, and `notebooks/Week1_EDA.ipynb` exist and what they contain. Report what you find before changing anything.
2. **Small, explained steps.** The owner wants to understand every part — no black boxes. For each module: say what it does and *why* in 2–4 sentences, write it, run it, show the output. Avoid giant monolithic code dumps.
3. **Commit after every working unit** (one module + its quick check = one commit) and push. Use conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`). The commit history is part of the portfolio.
4. **Respect constraints:** free/open-source only, no AWS/GCP, no Kubernetes, no paid tools. Local machine + optional Google Colab T4.
5. **`src/` is the source of truth.** Notebooks import from `src/`; never the other way round.
6. **Never commit** raw data, processed data, model weights (`*.pkl`, `*.pt`, `*.joblib`), or `mlruns/`. Track model versions in `trained_models/model_registry.csv` instead.
7. **If something is ambiguous, ask** rather than inventing project decisions.

---

## 1. TL;DR

Build an end-to-end ML **engineering** system that predicts the Remaining Useful Life (RUL, in cycles) of turbofan engines from multivariate sensor time-series (NASA C-MAPSS, primarily subset FD001). It must show the **full ML lifecycle**: data loading → feature engineering → multiple model families (XGBoost, TCN, Bi-LSTM+attention, ROCKET) → Optuna tuning → MLflow tracking → evaluation incl. the NASA asymmetric score → SHAP explainability → FastAPI serving in Docker → drift detection (FD001 → FD002) with an automated retraining trigger → tests + README + demo.

It is a **systems / production engineering showcase, not an ML research contribution.** Accuracy needs to be credible, not state-of-the-art.

---

## 2. Vision and why this project exists

### 2.1 The portfolio gap it fills
The owner's existing work is centred on GenAI / agentic AI systems. This project demonstrates the **traditional ML lifecycle** end to end: tabular/time-series modeling, honest evaluation, hyperparameter tuning, experiment tracking, model serving, monitoring and drift.

### 2.2 Framing decisions
- **Systems paper, not accuracy paper.** The interesting claims are about the pipeline: multi-model comparison including latency and training time, explainability, drift detection and retraining, containerized serving.
- **LLM/GenAI layer: optional and lowest priority.** GenAI fluency is already proven elsewhere; don't spend deadline time on it.
- **Deep turbofan physics: explicitly out of scope.** Know enough to choose sensors; don't study failure physics.
- **Research angle (later, not in the 2-day scope):** a 4–6 page workshop paper, e.g. *"A Production-Ready Predictive Maintenance Pipeline for Turbofan RUL Estimation: Multi-Model Comparison, Explainability, and Drift Detection"*. Candidate venues: IEEE PHM workshops, ML-for-engineering / physical-sciences workshops at ICML/NeurIPS. Cite Saxena et al. 2008 (C-MAPSS), Bai et al. 2018 (TCN), Dempster et al. 2020 (ROCKET).

### 2.3 The interview story this project must enable
- "I split by engine ID, not randomly, to prevent leakage."
- "I optimized and reported the NASA asymmetric score, because predicting late is far more dangerous than predicting early."
- "ROCKET trains in seconds with accuracy close to deep models — that matters for retraining cost."
- "Attention weights and SHAP show what drives each prediction."
- "My FD001-trained model detected covariate shift on FD002 via KS/PSI tests and triggered retraining automatically."
- "Anyone can `docker run` the API and get predictions."

---

## 3. Decision log (what we decided, and what changed from the original proposal)

An early project proposal (Executive Summary, not in this repo) exists. **Several of its choices were later overridden.** Where they conflict, this table wins.

| Topic | Executive Summary (original) | **Current decision** |
|---|---|---|
| Cloud | AWS/GCP, p3.2xlarge GPUs, S3 | **None.** Local + Google Colab T4 free tier only |
| Orchestration | Docker + Kubernetes | **Docker only.** Kubernetes dropped |
| Serving | Cloud REST service with API gateway | **Local FastAPI in Docker** (local deployment demo) |
| Storage | PostgreSQL / S3 | **Local files + SQLite** (SQLite for MLflow backend or prediction logs if needed) |
| Models | XGBoost/RF baselines, LSTM, 1D CNN, maybe transformer | **Tuned XGBoost (baseline), TCN, Bi-LSTM with Bahdanau attention, ROCKET**; Temporal Fusion Transformer = optional stretch |
| Tuning | Optuna or grid | **Optuna (TPE)** |
| Tracking | MLflow or SageMaker | **MLflow (local)** |
| Monitoring | Prometheus + Grafana | **KS test + PSI drift detection** is core; Prometheus `/metrics` endpoint optional; Grafana stretch |
| Fairness auditing | Mentioned | **Not applicable** (non-human sensor data) — drop; frame as "model health monitoring" |
| Data augmentation to 50–100K points | Proposed | **Dropped.** Use real C-MAPSS subsets only |
| Dataset variant | FD002/FD004 | **FD001 primary** (clean benchmark); **FD002 used for drift demo**; FD003/FD004 optional |
| Target RMSE < 10 | Stated | **Unrealistic for FD001 test set.** Credible targets in Section 10.4 |
| Timeline | 6–9 months | Became a 10-week plan → **now 1–2 days** |
| Stretch | — | N-CMAPSS dataset (generalization), TFT, LLM reasoning layer, paper |

**Tooling (all confirmed free/open source):** Python, PyTorch, XGBoost, scikit-learn, Optuna, SHAP, MLflow, sktime (ROCKET), FastAPI + Uvicorn + Pydantic, Docker, pytest, pandas/numpy/scipy, matplotlib/seaborn, prometheus_client (optional). Dev: Git/GitHub, Google Colab. Dataset via `kagglehub`.

**Workflow decision:** "Colab = lab notebook, IDE/Git = production code." Explore in Colab, extract working code into `src/` immediately, commit both together. *(See Section 8.1 for the deadline-adjusted version of this.)*

---

## 4. Dataset facts (C-MAPSS)

- Simulated turbofan run-to-failure data (NASA, Saxena et al. 2008). ~50 MB. Downloaded via `kagglehub`.
- **Discoveries already made:** files sit inside a nested `CMaps/` subdirectory in the kagglehub download path; the dataset's readme must be read with `encoding="latin-1"`.
- Files per subset: `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`.
- **Format:** space-separated, no header, **trailing spaces** → read with `sep=r"\s+"`, `header=None` (otherwise you get 2 extra all-NaN columns).
- **26 columns:** `unit_id, cycle, op1, op2, op3, s1 … s21` (3 operating settings + 21 sensors; the Exec Summary's "26 sensors" is wrong).

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 | 1 (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC) |
| FD003 | 100 | 100 | 1 | 2 (HPC + fan) |
| FD004 | 248 | 249 | 6 | 2 (HPC + fan) |

- `train_FD001.txt` has **20,631 rows** (shape `(20631, 26)`); `test_FD001.txt` has 13,096 rows. *(The planner says 17,731 — that's wrong; see Section 7.)*
- **Semantics (confirmed understanding):**
  - Train: each engine runs from healthy to **failure** → RUL label = `max_cycle(unit) − cycle`.
  - Test: each engine is **truncated** at some point before failure.
  - `RUL_FD00X.txt`: one number per test engine = true remaining cycles **at the last observed test cycle**.
- **Near-constant columns in FD001 (drop):** `op3`, `s1, s5, s6, s10, s16, s18, s19` (verify via variance in EDA). `op1`/`op2` are near-constant noise in FD001 too. **Keep 14 sensors:** `s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`.

---

## 5. Current state — what's done, the last thing we did, what's next

### 5.1 Done (per project records)
- Git repo scaffolded and committed with: `src/`, `notebooks/`, `data/`, `trained_models/`, `tests/`, `mlruns/`, `configs/`, `scripts/`, `results/` (planner also specifies `src/models/`, `data/raw/`, `data/processed/`, `__init__.py` files, `.gitignore`, `requirements.txt`).
- C-MAPSS downloaded via kagglehub; EDA discoveries made (CMaps nesting, latin-1 readme, FD001–FD004 structure and differences).
- `data/data_manifest.txt` **template provided, not yet confirmed populated.**
- Dataset semantics understood (Section 4).
- 10-week interactive planner (HTML, localStorage) working in browser.

### 5.2 Last thing we did / were about to do
We were in **Week 1, finishing EDA**, specifically about to:
1. Populate `data/data_manifest.txt`.
2. In `notebooks/Week1_EDA.ipynb`: compute RUL labels, plot per-engine degradation curves, plot sensor trends for ~5 engines, correlation/variance analysis to decide which sensors to drop, write a "Key Findings" cell.
3. Write `src/data_loader.py` (`load_cmapss`, `compute_rul`) and commit it with the notebook.

The plan was then to move into a **dedicated coding chat** for week-by-week implementation sprints (this handoff replaces that).

### 5.3 Unverified — agent must check
- Whether `src/data_loader.py`, the EDA notebook, `.gitignore`, `requirements.txt` exist and what's in them.
- Whether `data/raw/` contains the C-MAPSS files (or they're only in the kagglehub cache).
- The GitHub remote URL (not recorded).
- Which planner checkboxes are ticked (stored in the owner's browser localStorage, not in the file).

---

## 6. Planner status

The attached `pm_planner_fixed.html` is the latest planner version on record: it is the **v3 data model** (storage keys `pmv3-checks`/`pmv3-notes`) and the "fixed" localStorage build that was confirmed working in the browser. It contains the full 10-week plan (STUDY/CODE/COLAB/COMMIT/DOC/SETUP tasks). A condensed version is in Appendix A. **Its week-by-week schedule is superseded by Section 8**, and a few technical details in it are wrong (Section 7).

---

## 7. Corrections to the original plan — DO NOT reproduce these

1. **Per-engine normalization is wrong (leakage + impossible at test time).** The planner says "fit a MinMaxScaler per training engine, then apply the same params to test engines." Test engines are different engines, so there are no "same params"; and per-engine min/max over a full lifetime uses *future* information that won't exist at inference.
   **Fix:** fit **one global scaler on training rows only** (MinMax or Standard), save it (`trained_models/scaler.joblib`), and apply it to validation/test/API inputs. For multi-condition subsets (FD002/FD004), use **operating-condition-wise normalization** (cluster `op1–op3` into 6 conditions, e.g. KMeans or rounding, and fit one scaler per condition) — optional, only if time allows.
2. **Row count:** `train_FD001` is `(20631, 26)`, not `(17731, 26)`. Fix any test/assert using 17731.
3. **Read with `sep=r"\s+"`**, not `sep=" "`.
4. **Test-set evaluation protocol:** evaluate on the **last available window of each test engine** vs `RUL_FD001.txt` (100 values). Don't compute RUL for test rows from `max(cycle)` — that would be wrong (test engines didn't fail).
5. **NaNs from rolling windows:** use `rolling(W, min_periods=1)` per engine instead of forward-fill hacks. All rolling/EMA/lag features must be **trailing (past-only)** and computed `groupby("unit_id")`.
6. **Bi-LSTM "inference caveat" is overstated.** Bidirectional processing *within a past window* is legitimate at inference (the whole window is already observed). The real caveat is only that it can't be used in a strictly streaming, per-timestep online setting. Phrase it that way in README/paper.
7. **Expected XGBoost RMSE "25–45" is pessimistic** once RUL is clipped at 125 and features are engineered; expect roughly mid-teens to low-20s on FD001 test. If it's above ~30, suspect a pipeline bug (scaling, split, or test protocol).
8. **Short test engines:** some test engines (especially FD004) have fewer cycles than `seq_len`. Pad at the start by repeating the first row (or use `seq_len` ≤ min length). Handle this explicitly in `generate_sequences`.
9. **`@app.on_event("startup")` is deprecated** in recent FastAPI — use the `lifespan` context manager.
10. **Docker image size:** don't install full PyTorch in the serving image if the served model is XGBoost. Use a separate `requirements-serve.txt`.

---

## 8. The 2-day execution plan

### 8.1 Workflow adjustment for the deadline
Write code **directly in `src/`** and run it locally — FD001 is small enough that XGBoost, ROCKET, and even a small TCN train on CPU in minutes. Use Colab T4 only if deep-model training is too slow locally. Notebooks become thin **showcase notebooks** that import from `src/` and render plots (EDA, results, SHAP, attention). This keeps the "src is production, notebooks are the lab" principle while removing copy-paste time. *(Owner to confirm.)*

### 8.2 Priority tiers
- **P0 — must ship (the project is incomplete without these):** data loader, feature pipeline, evaluation suite with NASA score, tuned XGBoost with MLflow + Optuna, **one** deep model (TCN), ROCKET, comparison table, SHAP on XGBoost, FastAPI + Docker, drift detection FD001→FD002 + retrain script, core tests, README with results + architecture diagram.
- **P1 — ship if on schedule:** Bi-LSTM with attention + attention heatmap (best portfolio visual — high value), Prometheus `/metrics` endpoint, Optuna for TCN (10–15 trials), multi-model prediction plot, demo video.
- **P2 — stretch / post-deadline:** TFT, FD003/FD004 experiments, condition-wise normalization, Grafana, N-CMAPSS, LLM layer, paper outline, walk-forward CV.

### 8.3 Schedule (≈ 13–15 focused hours per day)

**Day 1 — Data → Models**

| Block | Time | Work | Commit message |
|---|---|---|---|
| 1 | 1.5 h | Inspect repo; fix `.gitignore`/`requirements.txt`; put data in `data/raw/`; populate `data_manifest.txt`; `src/config.py` | `chore: finalize env, config, data manifest` |
| 2 | 1.5 h | `src/data_loader.py` + quick EDA showcase notebook (RUL curve, lifetime histogram, sensor trends, variance/correlation → drop list) | `feat: C-MAPSS loader + EDA` |
| 3 | 2 h | `src/feature_engineering.py` (drop, global scale, rolling/lag/EMA, clip RUL, sequences, test last-window) | `feat: feature pipeline` |
| 4 | 1.5 h | `src/evaluation.py` (RMSE, MAE, R², NASA score, latency) + tests for NASA score | `feat: evaluation suite with NASA score` |
| 5 | 1.5 h | `src/models/xgboost_rul.py`, `src/train_xgboost.py` with MLflow + registry | `feat: XGBoost baseline + MLflow` |
| 6 | 1 h | `src/train_xgboost_optuna.py` (50 trials), save best params | `feat: Optuna HPO for XGBoost` |
| 7 | 3 h | `src/dataset.py`, `src/models/tcn_rul.py`, `src/train_tcn.py` (early stopping, MLflow) | `feat: TCN model + training pipeline` |
| 8 | 2.5 h | **P1:** `src/models/lstm_rul.py` (Bi-LSTM + Bahdanau), `src/train_lstm.py`, attention heatmap | `feat: Bi-LSTM with attention` |

**Day 2 — Compare → Serve → Monitor → Ship**

| Block | Time | Work | Commit message |
|---|---|---|---|
| 9 | 1 h | `src/models/rocket_rul.py` (sktime) + training-time logging | `feat: ROCKET model` |
| 10 | 2 h | `scripts/compare_models.py` → `results/model_comparison.csv`, multi-model plot; SHAP notebook (TreeExplainer on XGBoost) | `feat: model comparison + SHAP` |
| 11 | 3 h | `src/api/` (schemas, model loader, main), `Dockerfile`, `requirements-serve.txt`; build, run, curl, screenshot `/docs` | `feat: FastAPI service + Docker` |
| 12 | 2.5 h | `src/monitoring.py` (KS, PSI), `scripts/simulate_drift.py`, `scripts/retrain_if_drift.py`; optional `/metrics` | `feat: drift detection + retraining trigger` |
| 13 | 1.5 h | `tests/` (loader, features, evaluation, API, monitoring); `pytest -v` green | `test: core test suite` |
| 14 | 2 h | README (results table, diagram, insights, setup), clean notebooks, demo recording | `docs: README, results, demo` |
| 15 | 1 h | Buffer, final cleanup, `git tag v1.0.0 && git push --tags` | `chore: release v1.0.0` |

**If behind schedule, cut in this order:** demo video → Bi-LSTM → `/metrics` → TCN Optuna → multi-model plot. Never cut: split-by-engine, NASA score, Docker API, drift script, README results table.

---

## 9. Target repository layout

```
predictive-maintenance/
├── README.md
├── HANDOFF.md
├── requirements.txt            # full dev/training deps
├── requirements-serve.txt      # slim API deps (fastapi, uvicorn, pydantic, xgboost, sklearn, pandas, numpy, joblib, prometheus_client)
├── Dockerfile
├── .gitignore
├── configs/
│   ├── config.yaml             # (or src/config.py) paths, sensor lists, windows, seq_len, max_rul, seeds
│   ├── xgboost_best_params.json
│   ├── tcn_best_params.json
│   └── lstm_best_params.json
├── data/
│   ├── data_manifest.txt
│   ├── raw/                    # gitignored: train/test/RUL_FD00X.txt
│   └── processed/              # gitignored
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Results_and_Comparison.ipynb
│   ├── 03_SHAP.ipynb
│   └── 04_Attention.ipynb      # P1
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── monitoring.py
│   ├── registry.py             # append/read model_registry.csv
│   ├── train_xgboost.py
│   ├── train_xgboost_optuna.py
│   ├── train_tcn.py
│   ├── train_lstm.py           # P1
│   ├── train_rocket.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_rul.py
│   │   ├── tcn_rul.py
│   │   ├── lstm_rul.py
│   │   └── rocket_rul.py
│   └── api/
│       ├── __init__.py
│       ├── schemas.py
│       ├── model_loader.py
│       └── main.py
├── scripts/
│   ├── compare_models.py
│   ├── simulate_drift.py
│   └── retrain_if_drift.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_pipeline.py
│   ├── test_evaluation.py
│   ├── test_monitoring.py
│   └── test_api.py
├── trained_models/
│   ├── model_registry.csv      # committed
│   └── *.pkl / *.pt / *.joblib # gitignored
├── results/                    # plots + comparison CSV (committed; small)
└── mlruns/                     # gitignored
```

Run scripts as modules from repo root: `python -m src.train_xgboost`.

---

## 10. Technical specifications

### 10.1 Config constants
```python
SEED = 42
SUBSET = "FD001"
COLUMNS = ["unit_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
DROP_COLS = ["op1", "op2", "op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]  # FD001; confirm in EDA
SENSORS = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
MAX_RUL = 125           # piecewise-linear clipping
SEQ_LEN = 30            # deep models / ROCKET window
ROLL_WINDOWS = [5, 10, 30]
LAGS = [1, 5, 10]
EMA_ALPHA = 0.3
VAL_FRACTION = 0.2      # by engine ID
```
Seed everything (`random`, `numpy`, `torch`, XGBoost `random_state`, Optuna sampler `seed`).

### 10.2 Data loader
```python
def load_cmapss(subset: str, split: str, data_dir="data/raw") -> pd.DataFrame:
    # split in {"train", "test"}; sep=r"\s+", header=None, names=COLUMNS
def load_rul_truth(subset: str, data_dir="data/raw") -> np.ndarray:   # RUL_FD00X.txt, length = n test engines
def compute_rul(train_df) -> pd.DataFrame:
    # rul = groupby(unit_id).cycle.transform("max") - cycle ; assert last cycle per unit has rul == 0
```
Check: `load_cmapss("FD001","train").shape == (20631, 26)`, no NaNs.

### 10.3 Feature pipeline (order matters)
1. `drop_low_variance(df, DROP_COLS)`
2. `compute_rul` (train only) → `clip_rul(df, MAX_RUL)`
3. `split_by_engine(train_df, val_fraction, seed)` → train/val **by unit_id** (e.g. GroupShuffleSplit or ids 1–80 / 81–100)
4. `fit_scaler(train_part[SENSORS])` → **global** scaler, saved with joblib; `apply_scaler` to train/val/test/API
5. **Tabular features (XGBoost):** per-engine trailing `rolling(W, min_periods=1)` mean/std/min/max for W in ROLL_WINDOWS; lags via `groupby.shift(lag)` (fill leading NaN with the engine's first value); EMA via `groupby.transform(lambda s: s.ewm(alpha=EMA_ALPHA).mean())`. Naming: `{sensor}_{feat}_{param}`, e.g. `s11_rollstd_10`. Fill rolling std NaN (first row) with 0. Optional: `cycle` as a feature.
   ≈14 × (1 raw + 12 rolling + 3 lags + 1 EMA) = **238 features**.
6. **Sequence input (TCN / LSTM / ROCKET):** scaled 14 raw sensors only. `generate_sequences(df, seq_len, feature_cols, target="rul")` → `X (n, seq_len, n_feat)`, `y (n,)`, sliding stride 1 within each engine; target = RUL at the window's last cycle.
7. **Test inputs:** `last_window_per_engine(test_df, seq_len)` for sequences (pad short engines by repeating the first row) and `last_row_per_engine(test_feat_df)` for tabular. Ground truth = `load_rul_truth`.

### 10.4 Evaluation
```python
def nasa_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred) - np.asarray(y_true)          # d > 0 means LATE (over-estimate) → worse
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
```
- Report **sum** over test engines (standard in literature). Lower is better.
- Also RMSE, MAE, R², `time_inference(model, X, n_runs)` → ms/sample, and training time (s).
- Clip predictions to `[0, MAX_RUL]`. State in README whether test ground truth is clipped at 125 (pick one convention and use it everywhere; most papers compare against the provided RUL, with many also clipping — be explicit).
- Unit test: `nasa_score([10],[10]) == 0`; a late error of +10 scores higher than an early error of −10 (`e^1−1 ≈ 1.718` vs `e^(10/13)−1 ≈ 1.159`).

**Credible FD001 test targets (RMSE):** naive mean predictor ≈ 40+; XGBoost on engineered features ≈ 15–22; TCN/LSTM ≈ 13–18; ROCKET ≈ 14–20. Published state of the art is ~11–12. Anything < 10 is suspicious (likely leakage).

### 10.5 Models

| Model | Input | Starting config (fast) | Tuning |
|---|---|---|---|
| XGBoost | tabular (≈238 feats), last-row per engine at test | `tree_method="hist"`, `n_estimators=500`, `max_depth=6`, `lr=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, early stopping on val | Optuna 50 trials: n_estimators 100–1000, max_depth 3–12, lr 1e-3–0.3 (log), subsample 0.5–1, colsample 0.5–1, reg_alpha 0–10, reg_lambda 0–10; each trial = nested MLflow run |
| TCN | sequences (30 × 14) | channels `[32, 32, 64]`, kernel 3, dilations 1,2,4, dropout 0.2, Adam lr 1e-3, MSE, batch 256, ≤50 epochs, early stop patience 8, ReduceLROnPlateau, grad clip 1.0 | P1: Optuna 10–15 trials (channels, kernel, dropout, lr) |
| Bi-LSTM + Bahdanau (P1) | sequences | hidden 64–128, 1–2 layers, dropout 0.2; `forward` returns `(pred, attn_weights)` | optional |
| ROCKET | sequences transposed to `(n, 14, 30)` | sktime `MiniRocketMultivariate` (or `Rocket`) transform + `RidgeCV(alphas=np.logspace(-3,3,10))`; verify the sktime API for the installed version | none — log training time |
| TFT | — | **P2 only** | — |

All model classes share an interface: `fit(X, y, X_val=None, y_val=None)`, `predict(X)`, `evaluate(X, y) -> dict`, `save(path)`, `load(path)`.

TCN details: `TemporalBlock` = causal dilated `Conv1d` (left-pad `(k−1)·d`, then chomp) → ReLU → dropout ×2, plus residual (1×1 conv if channel counts differ). Take last timestep → `Linear` → scalar. Input must be permuted to `(batch, channels, time)`.

Bahdanau attention: `score = vᵀ tanh(W1·h_last + W2·h_s)`, softmax over time → `α`, context `= Σ α·h_s`, concat/FC → RUL. Plot `α` over the 30-cycle window at several points in a test engine's life → `results/attention_weights.png`.

### 10.6 MLflow
- Local tracking: `mlflow.set_tracking_uri("file:./mlruns")` (or `sqlite:///mlflow.db`). Experiments: `rul-xgboost`, `rul-tcn`, `rul-lstm`, `rul-rocket`, `drift-retraining`.
- Log params, metrics (val + test RMSE/MAE/NASA, latency, train time), artifacts (plots, best-params JSON).
- `mlflow ui` → `http://localhost:5000` (screenshot for README).

### 10.7 Model registry (`trained_models/model_registry.csv`, committed)
```
model_name,version,subset,path,val_rmse,test_rmse,test_mae,nasa_score,latency_ms,train_time_s,created_at,notes
```
`src/registry.py`: `append_entry(dict)`, `get_best(metric="test_rmse")`, `get_latest(model_name)`.

---

## 11. API spec (FastAPI)

- **Served model:** best model by test RMSE with acceptable latency. Default recommendation: **tuned XGBoost** (CPU-fast, slim image). If TCN wins clearly, serve it via TorchScript on CPU and add `torch` (CPU wheel) to the serve image.
- `src/api/model_loader.py`: load model + scaler + feature config **once** via `lifespan`, store on `app.state`.
- `src/api/schemas.py`:
  - `SensorWindow`: `unit_id: str | None`, `readings: list[list[float]]` — each row = 24 raw values `[op1, op2, op3, s1..s21]` in cycle order; validator: ≥ 1 row (≥ 30 recommended), each row length 24. The server applies the **same** drop → scale → features pipeline as training, then predicts on the last row/window.
  - `PredictionResponse`: `predicted_rul: float`, `model_name: str`, `model_version: str`, `timestamp: str`.
- Endpoints: `GET /health` → `{"status":"ok","model_loaded":true}`; `GET /metadata` → model name/version/metrics from registry; `POST /predict` → `PredictionResponse` (422 on bad shape, 500 with message on inference error); optional `GET /metrics` (prometheus_client: `prediction_count` Counter, `prediction_latency_seconds` Histogram, `drift_score` Gauge).
- **Dockerfile:** `python:3.10-slim`; copy `requirements-serve.txt` first and `pip install` (layer caching); copy `src/`, the served model file(s), scaler, `model_registry.csv`, configs; `EXPOSE 8000`; `CMD ["uvicorn","src.api.main:app","--host","0.0.0.0","--port","8000"]`. Add `.dockerignore` (data/, notebooks/, mlruns/, .git/).
  Note: model weights are gitignored, so README must say "train first (`python -m src.train_xgboost`), then `docker build`".
- Provide `scripts/sample_request.json` built from a real FD001 test engine and a curl example in README.

---

## 12. Monitoring, drift and retraining

- `src/monitoring.py`:
  - `compute_ks_drift(ref, cur) -> {"ks_stat", "p_value", "drift": p < 0.05}` (`scipy.stats.ks_2samp`)
  - `compute_psi(ref, cur, bins=10) -> float` — bins from reference quantiles, epsilon for empty bins. PSI < 0.1 stable, 0.1–0.2 moderate, > 0.2 significant.
  - `compute_feature_drift(ref_df, cur_df, features) -> DataFrame`
  - `flag_drift(drift_df, ks_alpha=0.05, psi_threshold=0.2, min_features=3) -> bool`
  - Save reference stats at training time (`trained_models/reference_stats.json` or a sample parquet).
- `scripts/simulate_drift.py`: FD001-trained model + **FD001 scaler** applied to FD002 data → per-sensor KS/PSI report (`results/drift_report.csv`, distribution plot) and the FD001 model's degraded RMSE on FD002. Expected: many sensors drift strongly (6 operating conditions = **covariate shift**). Also run FD001-val vs FD001-test as a **no-drift control** — showing the detector doesn't fire falsely is a strong talking point.
- `scripts/retrain_if_drift.py`: load reference → load current batch → detect → if drift: log alert, retrain (on combined/current data), evaluate, append registry row with `notes="retrained_after_drift"`, log to MLflow experiment `drift-retraining`. Mention in README it would run as a cron job/scheduler in production.
- Concepts to state correctly: covariate shift = P(X) changes, P(Y|X) same; concept drift = P(Y|X) changes. Drift-triggered retraining chosen over scheduled (wasteful) and performance-triggered (needs labels, which arrive late in PdM).

---

## 13. Tests (pytest)

- `test_data_loader.py`: shape `(20631, 26)`, no NaNs, RUL ≥ 0, last cycle RUL == 0, RUL truth length == number of test engines.
- `test_feature_pipeline.py`: expected feature column count; scaler fit only on train (scaled train within [0,1] for MinMax); no NaNs after features; `clip_rul` max == 125; `generate_sequences` shapes; padding for short engines; train/val engine sets disjoint.
- `test_evaluation.py`: NASA score zero for perfect, asymmetric late > early; RMSE/MAE on toy arrays.
- `test_monitoring.py`: identical distributions → no drift; shifted normal → drift; PSI ≈ 0 on identical.
- `test_api.py` (`fastapi.testclient.TestClient`): `/health` 200; valid `/predict` returns float in [0,125]; wrong row length → 422; `/metadata` has version.
  If the model file is absent in CI, skip API tests with `pytest.mark.skipif` or use a tiny fixture model.

---

## 14. README outline

1. Title, one-line description, badges (Python, Docker, tests).
2. Hero visual (attention heatmap or multi-model prediction plot).
3. Problem: RUL on C-MAPSS, why the NASA score matters.
4. Architecture diagram (Mermaid):
   Sensors → Data Loader → Feature Engineering → {XGBoost, TCN, Bi-LSTM, ROCKET} → Optuna/MLflow → Model Registry → FastAPI (Docker) → Predictions; Monitoring branch: Drift Detection (KS/PSI) → Retraining Trigger → back to Training.
5. Results: `model_comparison.csv` as a markdown table (Model | RMSE | MAE | NASA | Latency ms | Train time s), winner bolded; Key Insights (SHAP top features, attention behaviour, ROCKET speed/accuracy trade-off, drift result incl. no-drift control).
6. Design decisions: split by engine, RUL clipping at 125, global scaler fit on train, NASA score, why these four model families, Docker-only/local constraint.
7. Setup & reproduction: clone → venv → `pip install -r requirements.txt` → download data (kagglehub/NASA link) into `data/raw/` → `python -m src.train_xgboost` → `mlflow ui` → `docker build -t pm-api . && docker run -p 8000:8000 pm-api` → curl example → `pytest -v`.
8. Limitations & future work: simulated data, FD001 focus, Bi-LSTM streaming caveat, TFT/N-CMAPSS/condition-wise normalization, Grafana.
9. Demo video link.

---

## 15. Requirements

`requirements.txt` (pin versions when installing; check they resolve together):
```
numpy pandas scipy scikit-learn matplotlib seaborn
xgboost optuna mlflow shap
torch            # CPU build is fine locally
sktime           # ROCKET / MiniRocket
fastapi uvicorn[standard] pydantic
prometheus_client joblib pyyaml kagglehub
pytest httpx
```
`requirements-serve.txt`: fastapi, uvicorn, pydantic, numpy, pandas, scikit-learn, xgboost, joblib, prometheus_client (+ torch CPU only if serving a deep model).

---

## 16. Topics the owner must understand (study guide, deadline-sized)

Ordered by when they're needed. "Must" = needed to build and defend the project in interviews. Time boxes assume reading alongside building.

**Before starting (≈1.5 h total)**
1. **RUL framing (Must, 20 min):** RUL = cycles left before failure; train labels from run-to-failure; test truncated; ground truth only at last test cycle.
2. **Piecewise-linear RUL clipping at 125 (Must, 15 min):** early-life cycles show no degradation signal, so treating all RUL > 125 as 125 removes unlearnable noise; universal benchmark convention.
3. **Data leakage in time-series (Must, 30 min):** split by engine, fit scalers on train only, trailing-only windows, never peek at test truth. #1 interview topic.
4. **NASA asymmetric scoring function (Must, 15 min):** formula, why late predictions are punished harder (e^(d/10) vs e^(−d/13)).

**Day 1, while building (≈3 h)**
5. **Time-series feature engineering (Must, 30 min):** rolling mean/std/min/max, lags, EMA — what each says about degradation (rising rolling std = instability).
6. **Normalization: global vs per-engine vs per-condition (Must, 15 min):** and why per-engine is leaky (Section 7.1).
7. **Gradient boosting / XGBoost (Must, 30 min):** sequential trees fitting residuals; key hyperparameters; why it's a strong tabular baseline.
8. **Bayesian optimization with Optuna/TPE (Must, 20 min):** vs grid and random search — a crisp 2-sentence answer.
9. **MLflow basics (Must, 15 min):** experiments, runs, params/metrics/artifacts, nested runs, UI.
10. **PyTorch training loop (Must, 30 min):** Dataset/DataLoader, forward → loss → backward → step, eval with `no_grad`, early stopping, LR scheduler, gradient clipping.
11. **TCN (Must, 30 min):** causal convolutions (no future leakage), dilations → exponential receptive field, residual blocks; parallel training vs LSTM.
12. **LSTM gates + Bahdanau attention (P1, 30 min):** forget/input/candidate/output gates, cell state; attention scores → softmax → context; attention as interpretability.

**Day 2, while building (≈2.5 h)**
13. **ROCKET (Must, 20 min):** thousands of random convolution kernels → max and PPV features → ridge regression; seconds to train; the retraining-cost argument.
14. **SHAP (Must, 30 min):** Shapley values, TreeExplainer, summary vs waterfall plots; how it differs from built-in feature importance.
15. **FastAPI + Pydantic (Must, 20 min):** endpoints, request validation (422), lifespan model loading, `/docs`.
16. **Docker (Must, 20 min):** image vs container, layer caching, slim images, port mapping.
17. **Data drift (Must, 30 min):** covariate vs concept drift; KS test (p-value meaning); PSI and its thresholds; drift-triggered vs scheduled vs performance-triggered retraining.
18. **Testing ML code with pytest (Must, 15 min):** shape/invariant tests, TestClient.
19. **Prometheus metrics types (P1, 10 min):** Counter, Histogram, Gauge.

**Know at a sentence level only (if asked):** walk-forward validation (rigorous alternative to a single engine split), operating-condition normalization for FD002/FD004, TFT, N-CMAPSS, Bi-LSTM streaming caveat, model quantization/ONNX for latency.

**Explicitly skip:** turbofan failure physics, LLM layer, Kubernetes, cloud deployment, paper writing (until after the deadline).

---

## 17. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Deep models train slowly on CPU | Use Colab T4 for TCN/LSTM, or reduce epochs/channels; FD001 is small |
| sktime install/API issues | Implement MiniRocket-lite: ~2,000 random 1D kernels with numpy → max + PPV → RidgeCV |
| SHAP on deep model fails/slow | SHAP on XGBoost only (P0); attention plot covers deep-model interpretability |
| Docker build heavy | Serve XGBoost with slim requirements; no torch in image |
| XGBoost RMSE > 30 | Check: scaler leakage, split, test last-row protocol, RUL clipping, feature NaNs |
| Running out of time | Cut order in Section 8.3 |

---

## Appendix A — Original 10-week plan (condensed reference)

1. **Environment + EDA:** scaffold repo, `.gitignore`, `requirements.txt`, data manifest; study RUL framing, EDA, Saxena 2008; EDA notebook (RUL labels, 5-engine sensor plots, correlation heatmap, drop list); `src/data_loader.py`.
2. **Feature engineering:** rolling windows [5,10,30], lags [1,5,10], EMA α=0.3, RUL clip 125, sequences seq_len=30; `src/feature_engineering.py`.
3. **XGBoost baseline + MLflow + evaluation:** split by engine ID (1–80/81–100), NASA score, `src/evaluation.py`, `src/models/xgboost_rul.py`, `src/train_xgboost.py`, `model_registry.csv`.
4. **Optuna:** 100 XGBoost trials as nested MLflow runs, `configs/xgboost_best_params.json`, Optuna visualizations; study walk-forward CV.
5. **TCN:** dilated causal convs, `src/dataset.py` (RULDataset), `train_tcn.py` + `train_tcn_optuna.py` (50 trials).
6. **Bi-LSTM + Bahdanau attention:** attention heatmap as portfolio visual; `train_lstm_optuna.py` (hidden 64–256, layers 1–3, dropout 0.1–0.5, lr 1e-4–1e-2).
7. **ROCKET + SHAP + comparison:** 10k kernels + Ridge; `time_inference`; `results/model_comparison.csv`; multi-model plot; SHAP Tree/GradientExplainer.
8. **FastAPI + Docker:** `/health`, `/metadata`, `/predict`; Pydantic schemas; singleton model loader; Dockerfile; `/docs` screenshot.
9. **Monitoring + drift + retraining:** KS + PSI in `src/monitoring.py`; FD001→FD002 `simulate_drift.py`; `retrain_if_drift.py`; optional Prometheus `/metrics`.
10. **Docs, tests, paper:** README (badges, hero visual, Mermaid, results, setup), pytest suite, paper outline + venues, clean notebooks, 3–5 min demo, `v1.0.0` tag.

## Appendix B — Open questions for the owner
- GitHub repo URL / name?
- OK to write directly in `src/` and use notebooks only for showcase (Section 8.1)?
- Ground-truth clipping convention for test RMSE (clip at 125 or raw)?
- Which model to serve if TCN beats XGBoost only slightly (default: XGBoost for latency/simplicity)?

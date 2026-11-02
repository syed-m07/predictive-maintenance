# Session handoff: state of the project on 2026-11-02 (after Block 14: v1.0.0)

Written for the **next chat session** (a new assistant session or a collaborator). Read this first,
then the documents it points to. Everything here is also in the repo history; this file only
gathers it in one place.

---

## 1. Where things are documented (read in this order)

| File | What it holds |
|---|---|
| `docs/SESSION_HANDOFF.md` | This file: state, rules, open items. |
| `docs/DECISIONS.md` | **Every decision and why**: P (project), E (environment), D (data), F (features), V (evaluation), M (deep models), K (ROCKET), R (repo hygiene), C (comparison + SHAP), W (drift + retraining), A (serving), plus corrections to the original plan. Authoritative. |
| `docs/EXPERIMENTS.md` | Every result, with what we learned: reference rows (oracle, mean predictor), E1–E2 XGBoost, E3 TCN, E4 Bi- vs uni-LSTM, E5 ROCKET, E6 comparison with bootstrap intervals, E7 SHAP, E8 drift + retraining, E9 serving, E10 attention vs SHAP. |
| `docs/DEVIATIONS.txt` | Approved deviations from the original plan (DV-002: 80/20 for final models; DV-003: ROCKET α grid; DV-004: drift control + KS-AND-PSI rule; DV-005: serving shortlist). Rule: none without the owner's approval. |
| `docs/paper_outline.md` | Draft outline of the workshop paper, with every number traced to EXPERIMENTS. |
| `docs/COLAB.md` | How to train on Colab, import results, and recover from a disconnect. |
| `docs/pm_planner_v4.html` | The owner's corrected planner (the tick list). The original is `docs/pm_planner_fixed.html`. |
| `docs/HANDOFF.md` | The original project spec (redacted public copy). The full original is `docs/HANDOFF_private.md`, which is **gitignored, never commit it**. |
| `trained_models/model_registry.csv` | One row per trained model, with metrics, the git commit and the MLflow run id. |
| Outside the repo, in the parent folder `Predictive Maintenance Pipeline/` | `colab_outputs_*.zip` (Colab results), `mlflow_colab.db` (MLflow runs from Colab) + `mlflow_colab.backup_before_tagging.db`, `optuna_colab.db` (all Colab Optuna studies), `notes/`. Local `mlflow.db` and `optuna.db` are inside the repo folder but gitignored. |

---

## 2. Working rules set by the owner (follow exactly)

1. **No AI co-author / attribution lines** in commits or PRs.
2. Commits: clear conventional prefixes (`feat`, `fix`, `docs`, `test`, `chore`); the body explains **what and why**, so a later breakage can be traced. **Push after each checkpoint** (the repo is public: github.com/syed-m07/predictive-maintenance).
3. **No shortcuts.** Any deviation from the plan must be proposed first, approved by the owner, and logged in `docs/DEVIATIONS.txt`.
4. **If unsure about anything, ask first.**
5. Work in **checkpoints**. Test everything, code and installs. At each checkpoint give the owner a list of what to review, or steps to see and test it, plus **which planner tasks to tick**.
6. Keep `DECISIONS.md` / `EXPERIMENTS.md` up to date: what was done, what was not, and why.
7. Installs go **only into the repo `venv/` (Python 3.12)**, with versions checked against the pins in `requirements.txt`.
8. The owner is relearning ML: **explain** what was done and why, flag what to study, and explain how to read every figure (each figure has a "How to read" note).
9. Don't publish personal/career details (they live only in `HANDOFF_private.md`).
10. Notebooks are for showcase only; all logic lives in `src/`.

---

## 3. What is done

**All blocks (1–14) are complete; tagged `v1.0.0`. 131 tests pass.** Check `git log` for the latest commit.

| Block | Result |
|---|---|
| 1–2 Data + EDA | Loader with structure checks; checksummed data; drop rule for 10 columns → 14 sensors; `notebooks/01_EDA.ipynb` |
| 3 Features | Leakage-safe split (80 train / 20 validation engines, seed 42), scaler fit on training engines only, 238 past-only tabular features, 30×14 windows; `notebooks/02_Feature_Engineering.ipynb` |
| 4–5 Evaluation + XGBoost + MLflow | RMSE/MAE/R²/NASA (+ `_clip125` variants), CPU latency, bundles, registry, MLflow on SQLite |
| 6 Optuna XGBoost | 100 trials; only −1.6 % gain (E2) |
| 7 TCN (Colab) | causal TCN, tested; tuned 14.30 (E3) |
| 8 LSTMs (Colab) | Bi + uni with Bahdanau attention, 50 trials each; attention heatmaps (E4) |
| 9 ROCKET (local) | MiniRocket + ridge, α on validation engines (E5) |
| 10 Comparison + SHAP (local) | `scripts/compare_models.py` (protocol check, re-score, local re-timing, bootstrap CIs + paired test, multi-model plot) (E6); `scripts/shap_analysis.py` (Tree + Gradient SHAP, per-sensor agreement) (E7); `src/models/loader.py` loads any bundle (basis for the API) |
| 12 Drift + retraining (local) | `src/monitoring.py` (KS, PSI, rule KS AND PSI>0.2, ≥3 sensors), `src/drift_data.py`, `scripts/simulate_drift.py`, `scripts/retrain_if_drift.py` (E8). Registered `xgboost_retrained` v1 (subset FD001+FD002) |
| 11 API + Docker (local) | `src/api/` (schemas, model_loader, drift_monitor, main), `configs/serving.json`, `Dockerfile`, `.dockerignore`, `requirements-serve.txt`, `scripts/make_sample_request.py` (E9). Container reproduces all 300 official test predictions |
| 13 Tests | Final pass: comparison-script safety checks added (`tests/test_compare_script.py`); deliberate-bug checks done per block |
| 14 README + notebooks | `README.md`; notebooks `03_Results_and_Comparison`, `04_SHAP`, `05_Attention`, `06_Drift_and_Serving` (executed, with outputs); 01–02 re-executed; `docs/paper_outline.md` (E10) |

**Current results on the FD001 test set (100 engines, unclipped truth, sorted by test RMSE):**

| Model | Test RMSE | NASA | RMSE clip125 | Where |
|---|---|---|---|---|
| lstm_uni_tuned | **14.25** | **320.9** | 12.91 | Colab |
| tcn_tuned | 14.30 | 324.2 | 12.97 | Colab |
| lstm_bi | 14.46 | 354.8 | 13.27 | Colab |
| lstm_bi_tuned | 14.96 | 475.9 | 13.96 | Colab |
| lstm_uni | 15.20 | 400.1 | 13.89 | Colab |
| rocket | 16.39 | 425.2 | 14.93 | local CPU |
| tcn | 16.67 | 521.0 | 15.35 | Colab |
| xgboost_tuned | 17.76 | 671.3 | 16.50 | local |
| xgboost_baseline | 18.05 | 681.2 | — | local |
| (mean predictor, no skill) | 43.0 | 32,772 | — | reference |

**Headline findings so far** (details in EXPERIMENTS):
- Deep models on raw windows beat XGBoost on hand-built features: −19 % RMSE, −52 % NASA.
- **Four deep models are statistically tied for first** (paired bootstrap, E6): lstm_uni_tuned, tcn_tuned, lstm_bi, lstm_bi_tuned. Never name a single winner (C3). Deep vs XGBoost is statistically solid (+3.5 RMSE [+1.1, +5.8]).
- SHAP (E7): models agree once the correlated group s4/s7/s11/s12 is taken as one (share 0.41 / 0.37 / 0.38). The real disagreement is s3. lstm_uni_tuned's attention is flat, but SHAP shows it relies on recent cycles: attention ≠ attribution.
- Bidirectionality gives no measurable benefit on FD001. The uni-LSTM matches it with 7× fewer parameters and is causal.
- `lstm_bi_tuned` had the best validation score of all LSTMs but the worst test score of the three tuned deep models (14.96, behind even the untuned lstm_bi at 14.46): selection overfit on 20 validation engines. (An earlier version said "worst among the LSTMs"; lstm_uni at 15.20 is worse.)
- ROCKET: competitive without a GPU or a tuning study, but **not** "300× faster within 5 %". Don't put that planner claim in the README (E5).
- Seeded GPU training was exactly repeatable across two Colab machines.
- Drift (E8): control quiet, FD002 alarms with median PSI 3.0. The FD001 models break on FD002 (lstm_uni_tuned 63.6 RMSE, worse than predicting a constant, 54.1). Retraining → 30.4 (18.3 vs capped truth; the floor is 18.3 because 57 engines have RUL > 125). FD001-test also alarms (engines cut off before failure), yet the models are fine on it: an alarm is not proof of harm.

---

## 4. Problems we hit and how they were solved (so they don't recur)

| Problem | Fix / rule now |
|---|---|
| Linux `xgboost` wheel pulls 345 MB of nvidia-nccl into the image | `xgboost-cpu` in requirements-serve.txt (A8) |
| Rejected NaN input crashed the 422 response; numpy ints broke `/metadata` | Custom validation handler without input echo; registry values converted to plain Python (A4) |
| Docker engine not running | Start Docker Desktop (per-user install: `%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe`) |
| Spec's no-drift control (FD001 val vs test) alarms | Test engines are cut off before failure; control is train vs val engines (DV-004) |
| KS over-alarms on large correlated samples | Rule needs KS AND PSI > 0.2 (W3) |
| Smoke run (PM_MODELS_DIR redirected) wrote a file into the repo | Loader resolves registry paths against MODELS_DIR.parent (fixed + test) |
| Pooled reference after retraining → endless retrain loop | New reference = new regime's training engines (W8) |
| Python 3.15 can't install optuna/mlflow | venv on **Python 3.12** |
| sktime MiniRocket needs numba (not auto-installed) | `numba==0.68.1` pinned |
| MLflow 3.17 refuses a `./mlruns` file store; artifacts landed in cwd | SQLite `mlflow.db`; artifact root pinned in `src/tracking.py` |
| Git commit recorded at the end of a run / `-dirty` from the registry CSV | Commit captured at run **start**; the registry CSV is excluded from the dirty check |
| Notebook text claimed things the figure didn't show | Every claim is checked against the figure (finding F8) |
| Optuna parallel-coordinates plot unreadable | Replaced with slice plots |
| Pruned trials showed FAILED in MLflow | Now ended as KILLED (M5) |
| LSTM pruned too early during its ~3-epoch plateau | LSTM pruning warm-up of 10 epochs (M6) |
| Overnight interruption of the XGBoost study | Studies are resumable (SQLite); the stale parent run is marked KILLED |
| Colab disconnect → `No module named 'src'` | Recovery steps in `docs/COLAB.md`. **Never re-run the clone cell before checking**: it deletes the clone |
| Shell heredocs mangling quotes/backticks on Windows | Write scripts to a file, then run them |
| Ridge with sklearn per α too slow for 10k ROCKET features | Gram matrix + Cholesky path (K4) |
| Planner's ROCKET α grid too narrow | Widened, approved (DV-003) |
| `python script.py` in the scratchpad can't import `src` | Run from the repo root with `PYTHONPATH=.`, or use `python -m src.<module>` |

Verification habits that caught real bugs (keep doing them):
- **Mutation testing**: break the code on purpose and check that a test fails. This found a missing scaler-leakage test.
- **Re-scoring every imported Colab bundle locally** before trusting it.
- **Reading every figure** before committing it.

---

## 5. What remains (in order)

1. ~~Block 10: comparison + SHAP~~ **done** (E6, E7, DECISIONS C1–C9). Outputs in `results/model_comparison*.csv|png`, `multi_model_predictions.png`, `shap_*`, `test_predictions_all_models.csv`.
2. ~~Block 12: drift detection~~ **done** (E8, W1–W9, DV-004). Old plan text for reference: `src/monitoring.py` (KS test + PSI per sensor); `scripts/simulate_drift.py` for FD001 → FD002, **with a no-drift control** (FD001 validation vs FD001 test must NOT alarm); `scripts/retrain_if_drift.py`; experiment `drift-retraining`. FD002 has 6 operating conditions, so drift will be large; explain it.
3. ~~Block 11: FastAPI + Docker~~ **done** (E9, A1–A8, DV-005). Remaining: the `/docs` screenshot for the README (owner, needs a browser). Old plan text for reference: `src/api/schemas.py`, the model-agnostic loader already exists (`src/models/loader.py`, C9; reads `metadata.json`: `model_type`, `input_kind`), `main.py` (health, predict, metrics via prometheus-client), `Dockerfile` + `.dockerignore` using `requirements-serve.txt`. Torch-based models need CPU torch in the image (P4).
   - **Owner decisions (2026-11-02):** Docker Desktop **is installed**. Serve a **shortlist** of models in one API, with the model chosen per request (the owner first asked for all 9, then chose a shortlist to keep the image smaller than ~1.5–2 GB). **Open: which models are on the shortlist.** Ask the owner, using E6 (recommended: lstm_uni_tuned + tcn_tuned + xgboost_tuned; xgboost_retrained is also available for a FD002 demo). Then log it as **DV-005** (deviation from "serve one model") and drop unused dependencies from the image (no torch if no deep model, no sktime/numba if no ROCKET).
   - Proposed design (not yet confirmed by the owner): `GET /models` (shortlist + registry metrics), `POST /predict?model=<name>` (default model if omitted), `POST /predict/compare` (all shortlisted models on one input), Prometheus metrics labelled per model, ROCKET warm-up at startup if served (numba compiles on the first call).
   - Latency in E6 excludes feature building (C5). Measure XGBoost's 238-feature build per request in the API.
   - Planner w9t7 (optional Prometheus `/metrics` with `drift_score` gauge) belongs here: `src/monitoring.drift_summary` gives median/max PSI for the gauge.
4. ~~Block 13~~ **done.**
5. ~~Block 14: README~~ **done** (`README.md`). Old checklist for reference:
   - Results table with both metrics (unclipped + clip125).
   - Architecture diagram (Mermaid).
   - Key insights from E1–E7 (incl. the tied top group and the bootstrap intervals).
   - Caveats:
     - D9: the test truth is unclipped, so compare literature on clip125.
     - Bi-LSTM is not streaming-capable.
     - Latency hardware.
   - FD002–FD004 benchmarks as **future work** (the owner agreed). FD002 is used only for the drift demo.
6. ~~Showcase notebooks, paper outline, tag `v1.0.0`~~ **done**. **Left for the owner:** the `/docs` screenshot for the README (then add it under "Prediction API"), the 3–5 min demo video (w10t12), choosing target venues (w10t10; candidates in the paper outline), and the STUDY tasks.
7. Optional / future (ask first): ROCKET kernel-count speed/accuracy curve; FD002–FD004 training on Colab.

---

## 6. Things to keep in mind

- **Test metrics use the unclipped `RUL_FD001.txt` truth** (owner's decision P2/V-series). The `_clip125` columns exist only for literature comparison. Only train/validation *labels* are clipped at 125.
- **Every model uses the same protocol**: 80 training engines, the same 20 validation engines (8 9 13 18 37 46 48 50 55 62 64 69 72 74 76 84 85 87 90 95), and a scaler fit on the training engines only. Test input is one window/row per test engine, its last 30 cycles. Don't break this (DV-002).
- **Never tune on the test set.** Test is scored once per final model.
- Every training run goes through `src/run_utils.finish_run`: bundle + scaler + metadata, MLflow, and a registry row. New models must too.
- Scratch / smoke runs must not touch the real registry. Redirect output with the env vars `PM_MODELS_DIR`, `PM_RESULTS_DIR`, `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACT_ROOT` and `OPTUNA_STORAGE`.
- **Commit the code before a real training run**, so the registry row records a clean commit.
- Colab workflow: the owner runs `notebooks/colab_train.ipynb` → downloads the zip → `python scripts/import_colab_outputs.py <zip>` → re-score locally → write it up in EXPERIMENTS. The TCN/LSTM cells in that notebook are finished and commented/marked as done.
- Bundles load safely: XGBoost as native JSON, torch with `weights_only=True`, ROCKET ridge as npz. Only the scalers and the MiniRocket transformer use joblib (trusted local files).
- Windows: use the Bash tool with POSIX syntax. Run the venv Python as `venv/Scripts/python`.

---

## 7. Planner ticks up to now (`docs/pm_planner_v4.html`)

- **Weeks 1–5:** already ticked in earlier checkpoints.
- **Week 6:**
  - ✅ w6t4 (validate architecture + attention check)
  - ✅ w6t5 (attention heatmap)
  - ✅ w6t6 (lstm_rul.py)
  - ✅ w6t7 (train_lstm + optuna, both variants)
  - ✅ w6t8 (commit)
  - The STUDY tasks w6t1–w6t3 are the owner's own.
- **Week 7:**
  - ✅ w7t3 (rocket_rul.py)
  - ✅ w7t4 (inference timing; done in Block 4: `time_inference`)
  - ✅ w7t5 (commit)
  - ✅ w7t6 (comparison table; `results/model_comparison.csv`, with CIs)
  - ✅ w7t7 (multi-model plot; `results/multi_model_predictions.png`)
  - ✅ w7t8 (SHAP; XGBoost + best deep models). The planner's Week7_*.ipynb notebooks become `03_Results` / `04_SHAP` (R4), built from these scripts in Block 14.
  - The STUDY tasks w7t1–w7t2 are the owner's own.
- **Week 8:**
  - ✅ w8t3 (schemas.py)
  - ✅ w8t4 (model_loader.py)
  - ✅ w8t5 (main.py)
  - ✅ w8t6 (Dockerfile)
  - ✅ w8t7 end-to-end test done; ⬜ the `/docs` screenshot is the owner's
  - ✅ w8t8 (commit)
  - The STUDY tasks w8t1–w8t2 are the owner's own.
- **Week 10:**
  - ✅ w10t1 (README header + badges + visual), w10t2 (Mermaid), w10t3 (results), w10t4 (setup), w10t5–w10t8 (tests: 131 pass), w10t9 (paper outline), w10t11 (all local notebooks re-executed; `colab_train` is Colab-only and was already marked done), w10t13 (commit + `v1.0.0`)
  - ⬜ w10t10 (venue research) and w10t12 (demo video) are the owner's.
- **Week 9:**
  - ✅ w9t4 (`src/monitoring.py`)
  - ✅ w9t5 (`scripts/simulate_drift.py`)
  - ✅ w9t6 (`scripts/retrain_if_drift.py`)
  - ✅ w9t8 (commit)
  - ✅ w9t7 (Prometheus `/metrics`, incl. live drift gauges)
  - The STUDY tasks w9t1–w9t3 are the owner's own.

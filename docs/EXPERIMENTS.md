# Experiment Log

What each training run produced and what we learned from it. Numbers are copied from
`trained_models/model_registry.csv` (the source of truth, which also records each run's git commit and
MLflow run id). Decisions live in [`DECISIONS.md`](DECISIONS.md).

**How to read the metrics** (FD001 test set, one prediction per test engine, 100 engines):
- **RMSE / MAE**: typical error in cycles (lower is better). RMSE punishes big misses more.
- **R²**: share of the variation in true RUL that the model explains (1 = perfect, 0 = no better than the average).
- **NASA**: asymmetric penalty, summed over engines (lower is better). Late predictions cost more than early ones.
- **clip125**: the same predictions scored against truth capped at 125, comparable to papers that do this (P6).
- Floors under our convention: a perfect-but-capped model scores RMSE 3.74 / NASA 14.6. Predicting the average scores RMSE 43.0 / NASA 32,772.

---

## Reference rows (no training)

| Predictor | Test RMSE | MAE | R² | NASA | RMSE clip125 | NASA clip125 |
|---|---|---|---|---|---|---|
| Training-label mean (86.6) | 43.00 | 35.90 | −0.071 | 32,772 | 41.86 | 32,492 |
| Always 125 | 64.62 | 51.62 | −1.418 | 1,502,475 | 64.51 | 1,502,461 |
| Oracle: truth capped at 125 | 3.74 | 1.07 | 0.992 | 14.6 | 0.00 | 0.0 |

## E1 · `xgboost_baseline` v1: 2026-11-02, commit `445a914`

**Setup:** HANDOFF starting config (hist, depth 6, lr 0.05, subsample/colsample 0.8), up to 1,000 trees, early stopping 50 rounds on the 20 validation engines. 238 tabular features. Test input = the last row of each test engine.

| Val RMSE | **Test RMSE** | MAE | R² | NASA | RMSE clip125 | NASA clip125 | Latency single | Batch / engine | Train |
|---|---|---|---|---|---|---|---|---|---|
| 15.20 | **18.05** | 13.03 | 0.811 | 681.2 | 16.85 | 638.1 | 0.87 ms | 0.015 ms | 9.6 s |

**What we learned**
- **Credible, not suspicious:** within the expected 15–22 range for XGBoost on FD001 and far from the < 10 "leakage" zone. A 58 % RMSE reduction against the no-skill mean.
- Early stopping chose **139 trees** (best_iteration 138) of the 1,000 allowed. More trees would have memorised the training engines.
- **Where it errs** (`results/xgboost_baseline_test_predictions.png`): very accurate for engines close to failure (true RUL < 30). The largest misses are **late** predictions in the mid range (true RUL ≈ 30–80), which is what the NASA score penalises. Engines with true RUL > 125 are under-predicted, as expected from the cap.
- **Validation lives** (`..._val_trajectories.png`): the predicted curve follows the flat-then-descending target. Engine 74 stays ~40–50 cycles optimistic between cycles 90 and 125 before catching up.
- **Feature importance (gain):** dominated by smoothed level features (s4/s15/s11 EMA and rolling means). **No rolling-std feature is in the top 20**, consistent with finding F8. s9 appears twice (s9_ema, s9_rollmax_30), so check its role in SHAP (caveat D8).
- Latency: single-engine 0.87 ms vs 0.015 ms per engine in a batch. Per-call overhead dominates a single request (V4).
- Next: Optuna (Block 6) tunes these settings on **validation** RMSE. Test is evaluated once for the tuned model and never used for tuning.

## E2 · Optuna search + `xgboost_tuned` v1: 2026-11-02, commit `91a1b6c`

**Setup:** 100 TPE trials (20 random startup, seed 42). Objective = **validation** RMSE on the 20 held-out engines; the test set was not used. Search space per HANDOFF 10.5. Each trial trains with early stopping (50 rounds). Then `xgboost_tuned` was trained once with the best settings under the same 80/20 protocol (DV-002) and scored on test once.

**Best trial #84** (`configs/xgboost_best_params.json`): learning_rate 0.0178, max_depth 5, n_estimators 749 (early stopping used 521), subsample 0.525, colsample_bytree 0.584, reg_alpha 7.09, reg_lambda 7.06, i.e. **slow learning, shallow trees, heavy regularisation and subsampling**.

| Model | Val RMSE | **Test RMSE** | MAE | R² | NASA | RMSE clip125 | NASA clip125 | Latency single | Train |
|---|---|---|---|---|---|---|---|---|---|
| xgboost_baseline | 15.20 | 18.05 | 13.03 | 0.811 | 681.2 | 16.85 | 638.1 | 0.87 ms | 9.6 s |
| **xgboost_tuned** | **14.87** | **17.76** | 12.77 | 0.817 | 671.3 | 16.50 | 626.5 | 1.10 ms | 20.9 s |
| change | −2.2 % | **−1.6 %** | −2.0 % | | −1.5 % | −2.1 % | −1.8 % | +26 % | ×2.2 |

**What we learned**
- **Tuning helped only a little: −1.6 % test RMSE** (the original plan expected 5–15 %). The validation curve (`results/xgboost_optuna_history.png`) flattens at about 14.9 by trial ~45. After that, 55 more trials found nothing meaningfully better. The HANDOFF defaults were already near the best this model family can do with these 238 features. The remaining error is about **what the features can express**, not XGBoost settings. That's a real, reportable finding, not a failure.
- **Validation and test moved together** (val −2.2 %, test −1.6 %), so tuning on validation didn't overfit to it.
- **What mattered** (fANOVA, `..._param_importance.png`): n_estimators 85 %, learning_rate 13 %, the rest ≈ 0. Read with care: n_estimators matters because a *low* learning rate needs *many* trees to converge; trials with few trees and a small learning rate under-fit (`..._slices.png`). The "≈ 0" settings partly reflect that TPE concentrated its trials in a narrow good band (depth 4–7, subsample/colsample 0.5–0.6, high regularisation), where they barely vary.
- Cost of the gain: twice the training time, about 26 % slower single-engine inference (more, slower-learning trees).
- Search time: ~1.5 h on the laptop CPU, slowed by other jobs. Trials took seconds to over a minute (small learning rate × many trees × deep trees are slowest).

**Housekeeping (transparency):** the search was interrupted overnight after 90 completed trials (the process stopped with the laptop session). It **resumed** from `optuna.db` and ran trials 91–100. Trial 90, in progress at the interruption, stays in state RUNNING in the study and is ignored (0 effect on results). The interrupted MLflow parent run `optuna_study_FD001` (8c2c8d9d) was marked KILLED with a tag explaining why; the resumed search has its own parent run. Because the tuned model was trained after the resume, its registry row records commit `91a1b6c` (captured at run start), the code that actually ran.

## E3 · TCN on Colab T4: `tcn` v1 (default) and `tcn_tuned` v1 (50-trial Optuna), 2026-11-02, commit `3584495`

**Setup:** run by the owner with `notebooks/colab_train.ipynb` on a Colab T4 and imported with `scripts/import_colab_outputs.py`. Same data, split (identical 20 validation engines, verified) and scaler (identical, verified) as local. 30×14 windows, test = the last window of each test engine. Training: Adam + MSE, grad clip 1.0, ReduceLROnPlateau, early stopping (patience 8), ≤ 50 epochs. Optuna: 50 trials (10 random startup), objective = validation RMSE, MedianPruner → **22 completed, 28 pruned**.

**Best trial #39** (`configs/tcn_best_params.json`): channels [32, 64, 128], kernel 4, dropout 0.119, lr 0.0043, so a wider, deeper receptive field (43 cycles, covering the whole 30-cycle window) than the default (29).

| Model | Val RMSE | **Test RMSE** | MAE | R² | NASA | RMSE clip125 | NASA clip125 | Latency single* | Train (T4) |
|---|---|---|---|---|---|---|---|---|---|
| xgboost_tuned (E2, best so far) | 14.87 | 17.76 | 12.77 | 0.817 | 671.3 | 16.50 | 626.5 | 1.10 ms | 20.9 s (CPU) |
| tcn (default config) | 14.75 | 16.67 | 12.24 | 0.839 | 521.0 | 15.35 | 477.0 | 2.05 ms | 26.1 s |
| **tcn_tuned** | **13.35** | **14.30** | **10.68** | **0.882** | **324.2** | **12.97** | **282.9** | 3.03 ms | 24.3 s |

\*Timed on Colab's CPU. The final comparison (Block 10) re-times every model on the local machine (V4).

**Verification done locally (not just trusting Colab's numbers)**
- Both bundles were reloaded on the laptop CPU and test windows rebuilt from local data. Test RMSE / NASA reproduced **exactly** (14.3022 / 324.17 and 16.6694 / 520.99).
- The bundled scalers are identical to the locally fitted scaler; the validation engine lists match.
- The causality check passes on the **trained** networks (changing cycle 20 leaves cycles 0–19 untouched).
- Test RMSE 14.30 (clip125: 12.97) is well above the < 10 "leakage" zone. It sits in the range published for deep models on FD001 (best published results ≈ 11–12, clipped truth).

**What we learned**
- **The tuned TCN is the best model so far: test RMSE −19 % vs tuned XGBoost (14.30 vs 17.76), NASA score −52 % (324 vs 671).** The NASA drop is larger than the RMSE drop: the TCN makes far fewer large *late* predictions in the mid range (compare `results/tcn_tuned_test_predictions.png` with the XGBoost one). Learning patterns directly from the raw 30×14 window beats the hand-built rolling/lag/EMA features here.
- **Tuning mattered much more for the TCN than for XGBoost** (−14 % test RMSE vs −1.6 %). Optuna reached validation RMSE ≈ 13.4–13.5 by trial 11; later trials added little (`results/tcn_optuna_history.png`). Pruning stopped 28 of 50 trials early.
- **Caveat on the default `tcn` row:** it used all 50 epochs and its best epoch was 49, so validation RMSE was **still falling** when the epoch cap (HANDOFF spec: ≤ 50) stopped it (`results/tcn_learning_curves.png`). Its 16.67 is an epoch-capped result and slightly understates the default architecture. The tuned run converged properly (stopped at epoch 42, best epoch 34).
- Cost: single-engine inference ~2–3× slower than XGBoost (still ~3 ms). Final timings come from the local re-measurement.

**Housekeeping:** `tcn_tuned`'s registry row says `3584495-dirty`. The **code was clean**: the flag came from the registry CSV, which the preceding `tcn` run had appended to in the same session. Fixed in `6c69f57` (the registry file is now excluded from the check); the recorded row is left as written. In Colab's MLflow store (`mlflow_colab.db`, kept outside the repo) the 28 pruned trials show status FAILED (MLflow labels runs ended by an exception as failed) with tag `state=pruned`. That's cosmetic, and planned to be shown as KILLED from Block 8.

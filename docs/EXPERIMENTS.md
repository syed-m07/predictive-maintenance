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

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

## E4 · Bi-LSTM vs uni-LSTM with attention on Colab T4 (decision P7 ablation), 2026-11-02, commit `a267211`

**Setup:** same protocol as E3 (same windows, split, scaler, training loop, ≤ 50 epochs). Bahdanau attention over the 30-cycle window. Both variants got the **identical** 50-trial Optuna search (hidden 64–256, layers 1–3, dropout 0.1–0.5, lr 1e-4–1e-2, 10 random startup trials, MedianPruner with a 10-epoch warm-up): Bi 18 completed / 32 pruned, uni 26 completed / 24 pruned. Imported from `colab_outputs_20261102_120438.zip`.

**Best trials:** Bi #11 (hidden 256, 2 layers, dropout 0.34, lr 0.0024, **2.72 M parameters**); uni #7 (hidden 128, 3 layers, dropout 0.34, lr 0.0070, 387 k parameters). Configs in `configs/lstm_{bi,uni}_best_params.json`.

| Model | Params | Val RMSE | **Test RMSE** | MAE | R² | NASA | RMSE clip125 | NASA clip125 | Train (T4) |
|---|---|---|---|---|---|---|---|---|---|
| tcn_tuned (E3, reference) | — | 13.35 | 14.30 | 10.68 | 0.882 | 324.2 | 12.97 | 282.9 | 24.3 s |
| lstm_bi (default) | 90 k | 13.86 | 14.46 | 10.62 | 0.879 | 354.8 | 13.27 | 322.5 | 20.5 s |
| lstm_bi_tuned | 2.72 M | **12.66** | 14.96 | 11.25 | 0.870 | 475.9 | 13.96 | 447.8 | 119.6 s |
| lstm_uni (default) | 37 k | 13.98 | 15.20 | 11.52 | 0.866 | 400.1 | 13.89 | 358.9 | 16.4 s |
| **lstm_uni_tuned** | 387 k | 12.89 | **14.25** | **10.53** | **0.883** | **320.9** | **12.91** | **280.9** | 42.3 s |

Latencies are left out on purpose: they were measured on Colab's CPU and are re-measured locally for every model in Block 10.

**Verification done locally**
- All four bundles reloaded on the laptop CPU reproduce the Colab test RMSE / NASA **exactly** (4 decimals).
- The bundled scalers equal the local scaler; the validation engine lists match.
- The Optuna database (`optuna_colab.db`) holds exactly 50 trials per study.
- **Reproducibility check (accidental):** after a Colab reconnect the owner ran the notebook again on a *new* machine. The default runs were retrained from scratch and the tuned models from the saved best configs, and all four got **identical** test metrics to the first session (`colab_outputs_20261102_132051.zip`, not imported). So seeded GPU training here is repeatable, not just "close".

**What we learned**
1. **Bidirectionality brings no measurable benefit on FD001.** Default configs: Bi beats uni by 0.74 RMSE. Tuned: uni beats Bi by 0.71. The sign flips, and the gaps are the size that one or two badly predicted test engines can produce (only 100 test engines). The uni-LSTM reaches the same accuracy with 7× fewer parameters (387 k vs 2.72 M) and about 3× less training time. It is also causal, which matters for streaming use: each step only depends on the past.
2. **Validation-vs-test flip for `lstm_bi_tuned`: the best validation score (12.66) became the worst LSTM test score (14.96).** Choosing the best of 50 trials on only 20 validation engines rewards configs that happen to suit those 20 engines (a "winner's curse" / selection overfit), and the largest model (2.7 M params) is the most able to do that. The uni and TCN tuning did transfer to test. Lesson for the write-up: validation scores of tuned models are optimistic, and only the untouched test set is evidence.
3. **The top deep models are statistically indistinguishable from each other:** `lstm_uni_tuned` 14.25, `tcn_tuned` 14.30, `lstm_bi` 14.46. Calling one "the winner" would over-read noise. Block 10 should show this honestly (open decision for the owner: add bootstrap confidence intervals over the 100 test engines to the comparison table).
4. **Attention (`results/lstm_*_attention.png`) behaves very differently per model.** Note that each figure has its own colour scale, so check the colour bar:
   - **`lstm_bi`** (1 layer) concentrates attention on the newest 1–5 cycles (peak weight ≈ 0.27 against uniform 0.033). 150 cycles before failure, the attention is spread over the middle of the window instead.
   - **`lstm_uni_tuned`** (3 layers) has nearly **uniform** attention (max ≈ 0.065), i.e. it mostly averages the window. The stacked LSTM already summarises the history in its states, so attention adds little there.
   - Caveat for the write-up: attention weights show where the model *looks*, not *why* it predicts a value. They are not a full explanation (SHAP comes in Block 10).

**Housekeeping:** in `mlflow_colab.db` (outside the repo) the four session-2 runs are tagged `duplicate_of=<original run id>`. A backup taken before tagging is `../mlflow_colab.backup_before_tagging.db`. No run was left in RUNNING state. The pruned trials show as KILLED (as planned in M5).

## E5 · ROCKET (MiniRocket + ridge), local CPU, 2026-11-02, commit `3580b9d`

**Setup:** `python -m src.train_rocket`. The same 30×14 windows, split and scaler as the deep models. 10,000 MiniRocket kernels → 9,996 features → standardised → ridge. α chosen from `logspace(-3, 6, 10)` on the 20 validation engines (DECISIONS K1–K5, DV-003). Laptop CPU (4 logical cores).

**α search** (`results/rocket_alpha_search.png`): validation RMSE falls steadily from 20.80 (α=10⁻³) to **17.37 at α=10⁵**, then rises again (18.32 at 10⁶). The optimum is inside the widened grid. With the planner's grid (max 10³) a scratch run got validation 18.91 / test 17.92, so widening improved test RMSE by 1.5 cycles.

| Model | Val RMSE | **Test RMSE** | MAE | R² | NASA | RMSE clip125 | NASA clip125 | Train |
|---|---|---|---|---|---|---|---|---|
| xgboost_tuned (E2) | 14.87 | 17.76 | 12.77 | 0.817 | 671.3 | 16.50 | 626.5 | 20.9 s (CPU) |
| tcn (default, E3) | 14.75 | 16.67 | 12.24 | 0.839 | 521.0 | 15.35 | 477.0 | 26.1 s (T4) |
| **rocket** | 17.37 | **16.39** | 13.13 | 0.844 | 425.2 | 14.93 | 372.6 | **97.1 s (CPU)** = transform 9.3 s + ridge/α search 87.2 s |
| lstm_uni_tuned (E4, best) | 12.89 | 14.25 | 10.53 | 0.883 | 320.9 | 12.91 | 280.9 | 42.3 s (T4) |

**What we learned**
- **ROCKET beats both XGBoost models and the default TCN, but trails the tuned deep models by ~15 % RMSE** (16.39 vs 14.25–14.30), with no tuning beyond α. Its NASA score (425) is much better than XGBoost's (671).
- **The planner's story ("300× faster than LSTM, within 5 %") does not hold here, and the README must not claim it.** Measured: ROCKET takes 97 s on the laptop CPU, and **90 % of that is the ridge solve** over ~10,000 features (the kernel transform itself takes only 9 s). The LSTMs trained in 16–42 s, but on a T4 GPU, so the two numbers are not on the same hardware. A rough same-hardware reference: the Bi-LSTM took ~4 s/epoch on this CPU (Block 8 check), i.e. ~3 min for a 45-epoch run, so ROCKET is roughly 2× faster on CPU, not 300×. The honest, still useful MLOps point is that **ROCKET needs no GPU and no hyperparameter study** to reach a competitive score.
- Its validation RMSE (17.37) is *worse* than its test RMSE (16.39). That is the opposite of the tuned deep models, and is consistent with ROCKET having been selected on only one parameter (little selection overfit).
- Error pattern (`results/rocket_test_predictions.png`): it **over-predicts the engines closest to failure** (true RUL 10–40, predicted ~40). These are late predictions, the costly kind under the NASA score.
- Possible follow-up, not done: fewer kernels (e.g. 2,000–5,000) would cut the ridge time roughly with the square of the feature count. That is a speed/accuracy trade-off worth measuring for the paper.

## E6 · Model comparison with bootstrap intervals (Block 10), local CPU, 2026-11-02

**Setup:** `python -m scripts.compare_models` (DECISIONS C1–C6). All 9 registered models were re-loaded, protocol-checked (same split, same scaler) and re-scored. **Every re-score matched the registry.** Latency was re-measured on the laptop CPU (Intel Kaby Lake, 4 logical CPUs, torch using 2 threads). Intervals: 10,000 bootstrap resamples of the 100 test engines; paired differences against `lstm_uni_tuned`, the lowest test RMSE. Full table: `results/model_comparison.csv`.

| Model | **Test RMSE** [95 % CI] | NASA [95 % CI] | RMSE − best [paired 95 % CI] | Verdict vs best | Latency 1 engine | Params / trees |
|---|---|---|---|---|---|---|
| **lstm_uni_tuned** | **14.25** [11.93, 16.51] | **320.9** [204, 481] | — | best (point) | 2.66 ms | 387 k |
| tcn_tuned | 14.30 [11.93, 16.59] | 324.2 [204, 486] | +0.06 [−1.27, +1.19] | **tie** | 3.15 ms | 141 k |
| lstm_bi | 14.46 [12.12, 16.71] | 354.8 [219, 534] | +0.22 [−0.93, +1.39] | **tie** | 1.76 ms | 90 k |
| lstm_bi_tuned | 14.96 [12.44, 17.53] | 475.9 [230, 819] | +0.71 [−0.72, +2.21] | **tie** | 7.53 ms | 2.72 M |
| lstm_uni | 15.20 [12.77, 17.57] | 400.1 [246, 603] | +0.96 [+0.05, +1.87] | worse (barely) | 1.31 ms | 37 k |
| rocket | 16.39 [14.21, 18.42] | 425.2 [292, 588] | +2.15 [+0.25, +4.04] | worse | 1.54 ms | 10 k (ridge) |
| tcn | 16.67 [13.93, 19.31] | 521.0 [301, 804] | +2.42 [+0.76, +3.98] | worse | 2.58 ms | 32 k |
| xgboost_tuned | 17.76 [14.69, 20.67] | 671.3 [372, 1053] | +3.51 [+1.12, +5.79] | worse | 1.27 ms | 521 trees |
| xgboost_baseline | 18.05 [15.07, 20.92] | 681.2 [387, 1058] | +3.80 [+1.54, +5.98] | worse | 0.99 ms | 139 trees |
| mean predictor (no skill) | 43.00 [38.45, 47.28] | 32,772 | +28.75 | worse | — | — |

Latency = median time of `model.predict` for one engine's prepared input. It excludes feature building and windowing (C5). Training times and hardware are in the CSV; they are not comparable across the local CPU and the Colab T4.

**What we learned**
1. **Four models are statistically tied for first place:** `lstm_uni_tuned`, `tcn_tuned`, `lstm_bi` and `lstm_bi_tuned`. Each paired interval contains 0. For `tcn_tuned` the reference wins in only 56 % of resamples, a coin flip. **The README must say "a group of tied deep models", not name a single winner** (C3).
2. **Separate intervals are wide (about ±2.3 RMSE)**, wider than every gap among the top six. That is the cost of a 100-engine test set. The paired test is what separates models: ROCKET's interval overlaps the best model's, but the paired test still shows it is worse (+2.15 [+0.25, +4.04]).
3. **Deep models on raw windows beat XGBoost on engineered features, and this holds up statistically** (+3.5 RMSE [+1.1, +5.8] for the tuned XGBoost). It is the most robust finding of the project so far.
4. **Tuning helped the uni-LSTM and the TCN; for the uni-LSTM the gain only just clears noise.** `lstm_uni` vs `lstm_uni_tuned`: +0.96 [+0.05, +1.87], with the interval's lower end just above 0. `tcn` vs `tcn_tuned` is clearer (+2.42 vs the best). For the Bi-LSTM, tuning gave nothing (E4).
5. **On the NASA score the ordering is similar.** `lstm_bi_tuned` is the weakest of the tied group (+155, interval just touching 0). It makes the most costly late errors.
6. **Latency doesn't separate the models in practice.** Every model predicts one engine in 1–8 ms on a laptop CPU. The slowest is the 2.7 M-parameter `lstm_bi_tuned`. For the API, the deciding factors are accuracy, image size (torch vs not) and causality, not speed.
7. **Multi-model plot** (`results/multi_model_predictions.png`, engines 76 / 17 / 13): over each engine's history, all 9 models behave almost the same. They sit at the 125 cap while the engine looks healthy, then follow the decline. Their differences show up mainly in the last stretch. ROCKET drops to 0 at the end of engine 76 (an early, safe-side miss). On engine 13 (true RUL 95) every model stays near the cap until about cycle 120, because nothing in the data says "degrading" yet. This is the D9 ceiling in action.

## E7 · SHAP: XGBoost vs the tied best deep models (Block 10), local CPU, 2026-11-02

**Setup:** `python -m scripts.shap_analysis` (DECISIONS C7–C8). Explained: the 100 official test inputs. `xgboost_tuned` with TreeExplainer: exact, additivity gap 1e-4. `lstm_uni_tuned` and `tcn_tuned` with GradientExplainer (200 training windows as background, 1,000 samples): mean additivity gap 1.2 / 1.3 cycles, against predictions spanning 0–125. Compared per sensor via signed grouped SHAP. Numbers: `results/shap_sensor_importance.csv`, `results/shap_rank_agreement.csv`.

| Sensor | xgboost_tuned | lstm_uni_tuned | tcn_tuned |
|---|---|---|---|
| s4 | **0.184** | 0.097 | 0.079 |
| s11 | **0.143** | **0.107** | 0.084 |
| s9 | 0.107 | 0.088 | 0.094 |
| s3 | 0.094 | 0.028 | 0.024 |
| s15 | 0.081 | 0.069 | 0.065 |
| s7 | 0.038 | 0.088 | **0.109** |
| s12 | 0.041 | 0.078 | **0.108** |
| (others: s2 s8 s13 s14 s17 s20 s21) | 0.31 | 0.45 | 0.44 |

Rank agreement (Spearman ρ over 14 sensors): XGBoost–LSTM 0.54, XGBoost–TCN 0.37, LSTM–TCN 0.79. **s11 and s9 are in all three top-5 lists.**

**What we learned**
1. **The models agree on the signal once correlated sensors are taken into account.** s4, s7, s11 and s12 form one tightly correlated group (pairwise |r| 0.80–0.85 on the training engines, each |r| 0.74–0.78 with RUL). XGBoost gives most of that group's credit to s4 and s11; the deep models spread it over all four. The group's total share is almost the same: **XGBoost 0.41, LSTM 0.37, TCN 0.38**. A tree model picks one of several near-duplicate inputs and SHAP credits the one it picked. So the lower ρ with XGBoost mostly reflects how credit is split inside the group, not a different signal.
2. **One real disagreement: s3** (9.4 % for XGBoost, ~2.5 % for both deep models). It is less tied to the group (|r| ≈ 0.7). XGBoost uses its 10-cycle rolling mean (`s3_rollmean_10` is its 2nd most important feature, overall and for engine 76). The deep models find little in s3. Worth a sentence in the paper. We don't claim to know why.
3. **XGBoost's importance is concentrated; the deep models' is spread out.** For XGBoost the top two sensors carry 33 %; for the deep models the top sensor carries ≤ 11 %. Consistent with the deep models summarising all sensors together, while trees split on a few strong ones.
4. **Where in the window the deep models look** (`results/shap_deep_time_sensor.png`, shared colour scale): both weight the **most recent ~8 cycles** most, with a weaker band at the window's start. The TCN's peak is concentrated on the very last cycle, as expected from its causal convolutions ending there. **Note the contrast with E4:** `lstm_uni_tuned`'s attention was nearly uniform, yet SHAP shows it depends mostly on recent cycles. Attention weights and attributions measure different things (E4 caveat), and here they visibly disagree. Good material for `05_Attention`.
5. **Waterfall for engine 76** (`results/shap_xgboost_waterfall.png`, true RUL 10, predicted 7.1): starting from the average prediction 86.5, high recent levels of s4, s3, s9 and s11 each pull the prediction down by 3–6 cycles. The remaining 227 features together pull it down by a further 32.5. With 238 correlated features no single one dominates, which is why the per-sensor grouping (C8) is needed for a readable story.

## E8 · Drift detection FD001 → FD002 and drift-triggered retraining (Block 12), local CPU, 2026-11-02, commit `94f56eb`

**Setup:** `python -m scripts.simulate_drift`, then `python -m scripts.retrain_if_drift --batch <batch>` (DECISIONS W1–W9, DV-004). Reference: raw readings of the 80 FD001 training engines (16,390 rows). Rule: a sensor drifts if KS p < 0.05 **and** PSI > 0.2; alarm at ≥ 3 sensors. Outputs: `results/drift_report.csv` (per sensor), `drift_summary.csv`, `drift_model_impact.csv`, `drift_psi_by_sensor.png`, `drift_distributions.png`; MLflow experiment `drift-retraining`.

**Detection** (14 sensors each):

| Batch vs FD001 reference | Rows | KS p<0.05 | PSI>0.2 | Drifting (both) | Median PSI | Alarm |
|---|---|---|---|---|---|---|
| **FD001 validation engines (no-drift control)** | 4,241 | 5 | 0 | 0 | 0.005 | **no** ✔ |
| FD001 test engines (cut off before failure) | 13,096 | 14 | 11 | 11 | 0.27 | yes |
| **FD002 test engines (6 operating conditions)** | 33,991 | 14 | 14 | 14 | **3.00** | **yes** ✔ |
| Diagnostic: FD001 validation cut off like test, vs reference (20 draws) | ~2,600 | 14 | 6.3 | 6.3 | 0.20 | 14/20 |
| Diagnostic: the same, vs FD001 test engines | ~2,600 | 13 | 0 | 0 | 0.017 | 0/20 |

**Impact on the FD001 models** (FD001 scaler from their bundles; FD002 truth unclipped; NASA compared per engine because the sum grows with the engine count, V3):

| Model | FD001 test RMSE | FD002 test RMSE | FD002 RMSE vs capped truth | FD002 NASA per engine |
|---|---|---|---|---|
| floor: perfect model capped at 125 | 3.74 | 18.32 | 0 | — |
| no skill: always predict 86.6 | 43.0 | 54.05 | — | — |
| lstm_uni_tuned (deployed as-is) | 14.25 | **63.65** | 59.75 | 8,015 |
| xgboost_tuned (deployed as-is) | 17.76 | 48.13 | 37.25 | 266 |
| **xgboost_retrained** (FD001+FD002, after the alarm) | 18.24 | **30.39** | **18.31** | 74 |

79 % of the FD002 inputs fall outside [0, 1] after the FD001 scaler, i.e. outside anything the models saw in training.

**Retraining run:** the trigger fired on FD002-test (14/14 sensors) and retrained XGBoost on 288 training engines (80 FD001 + 208 FD002; validation: 20 + 52 engines; 734 trees, 53.7 s on the laptop CPU). It was registered as `xgboost_retrained` v1 (`subset = FD001+FD002`, notes `retrained_after_drift`). Its new reference (FD002 training engines) went into its bundle. Re-checking the same batch against the new reference: 0 sensors drift, median PSI 0.01, no alarm. The next scheduled run (`--reference trained_models/xgboost_retrained_v1/reference.npz`) does nothing. The two no-retrain paths were also run: FD001-val → nothing to do; FD001-test → alarm, flagged for review, not retrained (W7).

**What we learned**
1. **The detector separates the three cases cleanly.** The control is quiet. FD002's covariate shift is huge: median PSI 3.0, about 600× the control and 11× the FD001-test shift. `drift_distributions.png` shows why: six operating conditions split every sensor into separate clusters, and only one of them overlaps FD001's narrow band.
2. **KS p-values alone would have raised a false alarm on the control** (5/14 sensors). The PSI effect-size requirement is what keeps it quiet (W2–W3).
3. **"Drift" is not the same as "harm".** FD001-test alarms because its engines were cut off before failure, so their readings are mostly healthy. The diagnostic proves it: cut-off validation engines reproduce the shift and then match the test engines. Yet the FD001 models score their normal 14–18 RMSE on exactly that batch. An input-drift alarm says "the data changed", not "the model broke". That's why the job flags same-regime alarms for review instead of retraining (W7).
4. **Deploying an FD001 model on FD002 is worse than useless.** The best FD001 model (`lstm_uni_tuned`) scores 63.6 RMSE on FD002, **worse than always predicting a constant (54.1)**, with a catastrophic NASA score (8,015 per engine vs 3.2 on FD001). XGBoost degrades less (48.1). A likely reason, not tested here: a tree ensemble predicts a bounded constant for inputs beyond its training range, while a neural network's output on far out-of-range inputs is unconstrained. This is what the alarm protects against.
5. **Retraining recovers most of the loss:** FD002 RMSE 48.1 → **30.4**, and 37.2 → **18.3** against capped truth. NASA per engine drops 266 → 74. FD001 accuracy stays about the same (17.76 → 18.24; one model, so no claim either way). Part of the remaining 30.4 is unavoidable under our convention: 57 of the 259 FD002 test engines have true RUL > 125, so even a perfect capped model scores 18.3 (D9, larger here than on FD001's 3.7).
6. **A pooled reference would make the loop retrain forever** (found in the smoke run, W8). After moving to a new regime, the monitor must compare against what the model now serves.
7. **Limits:** no operating-condition normalisation (the usual FD002 step in the literature; future work), one retrained model family (XGBoost), and FD002's test split serves as both the "live" batch and the evaluation set. Fine for a demo, but in production the evaluation engines would be ones that failed later.

## E9 · Serving: FastAPI in Docker (Block 11), 2026-11-02

**Setup:** `docker build -t pm-api .` then `docker run -p 8000:8000 pm-api` (DECISIONS A1–A8, DV-005). Shortlist: `lstm_uni_tuned` (default), `tcn_tuned`, `xgboost_tuned`. Image `python:3.12-slim` + CPU torch, 1.97 GB as reported by Docker Desktop; the first build was 2.75 GB before switching to `xgboost-cpu` (A8). Container healthy ~10 s after start (models load once, plus one warm-up prediction each).

**End-to-end check over HTTP against the running container:** the raw readings of all 100 FD001 test engines were sent to `/predict/compare`. **All 300 predictions equal the official test predictions** (`results/test_predictions_all_models.csv`; largest difference 0.0005, the file's rounding). The same check runs in-process in `tests/test_api.py`, and is mutation-checked (A2).

**Per-request latency inside the container** (server-side `latency_ms`: preprocessing + prediction, full engine history sent, median over 100 engines; laptop CPU via Docker Desktop):

| Model | Median | p95 | Model-only latency (E6, `model.predict`) |
|---|---|---|---|
| lstm_uni_tuned | 5.0 ms | 7.3 ms | 2.7 ms |
| tcn_tuned | 5.5 ms | 8.3 ms | 3.2 ms |
| xgboost_tuned | **43.8 ms** | 72.8 ms | 1.3 ms |

**What we learned**
1. **For XGBoost the cost is the features, not the model.** Building its 238 rolling/lag/EMA features from the raw history takes ~40 ms per request, ~30× its prediction time. The deep models read the raw window and need only scaling. The E6 table (C5) shows model-only latency, so it ranks XGBoost fastest; in the actual service it is the slowest. Both views are reported. A production version would keep running feature state per engine instead of recomputing from history (not done; not needed at these speeds).
2. **Live drift monitoring works in the service.** After 100 FD001 test requests followed by 259 FD002 requests, `/metrics` showed `drift_score` 1.54 (median PSI) and `drift_alarm` 1.
3. **Image size is dominated by torch (~0.8 GB).** A shortlist of only `xgboost_tuned` would avoid torch entirely (the loader never imports it, C9), at the cost of the two tied-best models.

**Not done here:** the `/docs` screenshot for the README (planner w8t7) needs a browser. The owner takes it when testing (`http://localhost:8000/docs`).

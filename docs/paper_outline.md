# Paper outline (draft)

Working title: **A Production-Oriented Pipeline for Turbofan Remaining-Useful-Life Estimation: Multi-Model Comparison
with Uncertainty, Explainability, and Drift-Triggered Retraining**

Type: 4–6 page workshop paper (systems / applied ML). Candidate venues: IEEE PHM workshops; ML-for-engineering or
physical-sciences workshops at ICML/NeurIPS. The contribution is the pipeline and its honest evaluation, not a new
state-of-the-art score.

All numbers below come from `docs/EXPERIMENTS.md` (E1–E10) and `results/`.

## 1. Introduction (~0.5 page)
- RUL prediction is well studied on C-MAPSS, but reported results often (a) compare single test scores without
  uncertainty, (b) mix truth-clipping conventions, (c) stop at offline accuracy.
- Contributions:
  1. one leakage-safe protocol for four model families (gradient boosting, TCN, LSTM+attention, ROCKET);
  2. paired-bootstrap comparison showing that the top deep models are statistically tied;
  3. explainability across model families via per-sensor grouped SHAP, and a measurement that attention ≠ attribution;
  4. a drift detector (KS + PSI) with a validated no-drift control, and drift-triggered retraining;
  5. a containerised multi-model API that provably reproduces offline predictions.

## 2. Data and protocol (~0.5 page)
- FD001: 100 run-to-failure training engines, 100 truncated test engines; 14 informative sensors (EDA drop rule).
- Engine-level split 80/20 (seed 42), scaler fit on training engines only, piecewise-linear labels capped at 125.
- Test: one window per engine (last 30 cycles), unclipped `RUL_FD001.txt`; clip125 reported for literature comparison
  (a perfect capped model scores RMSE 3.74 unclipped).
- Metrics: RMSE, MAE, NASA asymmetric score.

## 3. Models (~0.5 page)
- XGBoost on 238 past-only features (rolling stats, lags, EMA), Optuna 100 trials.
- TCN (causal, dilated), LSTM with Bahdanau attention (bi/uni ablation), Optuna 50 trials each, Colab T4.
- MiniRocket (10,000 kernels) + ridge with the penalty chosen on validation engines.

## 4. Results with uncertainty (~1 page)
- Table: test RMSE with 95 % bootstrap CIs, NASA, clip125, parameters, CPU latency (README table).
- Figure: paired bootstrap differences vs the best model (`results/model_comparison_paired.png`).
- Findings: four-way tie at the top (14.25–14.96); deep raw-window models beat engineered-feature XGBoost
  (+3.5 RMSE [+1.1, +5.8]); bidirectionality brings nothing; selection overfit (`lstm_bi_tuned`); ROCKET competitive
  without GPU or tuning, and not "300× faster" (ridge solve dominates).
- Model-only vs in-service latency (XGBoost's feature building dominates its request time).

## 5. Explainability (~0.75 page)
- Grouped SHAP per sensor as a common unit for 238 features vs 30 × 14 windows; agreement once the correlated
  s4/s7/s11/s12 group is pooled (~0.4 of importance in all models); s3 the real disagreement.
- Attention vs SHAP: tuned LSTMs barely use attention (one exactly uniform); attention and SHAP time profiles point in
  opposite directions (19 % vs 46 % on the newest 8 cycles).

## 6. Drift detection and retraining (~1 page)
- KS over-alarms on large autocorrelated samples (5/14 sensors between two halves of the same population); rule:
  KS AND PSI > 0.2, ≥ 3 sensors.
- Control (held-out FD001 engines): quiet. FD002: median PSI 3.0, 14/14 sensors.
- "Drift is not harm": FD001 test engines alarm because truncation shifts the health-state mix (diagnostic reproduces
  and explains it), while models perform normally on them.
- Impact: best FD001 model 63.6 RMSE on FD002 (constant predictor 54.1); retrained XGBoost 30.4 (18.3 capped = the floor).
- Design lessons: same-regime alarms → review, not retrain; reference must follow the served regime (else endless
  retraining) — and its limit for mixed fleets.

## 7. Serving (~0.25 page)
- FastAPI, shortlist of three models chosen per request, raw-reading input, end-to-end equivalence (300/300
  predictions), Prometheus metrics with a live drift score; Docker image 1.97 GB (CPU torch dominates).

## 8. Limitations and future work (~0.25 page)
- Single subset benchmark (FD002–FD004 future work, with operating-condition normalisation); 100-engine test set;
  FD002 test split used as both live batch and evaluation set; per-condition drift monitoring for mixed fleets.

## References
Saxena et al. 2008 (C-MAPSS); Bai et al. 2018 (TCN); Bahdanau et al. 2015 (attention); Dempster et al. 2021
(MiniRocket); Lundberg & Lee 2017 (SHAP); Chen & Guestrin 2016 (XGBoost); Akiba et al. 2019 (Optuna);
Efron & Tibshirani 1993 (bootstrap); Jain & Wallace 2019 ("Attention is not Explanation").

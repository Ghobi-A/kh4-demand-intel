# Behavioural demand forecasting report

> **Demo smoke run — not a substantive result.** This report was generated from the class-stratified 304-row portfolio extract, which is not volume-representative. It exists to prove the pipeline runs end to end. Measured forecasting results require a run over the full canonical dataset.


- Run: `6f0c63969a90` at 2026-10-17T00:18:39.275505+00:00
- Git commit: `ca7fca9`
- Input: `data/demo/weekly_demand_signals_demo.csv` (hash `f03bb95e6835a8ce`)
- Evaluation status: `demo_only`
- Volume representative: False
- Seed: 42

## Target definitions

`actionable_probability_mass` is the sum of calibrated P(actionable) over the comments observed in a period: an **expected actionable-signal volume**, or behavioural demand proxy. It is not expected purchases, revenue, or units. `total_comments` and `actionable_count` are raw observable counts.

## Temporal coverage

- Observed periods: 211
- Non-empty periods: 76
- Longest consecutive non-empty run: 7
- Usable event periods: 0

Model complexity is chosen from this measured coverage, not assumed in advance.

## Validation design

Rolling-origin (walk-forward) backtesting: minimum train 12 periods, horizon 2, step 2. Random splits are never used, and exogenous event features are rebuilt at each origin so an event that had not been announced yet cannot inform that origin's forecast.

## Target: `actionable_probability_mass`

- Kind: `continuous_nonnegative`
- Periods modelled: 211 (76 non-empty, 64% zero)
- Complexity tier: `rich`
- Forecast origins: 99

### Model comparison

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Interval calibration | Origins | Bias evidence (forecasts) |
|---|---|---|---|---|---|---|---|---|---|---|
| bayesian | 0.929 | 7.867 | 11.305 | -7.676 | 0.333 | 2.455 | yes | insufficient evidence | 3 | 6 |
| exponential_smoothing | 0.987 | 0.521 | 1.930 | -0.027 | 0.672 | 0.490 | yes | too narrow (overconfident) | 99 | 198 |
| mean | 1.014 | 0.536 | 1.957 | -0.220 | 0.833 | 0.625 | yes | close to nominal | 99 | 198 |
| naive | 1.016 | 0.536 | 1.941 | -0.023 | 0.707 | 0.489 | yes | close to nominal | 99 | 198 |

#### Matched-origin comparison (includes the Bayesian model)

Refitting the posterior at every origin is expensive, so the Bayesian model is scored on the most recent origins. This table re-scores every model on exactly those origins, so no two models are compared across different evaluation windows. Accuracy is therefore like-for-like, while the bias and calibration verdicts use each model's full evaluation — a much larger sample for the baselines, which is why the last column differs by row.

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Interval calibration | Origins | Bias evidence (forecasts) |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.819 | 6.937 | 10.490 | -1.657 | 0.500 | 0.992 | yes | close to nominal | 3 | 198 |
| exponential_smoothing | 0.827 | 7.007 | 10.572 | -1.629 | 0.500 | 0.978 | yes | too narrow (overconfident) | 3 | 198 |
| mean | 0.846 | 7.162 | 10.724 | -7.060 | 0.333 | 1.822 | yes | close to nominal | 3 | 198 |
| bayesian | 0.929 | 7.867 | 11.305 | -7.676 | 0.333 | 2.455 | yes | insufficient evidence | 3 | 6 |

#### How much this ranking can be trusted

6 forecasts over 3 origins per model. Lowest matched WAPE: naive (0.819); bayesian 0.929. The whole absolute-error gap between naive and bayesian comes from origin 208. Without it, WAPE is naive 0.961 vs bayesian 0.959. Leader with each origin left out: drop 204 -> naive, drop 206 -> exponential_smoothing, drop 208 -> bayesian. The ranking is fragile and should be read as anecdotal, not as evidence that one model is better.

Per-origin errors: `matched_origin_per_origin.csv`; leave-one-origin-out: `matched_origin_leave_one_out.csv`.

**Selected model: `naive`**

- Lowest WAPE: naive (0.819).
- Evaluated over too few forecasts to judge, so not eligible for selection over a more accurate model: bayesian.
- Every model with enough forecasts to judge shows persistent bias or poor interval calibration. None of them is reliable on this history; naive is reported as the lowest-error option, not as a recommendation.
- Selection used the origins on which every model, including the Bayesian one, was scored.
- The Bayesian model was backtested on the last 3 origins because refitting the posterior at every origin is expensive. Its comparison row is therefore based on fewer origins than the baselines.

### Forecast bias and reliability

- `bayesian`: only 6 forecasts, too few to call bias or interval calibration either way. Mean signed error -7.676, interval coverage 0.333, both reported as observations rather than verdicts.
- `exponential_smoothing`: mean signed error -0.027, interval coverage 0.672 (too narrow (overconfident)) over 198 forecasts; over forecasting for 6 periods, under forecasting for 4 periods.
- `mean`: mean signed error -0.220, interval coverage 0.833 (close to nominal) over 198 forecasts; over forecasting for 10 periods, under forecasting for 5 periods.
- `naive`: mean signed error -0.023, interval coverage 0.707 (close to nominal) over 198 forecasts; over forecasting for 4 periods, under forecasting for 4 periods.

### Bayesian model

- Likelihood: `hurdle_gamma` (complexity tier `rich`)
- Sampler: 4 chains, 1000 draws, 1000 tune, target_accept 0.9
- Max R-hat: 1.005, min bulk ESS: 4351.905, divergences: 0
- Converged: True
- 3 of 3 event features are constant across the training window, so the data cannot identify their effect. They are excluded rather than estimated from the prior alone.
- 135 of 211 periods are zero, so a hurdle model is used: a Bernoulli component for whether a period is non-zero and a Gamma component for its level. The target is never rounded and no epsilon is added to force a Gamma to accept a zero.
- Posterior predictive check: replicates are less variable than the data (predicted sd 1.09 against observed 2.16). The model under-states how large the busiest periods get, so its intervals should not be read as capturing spike risk.

| Parameter | Posterior mean | 5% | 95% |
|---|---|---|---|
| `zero_intercept` | -0.603 | -0.842 | -0.370 |
| `zero_trend` | 1.406 | 0.809 | 1.993 |
| `zero_seasonal_sin` | 0.096 | -0.189 | 0.383 |
| `zero_seasonal_cos` | 0.070 | -0.205 | 0.346 |
| `intercept` | -0.562 | -0.755 | -0.354 |
| `trend` | 1.322 | 0.808 | 1.822 |
| `seasonal_sin` | 0.207 | -0.051 | 0.456 |
| `seasonal_cos` | 0.675 | 0.487 | 0.862 |
| `gamma_shape` | 1.303 | 0.990 | 1.651 |

Event terms describe an association between an event window and the observed proxy. They are not causal estimates.

### Forecast

| Period | Forecast | Lower | Upper |
|---|---|---|---|
| 2026-04-27 | 1.979 | 0 | 5.829 |
| 2026-05-04 | 1.992 | 0 | 5.909 |
| 2026-05-11 | 1.949 | 0 | 5.928 |
| 2026-05-18 | 1.882 | 0 | 5.647 |

## Target: `actionable_count`

- Kind: `count`
- Periods modelled: 211 (70 non-empty, 67% zero)
- Complexity tier: `rich`
- Forecast origins: 99

### Model comparison

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Interval calibration | Origins | Bias evidence (forecasts) |
|---|---|---|---|---|---|---|---|---|---|---|
| bayesian | 0.936 | 14.659 | 20.952 | -14.226 | 0.333 | 4.500 | yes | insufficient evidence | 3 | 6 |
| exponential_smoothing | 1.033 | 1.059 | 3.586 | -0.012 | 0.758 | 1.475 | yes | close to nominal | 99 | 198 |
| mean | 1.046 | 1.072 | 3.694 | -0.395 | 0.894 | 1.528 | yes | close to nominal | 99 | 198 |
| naive | 1.099 | 1.126 | 3.754 | 0.015 | 0.838 | 1.481 | yes | close to nominal | 99 | 198 |

#### Matched-origin comparison (includes the Bayesian model)

Refitting the posterior at every origin is expensive, so the Bayesian model is scored on the most recent origins. This table re-scores every model on exactly those origins, so no two models are compared across different evaluation windows. Accuracy is therefore like-for-like, while the bias and calibration verdicts use each model's full evaluation — a much larger sample for the baselines, which is why the last column differs by row.

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Interval calibration | Origins | Bias evidence (forecasts) |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.787 | 12.333 | 18.982 | -3 | 0.500 | 1.943 | yes | close to nominal | 3 | 198 |
| exponential_smoothing | 0.791 | 12.388 | 18.962 | -2.575 | 0.333 | 2.270 | yes | close to nominal | 3 | 198 |
| mean | 0.851 | 13.333 | 19.817 | -13.125 | 0.333 | 3.648 | yes | close to nominal | 3 | 198 |
| bayesian | 0.936 | 14.659 | 20.952 | -14.226 | 0.333 | 4.500 | yes | insufficient evidence | 3 | 6 |

#### How much this ranking can be trusted

6 forecasts over 3 origins per model. Lowest matched WAPE: naive (0.787); bayesian 0.936. The whole absolute-error gap between naive and bayesian comes from origin 208. Without it, WAPE is naive 0.958 vs bayesian 0.958. Leader with each origin left out: drop 204 -> exponential_smoothing, drop 206 -> naive, drop 208 -> bayesian. The ranking is fragile and should be read as anecdotal, not as evidence that one model is better.

Per-origin errors: `matched_origin_per_origin.csv`; leave-one-origin-out: `matched_origin_leave_one_out.csv`.

**Selected model: `naive`**

- Lowest WAPE: naive (0.787).
- Evaluated over too few forecasts to judge, so not eligible for selection over a more accurate model: bayesian.
- Every model with enough forecasts to judge shows persistent bias or poor interval calibration. None of them is reliable on this history; naive is reported as the lowest-error option, not as a recommendation.
- Selection used the origins on which every model, including the Bayesian one, was scored.
- The Bayesian model was backtested on the last 3 origins because refitting the posterior at every origin is expensive. Its comparison row is therefore based on fewer origins than the baselines.

### Forecast bias and reliability

- `bayesian`: only 6 forecasts, too few to call bias or interval calibration either way. Mean signed error -14.226, interval coverage 0.333, both reported as observations rather than verdicts.
- `exponential_smoothing`: mean signed error -0.012, interval coverage 0.758 (close to nominal) over 198 forecasts; over forecasting for 6 periods.
- `mean`: mean signed error -0.395, interval coverage 0.894 (close to nominal) over 198 forecasts; over forecasting for 10 periods, under forecasting for 5 periods.
- `naive`: mean signed error 0.015, interval coverage 0.838 (close to nominal) over 198 forecasts; over forecasting for 4 periods.

### Bayesian model

- Likelihood: `negative_binomial` (complexity tier `rich`)
- Sampler: 4 chains, 1000 draws, 1000 tune, target_accept 0.9
- Max R-hat: 1.004, min bulk ESS: 3918.586, divergences: 0
- Converged: True
- 3 of 3 event features are constant across the training window, so the data cannot identify their effect. They are excluded rather than estimated from the prior alone.
- Integer count target; variance/mean = 15.21, so a Negative Binomial is used to admit overdispersion a Poisson would not.
- Posterior predictive check: replicates are less variable than the data (predicted sd 2.33 against observed 4.19). The model under-states how large the busiest periods get, so its intervals should not be read as capturing spike risk.

| Parameter | Posterior mean | 5% | 95% |
|---|---|---|---|
| `intercept` | -0.446 | -0.700 | -0.187 |
| `trend` | 1.504 | 0.930 | 2.045 |
| `seasonal_sin` | 0.117 | -0.197 | 0.427 |
| `seasonal_cos` | 0.624 | 0.360 | 0.899 |
| `nb_alpha` | 0.334 | 0.232 | 0.454 |

Event terms describe an association between an event window and the observed proxy. They are not causal estimates.

### Forecast

| Period | Forecast | Lower | Upper |
|---|---|---|---|
| 2026-04-27 | 2.936 | 0 | 9 |
| 2026-05-04 | 3.033 | 0 | 9 |
| 2026-05-11 | 3.027 | 0 | 9 |
| 2026-05-18 | 2.846 | 0 | 8 |

## Target: `total_comments`

- Kind: `count`
- Periods modelled: 211 (76 non-empty, 64% zero)
- Complexity tier: `rich`
- Forecast origins: 99

### Model comparison

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Interval calibration | Origins | Bias evidence (forecasts) |
|---|---|---|---|---|---|---|---|---|---|---|
| bayesian | 0.945 | 20.006 | 28.598 | -19.452 | 0.333 | 5 | yes | insufficient evidence | 3 | 6 |
| exponential_smoothing | 0.989 | 1.279 | 4.509 | -0.098 | 0.768 | 1.752 | yes | close to nominal | 99 | 198 |
| naive | 1.023 | 1.323 | 4.623 | -0.091 | 0.864 | 1.713 | yes | close to nominal | 99 | 198 |
| mean | 1.028 | 1.330 | 4.951 | -0.542 | 0.909 | 1.853 | yes | too wide (underconfident) | 99 | 198 |

#### Matched-origin comparison (includes the Bayesian model)

Refitting the posterior at every origin is expensive, so the Bayesian model is scored on the most recent origins. This table re-scores every model on exactly those origins, so no two models are compared across different evaluation windows. Accuracy is therefore like-for-like, while the bias and calibration verdicts use each model's full evaluation — a much larger sample for the baselines, which is why the last column differs by row.

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Interval calibration | Origins | Bias evidence (forecasts) |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.811 | 17.167 | 24.190 | -5.500 | 0.333 | 1.943 | yes | close to nominal | 3 | 198 |
| exponential_smoothing | 0.814 | 17.234 | 24.234 | -5.169 | 0.333 | 2.243 | yes | close to nominal | 3 | 198 |
| mean | 0.862 | 18.250 | 27.015 | -18.042 | 0.333 | 4.232 | yes | too wide (underconfident) | 3 | 198 |
| bayesian | 0.945 | 20.006 | 28.598 | -19.452 | 0.333 | 5 | yes | insufficient evidence | 3 | 6 |

#### How much this ranking can be trusted

6 forecasts over 3 origins per model. Lowest matched WAPE: naive (0.811); bayesian 0.945. The whole absolute-error gap between naive and bayesian comes from origin 208. Without it, WAPE is naive 0.968 vs bayesian 0.967. Leader with each origin left out: drop 204 -> naive, drop 206 -> naive, drop 208 -> bayesian. The ranking is fragile and should be read as anecdotal, not as evidence that one model is better.

Per-origin errors: `matched_origin_per_origin.csv`; leave-one-origin-out: `matched_origin_leave_one_out.csv`.

**Selected model: `naive`**

- Lowest WAPE: naive (0.811).
- Evaluated over too few forecasts to judge, so not eligible for selection over a more accurate model: bayesian.
- Every model with enough forecasts to judge shows persistent bias or poor interval calibration. None of them is reliable on this history; naive is reported as the lowest-error option, not as a recommendation.
- Selection used the origins on which every model, including the Bayesian one, was scored.
- The Bayesian model was backtested on the last 3 origins because refitting the posterior at every origin is expensive. Its comparison row is therefore based on fewer origins than the baselines.

### Forecast bias and reliability

- `bayesian`: only 6 forecasts, too few to call bias or interval calibration either way. Mean signed error -19.452, interval coverage 0.333, both reported as observations rather than verdicts.
- `exponential_smoothing`: mean signed error -0.098, interval coverage 0.768 (close to nominal) over 198 forecasts; over forecasting for 6 periods, under forecasting for 4 periods.
- `mean`: mean signed error -0.542, interval coverage 0.909 (too wide (underconfident)) over 198 forecasts; over forecasting for 10 periods, under forecasting for 5 periods.
- `naive`: mean signed error -0.091, interval coverage 0.864 (close to nominal) over 198 forecasts; over forecasting for 6 periods, under forecasting for 4 periods.

### Bayesian model

- Likelihood: `negative_binomial` (complexity tier `rich`)
- Sampler: 4 chains, 1000 draws, 1000 tune, target_accept 0.9
- Max R-hat: 1.002, min bulk ESS: 3749.440, divergences: 0
- Converged: True
- 3 of 3 event features are constant across the training window, so the data cannot identify their effect. They are excluded rather than estimated from the prior alone.
- Integer count target; variance/mean = 21.21, so a Negative Binomial is used to admit overdispersion a Poisson would not.
- Posterior predictive check: replicates are less variable than the data (predicted sd 2.78 against observed 5.53). The model under-states how large the busiest periods get, so its intervals should not be read as capturing spike risk.

| Parameter | Posterior mean | 5% | 95% |
|---|---|---|---|
| `intercept` | -0.489 | -0.733 | -0.234 |
| `trend` | 1.606 | 1.039 | 2.170 |
| `seasonal_sin` | 0.152 | -0.151 | 0.455 |
| `seasonal_cos` | 0.625 | 0.353 | 0.892 |
| `nb_alpha` | 0.333 | 0.242 | 0.446 |

Event terms describe an association between an event window and the observed proxy. They are not causal estimates.

### Forecast

| Period | Forecast | Lower | Upper |
|---|---|---|---|
| 2026-04-27 | 3.866 | 0 | 12 |
| 2026-05-04 | 3.837 | 0 | 11 |
| 2026-05-11 | 3.761 | 0 | 11 |
| 2026-05-18 | 3.719 | 0 | 11 |

## Limitations

- Community discussion is not unit sales. These targets measure observable public behaviour, and Kingdom Hearts IV is unreleased, so no purchase conversion exists to validate against.
- Sampling is platform-dependent. Comment volume reflects what was collected, which videos and threads existed, and how platform algorithms surfaced them.
- The labelled intent corpus is still small, so the calibrated probabilities underlying the demand proxy are development-preliminary.
- Event coefficients describe association, not causation. An uplift after a trailer is a modelled event response, not proof the trailer caused it.
- Uncertainty grows with the horizon; long-range forecasts on this history are weak and their intervals should be read as such.

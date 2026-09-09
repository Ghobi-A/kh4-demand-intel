# Behavioural demand forecasting report

> **Demo smoke run — not a substantive result.** This report was generated from the class-stratified 304-row portfolio extract, which is not volume-representative. It exists to prove the pipeline runs end to end. Measured forecasting results require a run over the full canonical dataset.


- Run: `418aaacc1e35` at 2026-09-09T06:04:05.502061+00:00
- Git commit: `b7d0c09`
- Input: `data/demo/weekly_demand_signals_demo.csv` (hash `470790bb1fc09117`)
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
| bayesian | 0.939 | 9.563 | 13.770 | -9.316 | 0.333 | 2.641 | yes | insufficient evidence | 3 | 6 |
| exponential_smoothing | 0.986 | 0.610 | 2.180 | -0.045 | 0.747 | 0.772 | yes | close to nominal | 99 | 198 |
| naive | 1.019 | 0.630 | 2.217 | -0.045 | 0.773 | 0.767 | yes | close to nominal | 99 | 198 |
| mean | 1.021 | 0.632 | 2.383 | -0.263 | 0.874 | 0.826 | yes | close to nominal | 99 | 198 |

#### Matched-origin comparison (includes the Bayesian model)

Refitting the posterior at every origin is expensive, so the Bayesian model is scored on the most recent origins. This table re-scores every model on exactly those origins, so no two models are compared across different evaluation windows. Accuracy is therefore like-for-like, while the bias and calibration verdicts use each model's full evaluation — a much larger sample for the baselines, which is why the last column differs by row.

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Interval calibration | Origins | Bias evidence (forecasts) |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.816 | 8.305 | 11.734 | -2.590 | 0.333 | 1.003 | yes | close to nominal | 3 | 198 |
| exponential_smoothing | 0.820 | 8.345 | 11.783 | -2.481 | 0.333 | 1.070 | yes | close to nominal | 3 | 198 |
| mean | 0.861 | 8.767 | 13.053 | -8.665 | 0.333 | 2.014 | yes | close to nominal | 3 | 198 |
| bayesian | 0.939 | 9.563 | 13.770 | -9.316 | 0.333 | 2.641 | yes | insufficient evidence | 3 | 6 |

**Selected model: `naive`**

- Lowest WAPE: naive (0.816).
- Evaluated over too few forecasts to judge, so not eligible for selection over a more accurate model: bayesian.
- Every model with enough forecasts to judge shows persistent bias or poor interval calibration. None of them is reliable on this history; naive is reported as the lowest-error option, not as a recommendation.
- Selection used the origins on which every model, including the Bayesian one, was scored.
- The Bayesian model was backtested on the last 3 origins because refitting the posterior at every origin is expensive. Its comparison row is therefore based on fewer origins than the baselines.

### Forecast bias and reliability

- `bayesian`: only 6 forecasts, too few to call bias or interval calibration either way. Mean signed error -9.316, interval coverage 0.333, both reported as observations rather than verdicts.
- `exponential_smoothing`: mean signed error -0.045, interval coverage 0.747 (close to nominal) over 198 forecasts; over forecasting for 6 periods, under forecasting for 4 periods.
- `mean`: mean signed error -0.263, interval coverage 0.874 (close to nominal) over 198 forecasts; over forecasting for 10 periods, under forecasting for 5 periods.
- `naive`: mean signed error -0.045, interval coverage 0.773 (close to nominal) over 198 forecasts; over forecasting for 6 periods, under forecasting for 4 periods.

### Bayesian model

- Likelihood: `hurdle_gamma` (complexity tier `rich`)
- Sampler: 4 chains, 1000 draws, 1000 tune, target_accept 0.9
- Max R-hat: 1.004, min bulk ESS: 4988.241, divergences: 0
- Converged: True
- 3 of 3 event features are constant across the training window, so the data cannot identify their effect. They are excluded rather than estimated from the prior alone.
- 135 of 211 periods are zero, so a hurdle model is used: a Bernoulli component for whether a period is non-zero and a Gamma component for its level. The target is never rounded and no epsilon is added to force a Gamma to accept a zero.
- Posterior predictive check: replicates are less variable than the data (predicted sd 1.34 against observed 2.64). The model under-states how large the busiest periods get, so its intervals should not be read as capturing spike risk.

| Parameter | Posterior mean | 5% | 95% |
|---|---|---|---|
| `zero_intercept` | -0.604 | -0.842 | -0.366 |
| `zero_trend` | 1.406 | 0.806 | 1.983 |
| `zero_seasonal_sin` | 0.097 | -0.194 | 0.388 |
| `zero_seasonal_cos` | 0.069 | -0.198 | 0.343 |
| `intercept` | -0.505 | -0.692 | -0.298 |
| `trend` | 1.117 | 0.616 | 1.602 |
| `seasonal_sin` | 0.135 | -0.118 | 0.392 |
| `seasonal_cos` | 0.689 | 0.488 | 0.884 |
| `gamma_shape` | 1.278 | 0.973 | 1.624 |

Event terms describe an association between an event window and the observed proxy. They are not causal estimates.

### Forecast

| Period | Forecast | Lower | Upper |
|---|---|---|---|
| 2026-04-27 | 2.266 | 0 | 6.717 |
| 2026-05-04 | 2.237 | 0 | 6.735 |
| 2026-05-11 | 2.132 | 0 | 6.578 |
| 2026-05-18 | 2.074 | 0 | 6.188 |

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

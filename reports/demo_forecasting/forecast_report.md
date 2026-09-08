# Behavioural demand forecasting report

> **Demo smoke run — not a substantive result.** This report was generated from the class-stratified 304-row portfolio extract, which is not volume-representative. It exists to prove the pipeline runs end to end. Measured forecasting results require a run over the full canonical dataset.


- Run: `c8d4c90dcaa6` at 2026-09-08T16:03:39.326767+00:00
- Git commit: `ab063db`
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

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Origins |
|---|---|---|---|---|---|---|---|---|
| bayesian | 0.936 | 9.525 | 13.727 | -9.257 | 0.333 | 2.706 | yes | 3 |
| exponential_smoothing | 0.986 | 0.610 | 2.180 | -0.045 | 0.747 | 0.772 | yes | 99 |
| naive | 1.019 | 0.630 | 2.217 | -0.045 | 0.773 | 0.767 | yes | 99 |
| mean | 1.021 | 0.632 | 2.383 | -0.263 | 0.874 | 0.826 | yes | 99 |

#### Matched-origin comparison (includes the Bayesian model)

Refitting the posterior at every origin is expensive, so the Bayesian model is scored on the most recent origins. This table re-scores every model on exactly those origins, so no two models are compared across different evaluation windows.

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Origins |
|---|---|---|---|---|---|---|---|---|
| naive | 0.816 | 8.305 | 11.734 | -2.590 | 0.333 | 1.003 | yes | 3 |
| exponential_smoothing | 0.820 | 8.345 | 11.783 | -2.481 | 0.333 | 1.070 | yes | 3 |
| mean | 0.861 | 8.767 | 13.053 | -8.665 | 0.333 | 2.014 | yes | 3 |
| bayesian | 0.936 | 9.525 | 13.727 | -9.257 | 0.333 | 2.706 | yes | 3 |

**Selected model: `naive`**

- Lowest WAPE: naive (0.816).
- Every candidate shows persistent bias or poor interval calibration; the lowest-WAPE model is reported, but none of them is reliable on this history.
- Selection used the origins on which every model, including the Bayesian one, was scored.
- The Bayesian model was backtested on the last 3 origins because refitting the posterior at every origin is expensive. Its comparison row is therefore based on fewer origins than the baselines.

### Forecast bias and reliability

- `bayesian`: mean signed error -9.257, interval coverage 0.333 (too narrow (overconfident)); under forecasting for 4 periods.
- `exponential_smoothing`: mean signed error -0.045, interval coverage 0.747 (close to nominal); over forecasting for 6 periods, under forecasting for 4 periods.
- `mean`: mean signed error -0.263, interval coverage 0.874 (close to nominal); over forecasting for 10 periods, under forecasting for 5 periods.
- `naive`: mean signed error -0.045, interval coverage 0.773 (close to nominal); over forecasting for 6 periods, under forecasting for 4 periods.

### Bayesian model

- Likelihood: `hurdle_gamma` (complexity tier `rich`)
- Sampler: 2 chains, 400 draws, 400 tune, target_accept 0.9
- Max R-hat: 1.012, min bulk ESS: 819.955, divergences: 0
- Converged: False
  - Warning: Max R-hat 1.012 exceeds 1.01: chains have not mixed, so posterior summaries are unreliable.
- 135 of 211 periods are zero, so a hurdle model is used: a Bernoulli component for whether a period is non-zero and a Gamma component for its level. The target is never rounded and no epsilon is added to force a Gamma to accept a zero.

| Parameter | Posterior mean | 5% | 95% |
|---|---|---|---|
| `zero_intercept` | -1.288 | -1.695 | -0.915 |
| `zero_trend` | 1.372 | 0.790 | 1.961 |
| `zero_seasonal_sin` | 0.096 | -0.194 | 0.382 |
| `zero_seasonal_cos` | 0.068 | -0.231 | 0.383 |
| `zero_event_effect[0]` | -0.014 | -1.614 | 1.538 |
| `zero_event_effect[1]` | -0.034 | -1.613 | 1.677 |
| `zero_event_effect[2]` | 0.002 | -1.564 | 1.585 |
| `intercept` | -1.048 | -1.412 | -0.642 |
| `trend` | 1.100 | 0.597 | 1.562 |
| `seasonal_sin` | 0.133 | -0.123 | 0.392 |
| `seasonal_cos` | 0.685 | 0.501 | 0.868 |
| `event_effect[0]` | -0.059 | -1.834 | 1.686 |
| `event_effect[1]` | -0.042 | -1.792 | 1.679 |
| `event_effect[2]` | -0.042 | -1.683 | 1.458 |
| `gamma_shape` | 1.275 | 0.986 | 1.592 |

Event terms describe an association between an event window and the observed proxy. They are not causal estimates.

### Forecast

| Period | Forecast | Lower | Upper |
|---|---|---|---|
| 2026-04-27 | 2.067 | 0 | 6.333 |
| 2026-05-04 | 2.166 | 0 | 6.402 |
| 2026-05-11 | 2.334 | 0 | 6.690 |
| 2026-05-18 | 2.141 | 0 | 6.288 |

## Target: `total_comments`

- Kind: `count`
- Periods modelled: 211 (76 non-empty, 64% zero)
- Complexity tier: `rich`
- Forecast origins: 99

### Model comparison

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Origins |
|---|---|---|---|---|---|---|---|---|
| bayesian | 0.944 | 19.982 | 28.590 | -19.421 | 0.333 | 5.167 | yes | 3 |
| exponential_smoothing | 0.989 | 1.279 | 4.509 | -0.098 | 0.768 | 1.752 | yes | 99 |
| naive | 1.023 | 1.323 | 4.623 | -0.091 | 0.864 | 1.713 | yes | 99 |
| mean | 1.028 | 1.330 | 4.951 | -0.542 | 0.909 | 1.853 | yes | 99 |

#### Matched-origin comparison (includes the Bayesian model)

Refitting the posterior at every origin is expensive, so the Bayesian model is scored on the most recent origins. This table re-scores every model on exactly those origins, so no two models are compared across different evaluation windows.

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias | Origins |
|---|---|---|---|---|---|---|---|---|
| naive | 0.811 | 17.167 | 24.190 | -5.500 | 0.333 | 1.943 | yes | 3 |
| exponential_smoothing | 0.814 | 17.234 | 24.234 | -5.169 | 0.333 | 2.243 | yes | 3 |
| mean | 0.862 | 18.250 | 27.015 | -18.042 | 0.333 | 4.232 | yes | 3 |
| bayesian | 0.944 | 19.982 | 28.590 | -19.421 | 0.333 | 5.167 | yes | 3 |

**Selected model: `naive`**

- Lowest WAPE: naive (0.811).
- Every candidate shows persistent bias or poor interval calibration; the lowest-WAPE model is reported, but none of them is reliable on this history.
- Selection used the origins on which every model, including the Bayesian one, was scored.
- The Bayesian model was backtested on the last 3 origins because refitting the posterior at every origin is expensive. Its comparison row is therefore based on fewer origins than the baselines.

### Forecast bias and reliability

- `bayesian`: mean signed error -19.421, interval coverage 0.333 (too narrow (overconfident)); under forecasting for 4 periods.
- `exponential_smoothing`: mean signed error -0.098, interval coverage 0.768 (close to nominal); over forecasting for 6 periods, under forecasting for 4 periods.
- `mean`: mean signed error -0.542, interval coverage 0.909 (too wide (underconfident)); over forecasting for 10 periods, under forecasting for 5 periods.
- `naive`: mean signed error -0.091, interval coverage 0.864 (close to nominal); over forecasting for 6 periods, under forecasting for 4 periods.

### Bayesian model

- Likelihood: `negative_binomial` (complexity tier `rich`)
- Sampler: 2 chains, 400 draws, 400 tune, target_accept 0.9
- Max R-hat: 1.007, min bulk ESS: 487.890, divergences: 0
- Converged: True
- Integer count target; variance/mean = 21.21, so a Negative Binomial is used to admit overdispersion a Poisson would not.

| Parameter | Posterior mean | 5% | 95% |
|---|---|---|---|
| `intercept` | -1.277 | -1.705 | -0.836 |
| `trend` | 1.581 | 1.023 | 2.118 |
| `seasonal_sin` | 0.158 | -0.142 | 0.463 |
| `seasonal_cos` | 0.619 | 0.340 | 0.868 |
| `event_effect[0]` | -0.024 | -1.626 | 1.712 |
| `event_effect[1]` | 0.032 | -1.688 | 1.670 |
| `event_effect[2]` | 0.007 | -1.639 | 1.714 |
| `nb_alpha` | 0.332 | 0.238 | 0.438 |

Event terms describe an association between an event window and the observed proxy. They are not causal estimates.

### Forecast

| Period | Forecast | Lower | Upper |
|---|---|---|---|
| 2026-04-27 | 3.476 | 0 | 11 |
| 2026-05-04 | 3.856 | 0 | 12 |
| 2026-05-11 | 4.144 | 0 | 12 |
| 2026-05-18 | 4.124 | 0 | 12.100 |

## Limitations

- Community discussion is not unit sales. These targets measure observable public behaviour, and Kingdom Hearts IV is unreleased, so no purchase conversion exists to validate against.
- Sampling is platform-dependent. Comment volume reflects what was collected, which videos and threads existed, and how platform algorithms surfaced them.
- The labelled intent corpus is still small, so the calibrated probabilities underlying the demand proxy are development-preliminary.
- Event coefficients describe association, not causation. An uplift after a trailer is a modelled event response, not proof the trailer caused it.
- Uncertainty grows with the horizon; long-range forecasts on this history are weak and their intervals should be read as such.

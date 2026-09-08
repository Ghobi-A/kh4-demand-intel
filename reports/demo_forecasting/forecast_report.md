# Behavioural demand forecasting report

> **Demo smoke run — not a substantive result.** This report was generated from the class-stratified 304-row portfolio extract, which is not volume-representative. It exists to prove the pipeline runs end to end. Measured forecasting results require a run over the full canonical dataset.


- Run: `6604d9e3f4d4` at 2026-09-08T15:22:18.896062+00:00
- Git commit: `a1a791f`
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

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias |
|---|---|---|---|---|---|---|---|
| exponential_smoothing | 0.999 | 0.613 | 2.088 | -0.049 | 0.742 | 0.770 | yes |
| mean | 1.027 | 0.631 | 2.333 | -0.256 | 0.833 | 0.723 | yes |
| naive | 1.034 | 0.635 | 2.127 | -0.050 | 0.742 | 0.767 | yes |

**Selected model: `exponential_smoothing`**

- Lowest WAPE: exponential_smoothing (0.999).
- Every candidate shows persistent bias or poor interval calibration; the lowest-WAPE model is reported, but none of them is reliable on this history.

### Forecast bias and reliability

- `exponential_smoothing`: mean signed error -0.049, interval coverage 0.742 (close to nominal); over forecasting for 6 periods, under forecasting for 4 periods.
- `mean`: mean signed error -0.256, interval coverage 0.833 (close to nominal); over forecasting for 10 periods, under forecasting for 5 periods.
- `naive`: mean signed error -0.050, interval coverage 0.742 (close to nominal); over forecasting for 6 periods, under forecasting for 4 periods.

### Bayesian model

- Likelihood: `hurdle_gamma` (complexity tier `rich`)
- Sampler: 2 chains, 400 draws, 400 tune, target_accept 0.9
- Max R-hat: 1.010, min bulk ESS: 839.300, divergences: 0
- Converged: False
  - Warning: Max R-hat 1.010 exceeds 1.01: chains have not mixed, so posterior summaries are unreliable.
- 135 of 211 periods are zero, so a hurdle model is used: a Bernoulli component for whether a period is non-zero and a Gamma component for its level. The target is never rounded and no epsilon is added to force a Gamma to accept a zero.

| Parameter | Posterior mean | 5% | 95% |
|---|---|---|---|
| `zero_intercept` | -1.295 | -1.717 | -0.882 |
| `zero_trend` | 1.385 | 0.811 | 1.980 |
| `zero_seasonal_sin` | 0.098 | -0.192 | 0.381 |
| `zero_seasonal_cos` | 0.073 | -0.239 | 0.369 |
| `zero_event_effect[0]` | -0.003 | -1.808 | 1.772 |
| `zero_event_effect[1]` | -0.003 | -1.627 | 1.715 |
| `zero_event_effect[2]` | 0.003 | -1.682 | 1.552 |
| `intercept` | -1.046 | -1.391 | -0.646 |
| `trend` | 1.102 | 0.600 | 1.572 |
| `seasonal_sin` | 0.121 | -0.128 | 0.368 |
| `seasonal_cos` | 0.687 | 0.513 | 0.860 |
| `event_effect[0]` | -0.027 | -1.746 | 1.570 |
| `event_effect[1]` | 0.034 | -1.554 | 1.600 |
| `event_effect[2]` | 0.003 | -1.558 | 1.555 |
| `gamma_shape` | 1.279 | 1.002 | 1.592 |

Event terms describe an association between an event window and the observed proxy. They are not causal estimates.

### Forecast

| Period | Forecast | Lower | Upper |
|---|---|---|---|
| 2026-04-27 | 2.245 | 0 | 6.736 |
| 2026-05-04 | 2.132 | 0 | 6.373 |
| 2026-05-11 | 2.199 | 0 | 6.740 |
| 2026-05-18 | 1.977 | 0 | 5.709 |

## Target: `total_comments`

- Kind: `count`
- Periods modelled: 211 (76 non-empty, 64% zero)
- Complexity tier: `rich`
- Forecast origins: 99

### Model comparison

| Model | WAPE | MAE | RMSE | Signed bias | Coverage | Interval width | Persistent bias |
|---|---|---|---|---|---|---|---|
| exponential_smoothing | 0.989 | 1.279 | 4.509 | -0.098 | 0.768 | 1.752 | yes |
| naive | 1.023 | 1.323 | 4.623 | -0.091 | 0.864 | 1.713 | yes |
| mean | 1.028 | 1.330 | 4.951 | -0.542 | 0.864 | 1.634 | yes |

**Selected model: `exponential_smoothing`**

- Lowest WAPE: exponential_smoothing (0.989).
- Every candidate shows persistent bias or poor interval calibration; the lowest-WAPE model is reported, but none of them is reliable on this history.

### Forecast bias and reliability

- `exponential_smoothing`: mean signed error -0.098, interval coverage 0.768 (close to nominal); over forecasting for 6 periods, under forecasting for 4 periods.
- `mean`: mean signed error -0.542, interval coverage 0.864 (close to nominal); over forecasting for 10 periods, under forecasting for 5 periods.
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

# Forecasting behavioural demand

## What is being forecast

Public community discussion is not sales. Kingdom Hearts IV is unreleased, so
there is no purchase conversion to predict and nothing here should be read as a
sales, revenue or unit forecast. What *can* be forecast is the observable
behaviour itself, and the project models several such quantities so that no
single derived construct carries the whole argument.

### Primary target: `actionable_probability_mass`

For a period *t*:

```
actionable_probability_mass_t = sum over comments i observed in t of P(actionable_i)
```

where `P(actionable_i)` is the calibrated Stage 1 probability from the existing
hierarchical intent classifier. Summing probabilities rather than counting hard
labels keeps the uncertainty in the classifier's own output: a comment the model
is 55% sure about contributes 0.55, not a whole unit.

Read it as an **expected actionable-signal volume**, or a behavioural demand
proxy. It answers "how much actionable-looking public discussion happened this
week", not "how many people bought something".

### Secondary targets

| Target | What it is | Why it is included |
|---|---|---|
| `total_comments` | Raw weekly discussion volume | An independent attention benchmark that depends on no model |
| `actionable_count` | Hard-label actionable comments | Robustness check if calibrated probabilities are unstable |
| `actionable_rate` | Actionable share of comments | Composition rather than volume |

Every run benchmarks at least the probability mass and total volume, so a
reader can see whether conclusions hold across a derived quantity and a raw one.

## Probability provenance

The demand proxy depends on classifier probabilities, so where each probability
came from is recorded per row in `probability_source`:

| Value | Meaning |
|---|---|
| `oof_supervised` | Out-of-fold: produced by a model trained without this row *or any comment from its discussion group* |
| `persisted_supervised` | Produced by the persisted trained model, for rows outside the labelled corpus |
| `excluded_training_row` | In the labelled corpus but no out-of-fold value available; excluded from the proxy |
| `unavailable` | No usable probability; the row contributes nothing to the proxy |

The 0.7/0.3 rule-confidence indicator from the regex baseline is **never** used
as a probability. If no calibrated probabilities can be produced at all, the
proxy is marked unavailable rather than fabricated, and the count targets carry
the analysis.

## Coverage audit before model choice

Model complexity is chosen from measured history, not assumed. Before fitting,
`src/timestamps.py` reports earliest and latest observation, observed and
non-empty periods, per-source coverage, gaps, comments-per-period distribution
and how many events have usable pre/post windows. That audit sets the tier:

| Non-empty periods | Tier | Structure offered |
|---|---|---|
| under 30 | `minimal` | Intercept, linear trend, event effects |
| 30 to 60 | `moderate` | Adds a weakly regularised annual Fourier pair |
| over 60 | `rich` | Richer dynamics, if diagnostics support them |

Nothing is forced. A seasonal model is not offered when the earliest backtest
origin could not fit it, and a target with too little history is reported as
unmodellable rather than fitted anyway.

## Validation design

**Rolling-origin (walk-forward) only.** A random split lets a model learn from
weeks after the one it predicts, which no real forecast can do.

```
train weeks 1-12  -> predict 13-14
train weeks 1-14  -> predict 15-16
train weeks 1-16  -> predict 17-18
```

Minimum train size, horizon, step and the number of origins are configurable.
Two leakage properties are enforced by tests:

1. **No future values.** Mutating later weeks cannot change an earlier origin's
   forecast.
2. **No future knowledge.** Event features are rebuilt at each origin and gated
   on `announced_date`. An event that had not been publicly announced at the
   origin contributes nothing there, so a surprise trailer cannot inform a
   forecast made before anyone knew about it. The descriptive weekly table uses
   the full retrospective calendar and is labelled `event_annotation = retrospective`.

## Models

Simple baselines stay eligible to win, because on a short sparse series they
frequently should.

| Model | Notes |
|---|---|
| `naive` | Repeat the last observation |
| `mean` | Trailing-window mean; robust on spiky series |
| `seasonal_naive` | Only offered with two full cycles *and* a long enough earliest origin |
| `exponential_smoothing` | Holt-style, via statsmodels |
| `ridge_event` | Ridge on lags plus event exogenous features |

### Bayesian model

The likelihood follows the target rather than the reverse:

| Target kind | Likelihood | Why |
|---|---|---|
| Continuous, non-negative, contains zeros | **Hurdle**: Bernoulli for non-zero, Gamma for level | The proxy is continuous and legitimately zero in quiet weeks. It is never rounded, and no epsilon is added to force a Gamma to accept a zero |
| Continuous, no zeros | Gamma | The hurdle component is unnecessary; the reason is recorded |
| Integer counts | Negative Binomial | Admits the overdispersion these series show |
| Rates | Binomial on the counts | A Binomial fitted to a float rate discards the sample size; weeks with no comments carry no rate information and are dropped rather than read as 0% |

Linear predictor (on the log or logit scale as appropriate):

```
intercept + trend + event_effect x event_decay + seasonal terms (higher tiers)
```

## Metrics

MAE, RMSE, WAPE, signed bias, relative bias, interval coverage and mean interval
width. MASE only when the in-sample naive scale is non-degenerate.

**Ordinary MAPE is deliberately absent.** The weekly series contains zero weeks,
where a percentage error is undefined or explodes. WAPE gives the same
error-relative-to-scale reading while staying finite.

## Model selection

Multi-criterion, because accuracy alone hides the failure that matters most:

1. **WAPE** (primary)
2. **MAE**, **signed bias**, **interval coverage** (secondary)
3. **Simpler model** (tie-break)

A model that wins on WAPE while forecasting persistently in one direction, or
whose intervals are badly calibrated, is demoted, and the report says so. If no
candidate is sound, the report says that too rather than crowning one.

## Monitoring

Tracked over successive origins: actual, forecast, absolute and signed error,
rolling MAE, rolling WAPE, rolling bias, and interval hit or miss.

**Persistent bias detection** flags runs of consecutive same-signed errors that
also exceed a materiality threshold, so a long run of negligible errors does not
raise a false alarm. Both the run length and the materiality fraction are
configurable.

## Posterior predictive checking

The check compares observations against posterior predictive **replicates**:
draws pushed all the way through the observation model, including its sampling
noise and, for the hurdle model, its Bernoulli zero component. Summarising draws
of the latent *mean* instead makes a well-calibrated model look badly calibrated,
because the mean of a series is estimated far more precisely than any single
future observation of it. That defect was present in an earlier revision and
produced a coverage figure near 0.45 against a 0.90 nominal level; it is now
covered by regression tests.

Two failures the current check reports honestly:

- The model reproduces the observed share of empty weeks closely, so the hurdle
  component is doing its job.
- Its replicates are less variable than the data, so it understates how large the
  busiest weeks get. Its intervals should not be read as capturing spike risk.

## Identifiability

Event features that never vary inside a training window are dropped rather than
estimated. In the demo data every calendar event post-dates the observations, so
all event columns are constant; a coefficient fitted there would be a draw from
its prior applied to the forecast as though it had been learned from data.

## Evidence thresholds

Persistent bias and interval calibration are only reported as verdicts when there
are enough forecasts to support one. A run of four same-signed errors out of six
occurs roughly one time in five by chance, so with fewer than twelve forecasts
both are reported as observations and the model is marked
`insufficient evidence`. Such a model is never *selected over* a more accurate
one either: being evaluated less is not a merit, and rewarding it would
systematically favour the Bayesian model, which is backtested over fewer origins
than the baselines because refitting a posterior at every origin is expensive.

## Limitations

- Community signals are not unit sales, and no purchase data exists to validate against.
- Sampling is platform-dependent: volume reflects what was collected and how platform algorithms surfaced it.
- The labelled intent corpus is small, so the calibrated probabilities behind the proxy are development-preliminary.
- Event coefficients are associations, not causal effects.
- Uncertainty grows with horizon; long-range forecasts on this history are weak.
- The committed demo results come from a class-stratified sample and are not volume-representative.

## Reproducing

```bash
# Canonical run over the full local dataset
python scripts/build_temporal_dataset.py --input data/processed/signals_scored.csv
python scripts/run_forecasting.py --save-posterior

# Smoke run over the committed sample (not a substantive result)
python scripts/build_temporal_dataset.py --demo
python scripts/run_forecasting.py --demo
```

Each run records run ID, git commit, input hash, targets, likelihoods, complexity
tier, origins, metrics, seed, sampler configuration and diagnostics in
`run_metadata.json`.

# Synthetic marketing science lab

**SYNTHETIC MARKETING SCIENCE DEMONSTRATION. This is not Square Enix data.**

## Why synthetic

Kingdom Hearts IV is unreleased. There is no public media spend, pricing,
promotion or sales data for it, and no amount of analysis creates one. The
options were to invent figures and present them as observed, to skip
marketing-mix modelling entirely, or to simulate a market from an explicit
process and say so everywhere. This project takes the third.

Simulation has a second advantage over borrowed real data: the truth is known.
On real marketing data a plausible-looking fit cannot be checked, because nobody
observes the true contribution of a channel. Here the generating parameters are
recorded in `data/synthetic/mmm_ground_truth.json`, so the model is scored on
whether it recovers them, and its failures are reported.

## Separation from the real data

| | Real KH4 signals | Synthetic marketing |
|---|---|---|
| Data | `data/processed/`, `data/demo/` | `data/synthetic/` |
| Code | `src/temporal.py`, `src/forecasting/` | `src/mmm/` |
| Reports | `reports/forecasting/`, `reports/demo_forecasting/` | `reports/mmm/` |

The two never mix. Every synthetic row carries `data_type = synthetic`, the
fitter refuses input not marked synthetic, and every report and dashboard view
leads with the disclaimer.

## Data-generating process

104 simulated weeks:

```
simulated_sales_t = baseline + trend + seasonality
                    + sum over channels of beta_c * hill(adstock(spend_c))
                    + price_coefficient * (price_t - base_price)
                    + promotion_lift * promotion_t
                    + event_lift * event_t
                    + noise
```

Three channels (video, paid search, social), each with its own true adstock
decay, Hill shape, half-saturation point and coefficient. Scope is fixed there
deliberately: three channels plus price, promotion, event and seasonality is
enough to demonstrate adstock, saturation, contribution, identifiability and
budget reallocation.

An optional `correlated_channels` mode makes the channels move together, which
is the standard hard case in media attribution.

## Transforms

**Geometric adstock** carries advertising forward:

```
A_t = X_t + decay * A_(t-1)
```

**Hill saturation** bends response towards a ceiling:

```
response(x) = x^alpha / (x^alpha + theta^alpha)
```

Both are implemented in NumPy for the generator and mirrored in PyTensor for
estimation, with a test asserting the two agree exactly, so the model estimates
the same transform the generator used. In PyTensor adstock is a truncated
convolution rather than a scan.

Note that Hill saturation with `alpha > 1` is S-shaped: it accelerates from zero
before bending over. The recovery check therefore tests the economic claim —
response rises while its marginal return falls — rather than global concavity,
which the true process does not satisfy either.

## The model

The same structure as the generator, with decay, curve shape, half-saturation
point and all coefficients estimated. Media coefficients are constrained
non-negative, spend is scaled per channel so priors mean the same thing across
budgets, and priors are weakly informative.

## Recovery against known truth

`reports/mmm/recovery.csv` scores each check. On the committed run the model:

- recovers the **direction and rough magnitude** of the price, promotion and event effects;
- keeps the **true contribution inside the 90% credible interval** for every channel;
- shows **diminishing returns** in every response curve;
- recovers **adstock decay** for most channels;
- but **swaps the top two channels** by contribution.

That last failure is the honest headline. Channel contributions are only
partially identified: channels whose spend moves together, or whose adstock and
saturation shapes are similar, trade off against one another, and the model can
fit the data well while assigning credit differently from the truth. The
credible intervals are wide and overlapping, which is the model correctly
reporting what it cannot separate. Point estimates of channel contribution
should never be quoted without them.

## Scenario simulator

Change per-channel budgets, price, promotion or event status, and the plan is
pushed back through the model's own adstock and saturation curves. Because
saturation is non-linear, moving budget between channels does not move simulated
sales proportionally, which is the point of asking. Output includes the baseline
and scenario expectation, the difference with a credible interval, the
probability of an increase, and per-channel contribution changes.

A plan that pushes spend or price beyond the range in the training data raises a
warning: the saturation curve is unconstrained out there, so the answer is
extrapolation rather than an estimate.

## What this shows and does not show

- It **does** show that adstock, saturation, price, promotion and event effects can be estimated jointly with uncertainty, and that the estimator recovers a known process.
- It **does not** show causal media effectiveness. A fitted coefficient is an association within an assumed model; establishing causal effect needs experimentation such as geo tests or holdouts.
- It **does not** describe Square Enix, Kingdom Hearts IV, or any real market.

## Reproducing

```bash
python -m src.mmm.synthetic                    # regenerate the dataset (fixed seed)
python scripts/run_mmm.py --save-posterior     # fit, evaluate, scenario, report
python scripts/run_mmm.py --correlated-channels  # harder identifiability case
```

# Synthetic marketing data

**SYNTHETIC MARKETING SCIENCE DEMONSTRATION. This is not Square Enix data.**

Every file in this directory is simulated by `src/mmm/synthetic.py` from an
explicit data-generating process. Nothing here is observed, and no quantity
describes actual commercial performance by Square Enix or anyone else.

## Why synthetic

Kingdom Hearts IV is unreleased. There is no public media spend, pricing,
promotion or sales data for it, and inventing such figures and presenting them
as real would be dishonest. Simulating them instead makes the marketing-science
methods demonstrable *and* checkable: because the true parameters are known and
recorded in `mmm_ground_truth.json`, the fitted model can be judged on whether
it recovers them.

## Files

| File | Contents |
|---|---|
| `mmm_weekly_synthetic.csv` | 104 simulated weeks of spend, price, promotion, event, seasonality and `simulated_sales`. Every row carries `data_type = synthetic`. |
| `mmm_ground_truth.json` | The generator's seed and true parameters, plus the true channel contributions the model is scored against. |

Regenerate both with:

```bash
python -m src.mmm.synthetic
```

The seed is fixed, so the output is byte-identical on every run.

## Separation from the real data

This directory never mixes with the real Kingdom Hearts IV community signals.
Those live under `data/processed/` and `data/demo/`, are forecast in
`reports/forecasting/`, and are never combined with anything here. The
synthetic marketing lab writes only to `reports/mmm/`.

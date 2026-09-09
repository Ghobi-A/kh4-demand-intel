# Model card — hierarchical behavioural-intent classifier

**This is a portfolio research prototype and not a production commercial
demand model.**

## Intended use

- Demonstrating supervised NLP methodology (group-aware evaluation,
  calibration, threshold selection, ranking) on community-discussion
  intent classification.
- Prioritising public comments for *manual review* via actionable-signal
  ranking and active-learning queues.

## Out-of-scope use

- Commercial demand forecasting or revenue prediction.
- Automated decisions about individuals or content moderation.
- Any use treating the heuristic `demand_score` as a forecast.

## Architecture

- **Stage 1:** actionable vs general_discussion — TF-IDF (word 1–2-grams)
  + Logistic Regression (`class_weight="balanced"`), Platt-calibrated on
  validation, with a validation-selected decision threshold.
- **Stage 2:** intent class among actionable rows — same feature/model
  family. Alternatives benchmarked: majority, rule baseline, Linear SVM,
  word+char TF-IDF SVM; optional sentence-embedding and transformer
  interfaces exist behind extras.
- The regex rule classifier (`src/intent.py`) remains the interpretable
  baseline; its 0.7/0.3 outputs are confidence indicators, not
  probabilities.

## Training data

200 manually corrected labels over public YouTube/Reddit comments (see
`docs/DATA_CARD.md`). Two taxonomy classes have zero examples.

## Evaluation status

`development_preliminary`. Metrics in `reports/ml/` come from repeated
seeded group-holdout splits of the 200-row development corpus and must
not be quoted as final performance. Final held-out evaluation is gated
on the corpus re-audit and expansion.

## Confidence and calibration

Stage 1 probabilities are Platt-calibrated on validation only; Brier
score and ECE are reported per run. Uncalibrated SVM margins are always
labelled as scores, never probabilities. With ~30 validation rows,
calibration quality is itself preliminary.

## Downstream use in forecasting

The calibrated Stage 1 probabilities feed the weekly behavioural demand proxy
(`actionable_probability_mass`). For rows inside the labelled corpus those
probabilities are computed **out-of-fold** with group-aware folds, so no row is
scored by a model that saw its own discussion group; other rows use the
persisted model. Provenance is recorded per row.

Forecasting models built on that proxy are documented separately in
`docs/FORECASTING.md`. They forecast observable discussion behaviour, never
sales: Kingdom Hearts IV is unreleased and no purchase conversion exists.

A separate synthetic Bayesian marketing mix model (`docs/MMM_LAB.md`) is fitted
only to simulated data and shares no inputs with this classifier.

## Bias, drift, and limitations

- Inherits the platform/selection/topic biases in the data card;
  performance on Reddit text is effectively unmeasured.
- Offline distribution-shift analysis (`reports/ml/drift_report.md`)
  covers split and source shifts; there is no production drift
  detection.
- With 23 discussion groups, split composition dominates variance:
  per-seed standard deviations are large and single-seed numbers are not
  meaningful.

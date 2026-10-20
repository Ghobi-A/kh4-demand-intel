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
score and ECE are reported per run. This includes logistic regression,
whose native probabilities were previously used uncalibrated: with
`class_weight="balanced"` they averaged 0.46 out of fold against a 0.30
base rate (ECE 0.21). After validation Platt scaling the out-of-fold ECE
is about 0.07 (`reports/ml/group_cv_report.md`).
`ExperimentConfig.calibrate_probability_outputs = False` restores the old
behaviour, and artefacts trained before the change are retrained rather
than trusted. Uncalibrated SVM margins are always
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

## Group-aware cross-validation (development)

`python -m src.ml.group_cv` scores every labelled row out of fold (whole
discussion groups per fold, nested validation for calibration and
threshold). Mean over three fold seeds:

| Model | Stage 1 PR-AUC (group-bootstrap 95%) | Lift@10% | Brier | ECE |
|---|---|---|---|---|
| char_word_svm | 0.65 (0.55–0.77) | 2.5 | 0.16 | 0.07 |
| logistic (deployed) | 0.57 (0.47–0.69) | 2.1 | 0.18 | 0.07 |

Base rate 0.30. `high_intent` F1 is 0 out of fold. The deployed model is not
switched on this evidence; choose between the two under a frozen protocol on
the expanded corpus.

The paired comparison (`python -m src.ml.model_comparison --with-embedding`,
`reports/ml/evaluation_report.md`) adds the rule baseline, a word-only linear
SVM and a frozen MiniLM encoder on identical folds. word+char SVM is supported
as better than logistic and than word-only SVM. **No supervised model is
supported as better than the regex rules**, on the sample or design-weighted,
and the sample's stratification on the rule output limits what that comparison
can show either way.

## Bias, drift, and limitations

- Inherits the platform/selection/topic biases in the data card;
  performance on Reddit text is effectively unmeasured.
- Offline distribution-shift analysis (`reports/ml/drift_report.md`)
  covers split and source shifts; there is no production drift
  detection.
- Long comments (>25 words) have a 52% Stage 1 false-positive rate against 9%
  for short ones (char_word_svm, out of fold): length is a confounder.
- Labels come from one annotator; first-pass and final labels agree on 57%
  of rows (κ 0.30). Inter-annotator agreement is unmeasured.
- With 23 discussion groups, split composition dominates variance:
  per-seed standard deviations are large and single-seed numbers are not
  meaningful.

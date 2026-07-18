# Model comparison

> **Preliminary development benchmark.** These numbers come from the current 200-row audit corpus, which is still being re-audited and expanded. They are not final held-out performance and must not be quoted as production or CV headline metrics.

- Run: `f1921ed69f29` at 2026-07-18T17:38:39.543911+00:00
- Git commit: `886e6f6`
- Dataset hash: `450745418479c703` (200 rows, annotation `audit_v1_2025`)
- Seeds: [42, 43, 44, 45, 46] (seed scope: split_and_initialisation)
- Evaluation mode: `group_holdout`
- Evaluation status: `development_preliminary`
- Split rows (seed 42): train 138 / validation 33 / test 29
- Split groups: train 15 / validation 4 / test 4

Values are mean ± standard deviation over seeds (each seed regenerates the group split and retrains the model).

| Model | Stage 1 F1 | Stage 1 PR-AUC | Stage 2 macro-F1 | Flat macro-F1 | End-to-end macro-F1 | Lift@10% |
|---|---|---|---|---|---|---|
| char_word_svm | 0.643 ± 0.059 | 0.751 ± 0.075 | 0.367 ± 0.154 | 0.290 ± 0.037 | 0.327 ± 0.075 | 2.941 ± 0.416 |
| linear_svm | 0.655 ± 0.091 | 0.740 ± 0.098 | 0.313 ± 0.134 | 0.283 ± 0.033 | 0.304 ± 0.066 | 2.941 ± 0.416 |
| logistic | 0.622 ± 0.055 | 0.719 ± 0.080 | 0.318 ± 0.114 | 0.332 ± 0.073 | 0.308 ± 0.048 | 2.595 ± 0.833 |
| majority | 0.473 ± 0.087 | 0.313 ± 0.074 | 0.054 ± 0.057 | 0.173 ± 0.028 | 0.017 ± 0.022 | 0.164 ± 0.366 |
| rules | 0.650 ± 0.076 | 0.575 ± 0.061 | 0.273 ± 0.179 | 0.414 ± 0.159 | 0.414 ± 0.159 | 2.539 ± 0.627 |

## Reading notes

- Stage 1 = actionable vs general_discussion; PR-AUC and F1 are prioritised over accuracy because actionable signals are the minority.
- Stage 2 macro-F1 is measured on true-actionable test rows only.
- End-to-end macro-F1 routes every test row through Stage 1 then Stage 2.
- The rule baseline emits a fixed confidence indicator, not a probability, so probability-dependent metrics are blank for it.
- Lift@10% = how many times more actionable signals a reviewer finds in the top 10% model-ranked comments than under random review.

# Portfolio summary

## What was built

Built a hierarchical supervised NLP benchmark comparing rule-based
classification with TF-IDF Logistic Regression and Linear SVM (plus
majority and word+char-TF-IDF baselines) using group-aware evaluation
infrastructure: permalink-recovered discussion-group holdout splits with
automated leakage tests, validation-only calibration and threshold
selection, repeated-seed evaluation, ranking metrics (Precision@K,
Recall@K, Lift@K), active-learning queue generation, rule-vs-model
disagreement analysis, structured error analysis, offline
distribution-shift analysis, model persistence, unified inference, and a
model-evaluation dashboard.

## Development results (preliminary — not CV metrics)

> **Preliminary development benchmark** on the current 200-row audit
> corpus (23 discussion groups, 5 seeds, group holdout). These numbers
> exist to validate the pipeline, not to describe final performance.
> See `reports/ml/results_summary.csv` and
> `reports/ml/model_comparison.md` for the generated tables.

Headline observations from the current development run:

- The interpretable rule baseline still leads end-to-end macro-F1 on
  this tiny corpus, while SVM-family models lead Stage 1 actionable
  ranking (PR-AUC and Lift@10% ≈ 2.5–3× over random review).
- Per-seed standard deviations are large because 23 groups dominate
  split variance — exactly why the corpus must be expanded before any
  number is quoted.

## FINAL CV METRICS

**Pending final re-audit and frozen held-out test evaluation.**

After the corpus is re-audited and expanded and a final test set is
frozen, re-run `python -m src.ml.benchmark` with
`evaluation_status = final_held_out` and populate:

- End-to-end macro-F1 (model vs rule baseline): _pending_
- Actionable precision / recall: _pending_
- Stage 1 PR-AUC: _pending_
- Lift@10% / Precision@10%: _pending_

Target CV bullet shape (do not fill in with preliminary numbers):
"Improved macro-F1 from X to Y versus the rule baseline and achieved
Z× Lift@10% for identifying actionable behavioural signals."

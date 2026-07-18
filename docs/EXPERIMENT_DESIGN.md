# Experiment design

## Why random row splitting is insufficient

Comments from the same YouTube video or Reddit thread share topic,
vocabulary, and conversational context. A random row split places
near-duplicate context on both sides of the train/test boundary, so
measured performance partly reflects memorised discussion context rather
than generalisation to new discussions.

## Group holdout methodology

Real discussion-group IDs are recovered deterministically from platform
permalinks (`yt:<video_id>`, `rd:<submission_id>`); provenance is
recorded in `group_id_source`. Groups are shuffled by seed and assigned
greedily to train/validation/test (default 70/15/15) so that no group
crosses splits. Automated tests assert zero group overlap and zero
duplicate-text overlap across splits (`tests/ml/test_splits.py`).

The generated portfolio-display parent IDs produced by
`scripts/build_portfolio_dataset.py` are never used as ML grouping
identifiers — only permalink-derived or provided platform IDs are.

## Temporary development splitting

Rows without a recoverable group ID receive per-row fallback groups and
the split is marked `evaluation_mode = development_only`; final
evaluation requires recovered group metadata. The current audit corpus
recovers group IDs for all 200 rows, so splits run as `group_holdout`,
but with only 23 groups the split composition varies strongly by seed —
another reason results are development-preliminary.

Temporal holdout (`train earlier / test later`) is implemented but
secondary; the current audit rows lack timestamps, so it is inactive
until `created_at` is recovered.

## Hierarchical classification

Stage 1 classifies actionable vs general_discussion; Stage 2 classifies
intent among actionable rows. A flat multiclass benchmark runs alongside
for comparison, and end-to-end hierarchical macro-F1 is reported against
both the rule baseline and flat models.

## Validation-only tuning, threshold selection, test isolation

Hyperparameters are fixed conservatively (no large grids on 200 labels).
Calibration (Platt by default; isotonic available for larger data) and
decision-threshold selection (max-F1 or precision/recall-floor
objectives) are fitted on validation data only; the threshold is frozen
before the test split is touched. The test split is used exactly once
per run for reporting.

## Class imbalance

Classical models use `class_weight="balanced"`; Stage 1 additionally uses
threshold adjustment. Oversampling is not enabled by default and would
apply to training data only; synthetic text augmentation and SMOTE-style
interpolation are not used (interpolating sparse TF-IDF vectors does not
produce valid text distributions).

## Repeated seeds

Each seed regenerates the group split **and** model initialisation
(`seed_scope = split_and_initialisation`). Per-seed metrics are saved to
`reports/ml/results_per_seed.csv`; means, standard deviations, and
medians to `reports/ml/results_summary.csv`. The seed spread describes
variability of this pipeline on this corpus — it is not a population
confidence interval.

## Ranking metrics

Actionable-probability ranking is evaluated with Precision@K, Recall@K,
and Lift@K at absolute (10/25/50/100) and percentage (5/10/20%)
cut-offs, plus cumulative-gains and lift curves. Lift@10% answers: how
many times more actionable signals does a reviewer find in the top 10%
model-ranked comments than under random review? Only measured lift is
reported.

## Active learning

`python -m src.ml.active_learning` generates a review queue prioritising
rule/model disagreement, model/model disagreement, near-threshold
uncertainty, rare predicted classes, underrepresented sources/groups,
and a random control sample. The queue never overwrites labelled data.

## Preliminary vs final evaluation

Every run records `evaluation_status`. It stays
`development_preliminary` until the corpus is re-audited and expanded, a
final test set is frozen, and the same benchmark is re-run — at which
point `final_held_out` results can populate the gated CV section in
`reports/portfolio_summary.md`.

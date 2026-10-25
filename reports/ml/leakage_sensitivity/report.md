# Evaluation report: actionable-signal detection

> **Executed, development-preliminary results.** Every number below was produced by `python -m src.ml.model_comparison` on 2026-10-25 over the 200-row single-annotator audit corpus (22 discussion groups, dataset hash `450745418479c703`). It is cross-validated development evaluation, not final held-out performance: the taxonomy, rules and model families were designed around these rows.

Execution mode: `refit`. Prediction replay recomputes metrics and intervals from saved OOF scores; it does not verify encoder downloads or model refitting. Package versions and OOF SHA-256 are recorded in the companion summary.json.
Near-duplicate-connected groups linked before splitting: **yes**. Original PR #26 scores hold out discussion IDs but contain one cross-group near-duplicate pair; consult the separate linked-group sensitivity before interpreting the ordering.

## Protocol

- 5-fold **group** cross-validation (whole YouTube videos / Reddit threads held out), repeated over fold seeds [42, 43, 44]; every model sees identical folds.
- Inside each training fold, a group-held-out validation split fits Platt calibration and the Stage 1 threshold; the held-out fold is touched only for scoring.
- TF-IDF vocabularies and model weights are fitted per fold (scikit-learn pipelines).
- Uncertainty: 1000 paired group-bootstrap replicates (groups resampled once per replicate, all models scored on the same resample, averaged over fold seeds). These intervals condition on the fitted OOF predictions; they do not refit models in each replicate or account for model-selection uncertainty.
- Relevance target for ranking metrics: actionable vs general discussion (base rate 0.30, 60 actionable rows). Ties are broken in expectation.

## Headline: Stage 1 ranking and calibration

Mean over fold seeds; 95% group-bootstrap interval in brackets. PR-AUC of a random ranking equals the base rate (0.30); Lift@k of random review is 1.

| Model | PR-AUC | Lift@20 (top 10%) | Lift@60 (= #positives) | Brier | ECE (reference seed) |
|---|---|---|---|---|---|
| majority | 0.283 [0.21–0.35] | 0.88 [0.51–1.08] | 0.89 [0.64–1.07] | 0.228 [0.196–0.259] | 0.070 |
| rules | 0.634 [0.54–0.74] | 2.51 [2.00–3.24] | 2.39 [1.71–2.75] | 0.148 [0.115–0.182] | 0.122 |
| logistic | 0.515 [0.42–0.61] | 2.22 [1.83–2.57] | 1.65 [1.34–1.95] | 0.200 [0.169–0.237] | 0.114 |
| linear_svm | 0.537 [0.44–0.63] | 2.39 [1.85–2.63] | 1.76 [1.39–2.01] | 0.193 [0.163–0.228] | 0.092 |
| char_word_svm | 0.598 [0.50–0.68] | 2.39 [2.03–2.97] | 1.83 [1.50–2.18] | 0.182 [0.155–0.213] | 0.098 |

## Sampling design and what it does to these numbers

The audit corpus is **stratified on the rule baseline's own prediction**: 100 rows where the rule said `general_discussion` and 20 rows for each class the rule predicted (`scripts/sample_intent_audit.py`). Two consequences:

- Oversampling rule-positive strata changes class-wise precision, recall and F1. This can favour the rules; the direction and size of bias cannot be established without the original sampling frame. Rules-vs-model comparisons are conditional on this design, not population estimates.
- Actionable rows are over-represented (sample base rate 0.30).

**Design-weighted sensitivity analysis (approximate).** Weighting each row by its stratum's assumed population share over its sample count explores sensitivity, using rule-label shares read from `reports/figures/intent_distribution.png (2026-04-27)`, rounded to 0.1%. The rule set may have changed since that chart was made. Estimated population prevalence of actionable comments: **0.15**. Effective sample size 124 of 200.

| Model | Weighted PR-AUC [95%] | Weighted Brier [95%] |
|---|---|---|
| majority | 0.150 [0.09–0.22] | 0.161 [0.135–0.190] |
| rules | 0.315 [0.20–0.49] | 0.128 [0.091–0.174] |
| logistic | 0.342 [0.21–0.50] | 0.143 [0.120–0.169] |
| linear_svm | 0.360 [0.22–0.52] | 0.139 [0.115–0.165] |
| char_word_svm | 0.402 [0.25–0.57] | 0.132 [0.105–0.162] |

A random ranking's weighted PR-AUC is about 0.15. These weighted figures are illustrative scenarios, not corrected population performance. Rounded chart values do not recover verified sampling probabilities or representativeness.

## Which differences are supported

Paired differences (row model minus comparison model). A difference is called only when its 95% interval excludes zero; for Brier lower is better.
These are exploratory pointwise intervals across multiple comparisons, without a family-wise correction. An interval excluding zero is a follow-up hypothesis, not a confirmatory model-selection result. Do not switch the deployed model on this table.

| model | versus | metric | mean_difference | ci_low | ci_high | verdict |
|---|---|---|---|---|---|---|
| majority | rules | pr_auc | -0.347 | -0.443 | -0.255 | rules better (interval excludes 0) |
| majority | rules | lift_at_20 | -1.661 | -2.469 | -1.088 | rules better (interval excludes 0) |
| majority | rules | brier | +0.080 | +0.052 | +0.111 | rules better (interval excludes 0) |
| majority | rules | weighted_pr_auc | -0.172 | -0.323 | -0.064 | rules better (interval excludes 0) |
| logistic | rules | pr_auc | -0.113 | -0.238 | +0.004 | not distinguishable |
| logistic | rules | lift_at_20 | -0.358 | -1.087 | +0.128 | not distinguishable |
| logistic | rules | brier | +0.053 | +0.018 | +0.092 | rules better (interval excludes 0) |
| logistic | rules | weighted_pr_auc | +0.034 | -0.114 | +0.164 | not distinguishable |
| linear_svm | rules | pr_auc | -0.091 | -0.217 | +0.029 | not distinguishable |
| linear_svm | rules | lift_at_20 | -0.312 | -1.053 | +0.167 | not distinguishable |
| linear_svm | rules | brier | +0.046 | +0.010 | +0.084 | rules better (interval excludes 0) |
| linear_svm | rules | weighted_pr_auc | +0.048 | -0.116 | +0.180 | not distinguishable |
| char_word_svm | rules | pr_auc | -0.033 | -0.157 | +0.088 | not distinguishable |
| char_word_svm | rules | lift_at_20 | -0.087 | -0.820 | +0.506 | not distinguishable |
| char_word_svm | rules | brier | +0.034 | +0.001 | +0.069 | rules better (interval excludes 0) |
| char_word_svm | rules | weighted_pr_auc | +0.091 | -0.070 | +0.242 | not distinguishable |
| linear_svm | logistic | pr_auc | +0.022 | +0.010 | +0.031 | linear_svm better (interval excludes 0) |
| linear_svm | logistic | lift_at_20 | +0.046 | -0.145 | +0.251 | not distinguishable |
| linear_svm | logistic | brier | -0.007 | -0.011 | -0.004 | linear_svm better (interval excludes 0) |
| linear_svm | logistic | weighted_pr_auc | +0.014 | -0.009 | +0.036 | not distinguishable |
| char_word_svm | logistic | pr_auc | +0.080 | +0.045 | +0.112 | char_word_svm better (interval excludes 0) |
| char_word_svm | logistic | lift_at_20 | +0.271 | -0.057 | +0.689 | not distinguishable |
| char_word_svm | logistic | brier | -0.019 | -0.029 | -0.009 | char_word_svm better (interval excludes 0) |
| char_word_svm | logistic | weighted_pr_auc | +0.058 | +0.017 | +0.104 | char_word_svm better (interval excludes 0) |
| char_word_svm | linear_svm | pr_auc | +0.058 | +0.027 | +0.088 | char_word_svm better (interval excludes 0) |
| char_word_svm | linear_svm | lift_at_20 | +0.225 | -0.114 | +0.703 | not distinguishable |
| char_word_svm | linear_svm | brier | -0.011 | -0.021 | -0.002 | char_word_svm better (interval excludes 0) |
| char_word_svm | linear_svm | weighted_pr_auc | +0.043 | +0.014 | +0.074 | char_word_svm better (interval excludes 0) |

15 of 28 paired comparisons are supported at this sample size. Anything marked *not distinguishable* is not evidence that one model is better.

## Review-queue ranking at fixed budgets

If a reviewer reads only the top k of these 200 comments: expected precision, recall and lift (mean over fold seeds).

| Model | P@10 / R@10 / Lift@10 | P@20 / R@20 / Lift@20 | P@30 / R@30 / Lift@30 | P@60 / R@60 / Lift@60 |
|---|---|---|---|---|
| majority | 0.26 / 0.04 / 0.88 | 0.26 / 0.09 / 0.88 | 0.26 / 0.13 / 0.88 | 0.27 / 0.27 / 0.89 |
| rules | 0.75 / 0.12 / 2.50 | 0.75 / 0.25 / 2.51 | 0.75 / 0.37 / 2.49 | 0.72 / 0.72 / 2.39 |
| logistic | 0.67 / 0.11 / 2.22 | 0.67 / 0.22 / 2.22 | 0.62 / 0.31 / 2.07 | 0.49 / 0.49 / 1.65 |
| linear_svm | 0.70 / 0.12 / 2.33 | 0.72 / 0.24 / 2.39 | 0.61 / 0.31 / 2.04 | 0.53 / 0.53 / 1.76 |
| char_word_svm | 0.80 / 0.13 / 2.67 | 0.72 / 0.24 / 2.39 | 0.68 / 0.34 / 2.26 | 0.55 / 0.55 / 1.83 |

## Class-wise end-to-end results

Mean over fold seeds. Support is the full corpus count. Two classes have no labels; classes under ~10 rows move by large steps with one prediction. **The `rules` column is conditional on the sampling design above** (for example, 17 of 18 `confusion_barrier` rows were sampled because the rule predicted that class). `majority` routes every row to Stage 2: its constant Stage 1 score, thresholded for F1 on validation, labels everything actionable.

| Class | Support | majority F1 | rules F1 | logistic F1 | linear_svm F1 | char_word_svm F1 |
|---|---|---|---|---|---|---|
| high_intent | 7 | 0.00 | 0.30 | 0.00 | 0.00 | 0.00 |
| nostalgia_reactivation | 8 | 0.00 | 0.18 | 0.21 | 0.27 | 0.33 |
| new_customer_interest | 2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| frustrated_demand | 25 | 0.21 | 0.56 | 0.35 | 0.36 | 0.38 |
| content_drought_fatigue | 0 | — | — | — | — | — |
| confusion_barrier | 18 | 0.00 | 0.94 | 0.23 | 0.17 | 0.36 |
| expectation_decay | 0 | — | — | — | — | — |
| general_discussion | 140 | 0.00 | 0.88 | 0.75 | 0.75 | 0.81 |

## Confusion matrix: `rules` (reference seed 42, rows = true label)

| true \ predicted | high_intent | nostalgia_reactivation | new_customer_interest | frustrated_demand | confusion_barrier | general_discussion | content_drought_fatigue | expectation_decay |
|---|---|---|---|---|---|---|---|---|
| high_intent | 3 | 0 | 0 | 0 | 0 | 2 | 0 | 2 |
| nostalgia_reactivation | 2 | 1 | 0 | 0 | 0 | 5 | 0 | 0 |
| new_customer_interest | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| frustrated_demand | 3 | 0 | 0 | 12 | 0 | 9 | 1 | 0 |
| confusion_barrier | 0 | 0 | 0 | 0 | 17 | 1 | 0 | 0 |
| general_discussion | 5 | 2 | 0 | 6 | 1 | 124 | 1 | 1 |

## Error analysis (reference seed)

Full error table: `reports/ml/model_comparison/errors.csv` (every Stage 1 error with text, group, rule label, probability and an empty `error_category` column for human review).

| Model | False positives | False negatives |
|---|---|---|
| majority | 140 | 0 |
| rules | 16 | 17 |
| logistic | 53 | 15 |
| linear_svm | 58 | 14 |
| char_word_svm | 31 | 21 |

**Errors every supervised model makes** (13 rows): these are where more labelled data or a label re-audit, not a different classifier, is needed.

### Most confident false positives (`rules`)

| P(actionable) | True | Rule | All supervised wrong | Words | Text |
|---|---|---|---|---|---|
| 0.89 | general_discussion | confusion_barrier | no | 34 | Maybe it'll be an opportunity to simplify the lore, which is needed. I don't think KH is too complicated, but it's needlessly complex and c… |
| 0.89 | general_discussion | high_intent | yes | 24 | All the remasters and remakes had their difficulty nerfed big time. There is nothing like just buying an actual ps2 and playing the origina… |
| 0.89 | general_discussion | high_intent | no | 39 | Played a little bit of kingdom hearts 1 when I was young in the PS2. Bought it all for Xbox, but have 2 kids. Thinking about jumping back i… |
| 0.80 | general_discussion | nostalgia_reactivation | no | 28 | Kingdom Hearts 2 always be my favorite game in the Kingdom Hearts games. Kingdom Hearts is so special to me been in the fandom since I was … |
| 0.80 | general_discussion | frustrated_demand | yes | 21 | This game’s lore was way too complicated for my 12 year old brain to be irrationally obsessed with. What the hell. |
| 0.80 | general_discussion | frustrated_demand | no | 26 | Bluett in regard to KH2’s release: “This anticipation however was also tinged with a certain frustration at the long wait.” *Me looking at … |
| 0.80 | general_discussion | expectation_decay | yes | 61 | I hope dark road finale really gives us xehanort true reasoning for throwing the words into darkness , I'm not buying that reasoning in kh3… |
| 0.80 | general_discussion | high_intent | yes | 60 | I was in 7th grade when I first started seeing videos of people playing KH2FM. I was such an unhinged kid that I saved up my allowance, bou… |

### Most confident false negatives (`rules`)

| P(actionable) | True | Rule | All supervised wrong | Words | Text |
|---|---|---|---|---|---|
| 0.06 | confusion_barrier | general_discussion | no | 41 | Currently going through 1.5FM and was looking on consensus what to play in what order. Havent played 1 and 2 since launch, Chain of Memorie… |
| 0.06 | high_intent | general_discussion | yes | 19 | 23:40 “the two upcoming games in the series that we 100% know are coming out in the future” 😅 |
| 0.15 | frustrated_demand | general_discussion | no | 81 | It's crazy that we're getting all the 7 Remake Project being completed in between KH3 and KH4. Starting to get to me ngl. Starting to hate … |
| 0.15 | frustrated_demand | general_discussion | yes | 19 | How many Kingdom Hearts fans died while waiting for KH3? How many have died since the KH4 trailer dropped? |
| 0.15 | frustrated_demand | general_discussion | no | 57 | I don't mind the huge hiatus-like gaps, so long as the game is good. KH3 was fine. Better than fine for me, but even that gap was too long … |
| 0.15 | frustrated_demand | general_discussion | no | 69 | Been a huge fan of the series since 2006 and been following this channel since 2013 (when 1.5 released on PS3). Always been someone with an… |
| 0.15 | frustrated_demand | general_discussion | yes | 7 | Kingdom Hearts 4: 1,000 Days of Silence |
| 0.15 | nostalgia_reactivation | general_discussion | yes | 15 | I’m 40. I remember being a teenager when KH1 came out. I am ancient. 😂 |

## What these results do and do not support

- They describe ranking and classification of *public discussion* by behavioural intent. They are not purchase, sales or revenue predictions.
- With 23 groups, a single well-represented video can move every metric; the bootstrap intervals above are the honest width of that uncertainty.
- Labels come from one annotator (first-pass vs final κ 0.30); label noise bounds every number here. `docs/ANNOTATION_PROTOCOL.md` describes the independent double-annotation and adjudication needed before a final evaluation.
- Per-group stability, calibration bins and slice-level error rates: `reports/ml/group_cv_report.md`.

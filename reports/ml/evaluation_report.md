# Evaluation report: actionable-signal detection

> **Executed, development-preliminary results.** Every number below was produced by `python -m src.ml.model_comparison --with-embedding` on 2026-10-20 over the 200-row single-annotator audit corpus (23 discussion groups, dataset hash `450745418479c703`). It is cross-validated development evaluation, not final held-out performance: the taxonomy, rules and model families were designed around these rows.

## Protocol

- 5-fold **group** cross-validation (whole YouTube videos / Reddit threads held out), repeated over fold seeds [42, 43, 44]; every model sees identical folds.
- Inside each training fold, a group-held-out validation split fits Platt calibration and the Stage 1 threshold; the held-out fold is touched only for scoring.
- TF-IDF vocabularies and model weights are fitted per fold (scikit-learn pipelines).
- Uncertainty: 1000 paired group-bootstrap replicates (groups resampled once per replicate, all models scored on the same resample, averaged over fold seeds).
- Relevance target for ranking metrics: actionable vs general discussion (base rate 0.30, 60 actionable rows). Ties are broken in expectation.
- `embedding`: frozen `sentence-transformers/all-MiniLM-L6-v2` (revision `1110a24`) sentence embeddings + class-balanced logistic head. The encoder is not fine-tuned.

## Headline: Stage 1 ranking and calibration

Mean over fold seeds; 95% group-bootstrap interval in brackets. PR-AUC of a random ranking equals the base rate (0.30); Lift@k of random review is 1.

| Model | PR-AUC | Lift@20 (top 10%) | Lift@60 (= #positives) | Brier | ECE |
|---|---|---|---|---|---|
| majority | 0.308 [0.23–0.37] | 1.10 [0.85–1.28] | 1.01 [0.86–1.18] | 0.212 [0.187–0.234] | 0.089 |
| rules | 0.616 [0.52–0.72] | 2.27 [1.78–3.17] | 2.40 [1.71–2.79] | 0.145 [0.111–0.180] | 0.099 |
| logistic | 0.568 [0.47–0.67] | 2.11 [1.79–2.58] | 1.89 [1.54–2.20] | 0.176 [0.147–0.203] | 0.072 |
| linear_svm | 0.585 [0.49–0.69] | 2.28 [1.92–2.65] | 1.91 [1.60–2.22] | 0.172 [0.144–0.199] | 0.062 |
| char_word_svm | 0.653 [0.56–0.75] | 2.50 [2.06–2.89] | 2.11 [1.74–2.41] | 0.157 [0.134–0.179] | 0.051 |
| embedding | 0.493 [0.42–0.58] | 2.00 [1.48–2.69] | 1.59 [1.22–1.94] | 0.206 [0.165–0.249] | 0.139 |

## Sampling design and what it does to these numbers

The audit corpus is **stratified on the rule baseline's own prediction**: 100 rows where the rule said `general_discussion` and 20 rows for each class the rule predicted (`scripts/sample_intent_audit.py`). Two consequences:

- Rule recall on a class is close to 1 *by construction*: a row of that class only enters the sample without the rule firing if it comes from the 100-row `general_discussion` stratum. The rule's class-wise F1 below is therefore inflated, and rules-vs-model comparisons are conditional on this design.
- Actionable rows are over-represented (sample base rate 0.30).

**Design-weighted sensitivity analysis (approximate).** Weighting each row by its stratum's population share over its sample count undoes the stratification, using rule-label shares read from `reports/figures/intent_distribution.png (2026-04-27)`, rounded to 0.1%. The rule set may have changed since that chart was made. Estimated population prevalence of actionable comments: **0.15**. Effective sample size 124 of 200.

| Model | Weighted PR-AUC [95%] | Weighted Brier [95%] |
|---|---|---|
| majority | 0.157 [0.10–0.23] | 0.165 [0.143–0.187] |
| rules | 0.308 [0.20–0.49] | 0.129 [0.091–0.175] |
| logistic | 0.371 [0.23–0.54] | 0.142 [0.115–0.172] |
| linear_svm | 0.368 [0.23–0.53] | 0.141 [0.113–0.171] |
| char_word_svm | 0.426 [0.27–0.61] | 0.132 [0.104–0.166] |
| embedding | 0.321 [0.22–0.49] | 0.151 [0.119–0.183] |

A random ranking's weighted PR-AUC is about 0.15. These weighted figures are the closer guide to performance on unselected comments, but they rest on rounded chart values and a small effective sample.

## Which differences are supported

Paired differences (row model minus comparison model). A difference is called only when its 95% interval excludes zero; for Brier lower is better.

| model | versus | metric | mean_difference | ci_low | ci_high | verdict |
|---|---|---|---|---|---|---|
| majority | rules | pr_auc | -0.313 | -0.409 | -0.228 | rules better (interval excludes 0) |
| majority | rules | lift_at_20 | -1.299 | -2.087 | -0.776 | rules better (interval excludes 0) |
| majority | rules | brier | +0.068 | +0.039 | +0.095 | rules better (interval excludes 0) |
| majority | rules | weighted_pr_auc | -0.165 | -0.303 | -0.069 | rules better (interval excludes 0) |
| logistic | rules | pr_auc | -0.049 | -0.172 | +0.070 | not distinguishable |
| logistic | rules | lift_at_20 | -0.242 | -0.945 | +0.289 | not distinguishable |
| logistic | rules | brier | +0.032 | -0.002 | +0.065 | not distinguishable |
| logistic | rules | weighted_pr_auc | +0.060 | -0.059 | +0.168 | not distinguishable |
| linear_svm | rules | pr_auc | -0.031 | -0.150 | +0.082 | not distinguishable |
| linear_svm | rules | lift_at_20 | -0.113 | -0.760 | +0.405 | not distinguishable |
| linear_svm | rules | brier | +0.027 | -0.007 | +0.061 | not distinguishable |
| linear_svm | rules | weighted_pr_auc | +0.059 | -0.056 | +0.167 | not distinguishable |
| char_word_svm | rules | pr_auc | +0.035 | -0.074 | +0.136 | not distinguishable |
| char_word_svm | rules | lift_at_20 | +0.080 | -0.550 | +0.554 | not distinguishable |
| char_word_svm | rules | brier | +0.012 | -0.019 | +0.040 | not distinguishable |
| char_word_svm | rules | weighted_pr_auc | +0.117 | -0.000 | +0.246 | not distinguishable |
| embedding | rules | pr_auc | -0.120 | -0.253 | +0.005 | not distinguishable |
| embedding | rules | lift_at_20 | -0.345 | -1.029 | +0.181 | not distinguishable |
| embedding | rules | brier | +0.061 | +0.021 | +0.108 | rules better (interval excludes 0) |
| embedding | rules | weighted_pr_auc | +0.014 | -0.160 | +0.170 | not distinguishable |
| linear_svm | logistic | pr_auc | +0.018 | +0.001 | +0.036 | linear_svm better (interval excludes 0) |
| linear_svm | logistic | lift_at_20 | +0.129 | -0.065 | +0.356 | not distinguishable |
| linear_svm | logistic | brier | -0.005 | -0.008 | -0.001 | linear_svm better (interval excludes 0) |
| linear_svm | logistic | weighted_pr_auc | -0.001 | -0.020 | +0.022 | not distinguishable |
| char_word_svm | logistic | pr_auc | +0.084 | +0.034 | +0.133 | char_word_svm better (interval excludes 0) |
| char_word_svm | logistic | lift_at_20 | +0.322 | +0.055 | +0.595 | char_word_svm better (interval excludes 0) |
| char_word_svm | logistic | brier | -0.019 | -0.032 | -0.008 | char_word_svm better (interval excludes 0) |
| char_word_svm | logistic | weighted_pr_auc | +0.057 | +0.001 | +0.114 | char_word_svm better (interval excludes 0) |
| embedding | logistic | pr_auc | -0.071 | -0.138 | +0.007 | not distinguishable |
| embedding | logistic | lift_at_20 | -0.103 | -0.615 | +0.541 | not distinguishable |
| embedding | logistic | brier | +0.030 | +0.010 | +0.050 | logistic better (interval excludes 0) |
| embedding | logistic | weighted_pr_auc | -0.046 | -0.151 | +0.076 | not distinguishable |
| char_word_svm | linear_svm | pr_auc | +0.066 | +0.023 | +0.108 | char_word_svm better (interval excludes 0) |
| char_word_svm | linear_svm | lift_at_20 | +0.193 | -0.055 | +0.452 | not distinguishable |
| char_word_svm | linear_svm | brier | -0.015 | -0.026 | -0.005 | char_word_svm better (interval excludes 0) |
| char_word_svm | linear_svm | weighted_pr_auc | +0.058 | +0.013 | +0.102 | char_word_svm better (interval excludes 0) |
| embedding | linear_svm | pr_auc | -0.089 | -0.162 | -0.010 | linear_svm better (interval excludes 0) |
| embedding | linear_svm | lift_at_20 | -0.232 | -0.750 | +0.404 | not distinguishable |
| embedding | linear_svm | brier | +0.034 | +0.014 | +0.055 | linear_svm better (interval excludes 0) |
| embedding | linear_svm | weighted_pr_auc | -0.045 | -0.158 | +0.079 | not distinguishable |
| embedding | char_word_svm | pr_auc | -0.155 | -0.230 | -0.074 | char_word_svm better (interval excludes 0) |
| embedding | char_word_svm | lift_at_20 | -0.425 | -0.914 | +0.143 | not distinguishable |
| embedding | char_word_svm | brier | +0.049 | +0.024 | +0.078 | char_word_svm better (interval excludes 0) |
| embedding | char_word_svm | weighted_pr_auc | -0.103 | -0.240 | +0.040 | not distinguishable |

19 of 44 paired comparisons are supported at this sample size. Anything marked *not distinguishable* is not evidence that one model is better.

## Review-queue ranking at fixed budgets

If a reviewer reads only the top k of these 200 comments: expected precision, recall and lift (mean over fold seeds).

| Model | P@10 / R@10 / Lift@10 | P@20 / R@20 / Lift@20 | P@30 / R@30 / Lift@30 | P@60 / R@60 / Lift@60 |
|---|---|---|---|---|
| majority | 0.33 / 0.05 / 1.10 | 0.33 / 0.11 / 1.10 | 0.33 / 0.16 / 1.10 | 0.30 / 0.30 / 1.01 |
| rules | 0.62 / 0.10 / 2.07 | 0.68 / 0.23 / 2.27 | 0.74 / 0.37 / 2.45 | 0.72 / 0.72 / 2.40 |
| logistic | 0.60 / 0.10 / 2.00 | 0.63 / 0.21 / 2.11 | 0.63 / 0.32 / 2.11 | 0.57 / 0.57 / 1.89 |
| linear_svm | 0.63 / 0.11 / 2.11 | 0.68 / 0.23 / 2.28 | 0.66 / 0.33 / 2.19 | 0.57 / 0.57 / 1.91 |
| char_word_svm | 0.80 / 0.13 / 2.67 | 0.75 / 0.25 / 2.50 | 0.72 / 0.36 / 2.41 | 0.63 / 0.63 / 2.11 |
| embedding | 0.57 / 0.09 / 1.89 | 0.60 / 0.20 / 2.00 | 0.56 / 0.28 / 1.85 | 0.48 / 0.48 / 1.59 |

## Class-wise end-to-end results

Mean over fold seeds. Support is the full corpus count. Two classes have no labels; classes under ~10 rows move by large steps with one prediction. **The `rules` column is inflated by the sampling design above** (for example, 17 of 18 `confusion_barrier` rows were sampled because the rule predicted that class). `majority` routes every row to Stage 2: its constant Stage 1 score, thresholded for F1 on validation, labels everything actionable.

| Class | Support | majority F1 | rules F1 | logistic F1 | linear_svm F1 | char_word_svm F1 | embedding F1 |
|---|---|---|---|---|---|---|---|
| high_intent | 7 | 0.00 | 0.30 | 0.00 | 0.00 | 0.00 | 0.09 |
| nostalgia_reactivation | 8 | 0.00 | 0.18 | 0.07 | 0.07 | 0.28 | 0.34 |
| new_customer_interest | 2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| frustrated_demand | 25 | 0.20 | 0.56 | 0.39 | 0.41 | 0.46 | 0.40 |
| content_drought_fatigue | 0 | — | — | — | — | — | — |
| confusion_barrier | 18 | 0.02 | 0.94 | 0.29 | 0.25 | 0.35 | 0.24 |
| expectation_decay | 0 | — | — | — | — | — | — |
| general_discussion | 140 | 0.00 | 0.88 | 0.72 | 0.74 | 0.79 | 0.67 |

## Confusion matrix: `char_word_svm` (reference seed 42, rows = true label)

| true \ predicted | high_intent | nostalgia_reactivation | new_customer_interest | frustrated_demand | confusion_barrier | general_discussion |
|---|---|---|---|---|---|---|
| high_intent | 0 | 0 | 0 | 5 | 0 | 2 |
| nostalgia_reactivation | 0 | 2 | 0 | 3 | 0 | 3 |
| new_customer_interest | 0 | 0 | 0 | 1 | 0 | 1 |
| frustrated_demand | 0 | 0 | 0 | 22 | 1 | 2 |
| confusion_barrier | 0 | 0 | 0 | 6 | 12 | 0 |
| general_discussion | 0 | 0 | 0 | 36 | 11 | 93 |

## Error analysis (reference seed)

Full error table: `reports/ml/model_comparison/errors.csv` (every Stage 1 error with text, group, rule label, probability and an empty `error_category` column for human review).

| Model | False positives | False negatives |
|---|---|---|
| majority | 140 | 0 |
| rules | 44 | 12 |
| logistic | 32 | 23 |
| linear_svm | 32 | 25 |
| char_word_svm | 47 | 8 |
| embedding | 84 | 16 |

**Errors every supervised model makes** (25 rows): these are where more labelled data or a label re-audit, not a different classifier, is needed.

### Most confident false positives (`char_word_svm`)

| P(actionable) | True | Rule | All supervised wrong | Words | Text |
|---|---|---|---|---|---|
| 0.95 | general_discussion | frustrated_demand | no | 182 | I personally don't think the fact that we haven't had a KH4 re-reveal yet means the game won't be out in 2026. I have a feeling they'll kee… |
| 0.84 | general_discussion | general_discussion | yes | 133 | I think it's laughable that you didn't mention the reason behind all this is : FF Versus XIII. I don't know why KH fans are still so ignora… |
| 0.81 | general_discussion | general_discussion | yes | 37 | This game series is truly one of the best! I can't wait for KH4. I've played this game ever since it was available long ago on the PS2. The… |
| 0.80 | general_discussion | general_discussion | yes | 189 | not gonna lie i didnt even think this was on the table for "mysteries" but i swear to god i hope they NEVER show us the explanation for kin… |
| 0.74 | general_discussion | expectation_decay | yes | 61 | I hope dark road finale really gives us xehanort true reasoning for throwing the words into darkness , I'm not buying that reasoning in kh3… |
| 0.74 | general_discussion | general_discussion | yes | 264 | Potential Spoilers: Look this is cool and all, but I am soooooo hype for Kingdom Hearts getting Star Wars. If they do that really well, and… |
| 0.67 | general_discussion | general_discussion | no | 74 | I wonder if darkness and light were invited into the world by Sora's arrival Like did his presence cause that heartless to appear? Since th… |
| 0.63 | general_discussion | frustrated_demand | yes | 21 | This game’s lore was way too complicated for my 12 year old brain to be irrationally obsessed with. What the hell. |

### Most confident false negatives (`char_word_svm`)

| P(actionable) | True | Rule | All supervised wrong | Words | Text |
|---|---|---|---|---|---|
| 0.02 | frustrated_demand | general_discussion | yes | 7 | Kingdom Hearts 4: 1,000 Days of Silence |
| 0.06 | frustrated_demand | frustrated_demand | yes | 6 | See you in another 4 years. |
| 0.07 | new_customer_interest | new_customer_interest | yes | 41 | After 20 years of Kingdom Hearts, this is the best game, but I’ve never played it. I also wanna say: I hope more worlds will come to Kingdo… |
| 0.13 | nostalgia_reactivation | general_discussion | no | 2 | Sheesh nostalgia |
| 0.16 | high_intent | general_discussion | yes | 19 | fun fact the guy driving yozora at the end is 100% luxord's human form, which is a total brainfuck |
| 0.23 | nostalgia_reactivation | general_discussion | no | 15 | I loved your old KH video back then, this new one is just amazing. GOATED |
| 0.31 | high_intent | general_discussion | yes | 19 | 23:40 “the two upcoming games in the series that we 100% know are coming out in the future” 😅 |
| 0.34 | nostalgia_reactivation | general_discussion | no | 17 | 2:35:23 remember when Star Wars thought it was cool enough to pull off a Goofy Death Sequence? |

## What these results do and do not support

- They describe ranking and classification of *public discussion* by behavioural intent. They are not purchase, sales or revenue predictions.
- With 23 groups, a single well-represented video can move every metric; the bootstrap intervals above are the honest width of that uncertainty.
- Labels come from one annotator (first-pass vs final κ 0.30); label noise bounds every number here. `docs/ANNOTATION_PROTOCOL.md` describes the independent double-annotation and adjudication needed before a final evaluation.
- Per-group stability, calibration bins and slice-level error rates: `reports/ml/group_cv_report.md`.

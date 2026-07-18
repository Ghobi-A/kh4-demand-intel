# Offline distribution-shift analysis

> **Preliminary development benchmark.** These numbers come from the current 200-row audit corpus, which is still being re-audited and expanded. They are not final held-out performance and must not be quoted as production or CV headline metrics.

This is an offline diagnostic over the current development corpus, not production drift detection.

## train vs validation

- Rows: 138 vs 33
- Vocabulary overlap (top-term Jaccard): 0.338
- TF-IDF centroid cosine distance: 0.341
- Mean text length: 260 vs 261
- Max class-prevalence shift: 0.132

| Class | Prevalence shift (a − b) |
|---|---|
| frustrated_demand | -0.132 |
| confusion_barrier | +0.093 |
| general_discussion | +0.043 |
| new_customer_interest | -0.023 |
| nostalgia_reactivation | +0.013 |
| high_intent | +0.006 |

## train vs test

- Rows: 138 vs 29
- Vocabulary overlap (top-term Jaccard): 0.429
- TF-IDF centroid cosine distance: 0.343
- Mean text length: 260 vs 596
- Max class-prevalence shift: 0.162

| Class | Prevalence shift (a − b) |
|---|---|
| frustrated_demand | -0.162 |
| confusion_barrier | +0.123 |
| general_discussion | +0.021 |
| nostalgia_reactivation | +0.009 |
| new_customer_interest | +0.007 |
| high_intent | +0.002 |

> Warning: smallest slice has only 29 rows; differences here are not statistically meaningful.

## reddit vs youtube

- Rows: 3 vs 197
- Vocabulary overlap (top-term Jaccard): 0.361
- TF-IDF centroid cosine distance: 0.458
- Mean text length: 4631 vs 243
- Max class-prevalence shift: 0.550

| Class | Prevalence shift (a − b) |
|---|---|
| frustrated_demand | +0.550 |
| general_discussion | -0.372 |
| confusion_barrier | -0.091 |
| nostalgia_reactivation | -0.041 |
| high_intent | -0.035 |
| new_customer_interest | -0.010 |

> Warning: smallest slice has only 3 rows; differences here are not statistically meaningful.

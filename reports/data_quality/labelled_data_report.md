# Labelled data quality report

- Total rows: 200
- Dataset hash: `450745418479c703`
- Annotation version: `audit_v1_2025`
- Unique discussion groups: 23
- Rows without a recoverable group ID: 0
- Duplicate texts: 0
- Conflicting labels: 0
- Class imbalance ratio (max/min): 70.0

## Class counts

| Class | Rows |
|---|---|
| general_discussion | 140 |
| frustrated_demand | 25 |
| confusion_barrier | 18 |
| nostalgia_reactivation | 8 |
| high_intent | 7 |
| new_customer_interest | 2 |

## Actionable vs general

| Actionable | Rows |
|---|---|
| False | 140 |
| True | 60 |

## Source distribution

- youtube: 197
- reddit: 3

## Empty classes

These taxonomy classes have no labelled rows yet and cannot be learned or evaluated until the corpus is expanded:

- content_drought_fatigue
- expectation_decay

## Issues

- Taxonomy classes with zero labelled rows: ['content_drought_fatigue', 'expectation_decay']

---

Status: development corpus pending re-audit and expansion. Results derived from this dataset are preliminary.

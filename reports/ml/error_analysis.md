# Error analysis

> **Preliminary development benchmark.** These numbers come from the current 200-row audit corpus, which is still being re-audited and expanded. They are not final held-out performance and must not be quoted as production or CV headline metrics.

## Stage 1 / end-to-end error exports

| Export | Rows |
|---|---|
| false_positives.csv | 7 |
| false_negatives.csv | 2 |
| most_confident_errors.csv | 9 |
| multiclass_errors.csv | 12 |

## Rule vs model (end-to-end labels, held-out test rows)

- Agreement rate: 58.6%
- Both correct: 15
- Both wrong: 5
- Rule correct / model wrong: 7
- Model correct / rule wrong: 2

## Qualitative categories (human review template)

The categories below are provided for a human reviewer to tag error rows in the exported CSVs. They are not auto-assigned.

- [ ] sarcasm
- [ ] negation
- [ ] multi_intent
- [ ] lore_vs_behaviour
- [ ] purchase_of_old_title
- [ ] nostalgia_without_reactivation
- [ ] multilingual
- [ ] short_text
- [ ] ambiguous

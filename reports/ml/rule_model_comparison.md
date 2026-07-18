# Rule baseline vs supervised model

> **Preliminary development benchmark.** These numbers come from the current 200-row audit corpus, which is still being re-audited and expanded. They are not final held-out performance and must not be quoted as production or CV headline metrics.

- Test rows compared: 29
- Agreement rate: 58.6%
- Disagreement rate: 41.4%

| Outcome | Rows |
|---|---|
| Both correct | 15 |
| Both wrong | 5 |
| Rule correct, model wrong | 7 |
| Model correct, rule wrong | 2 |

## Disagreements by true class

| True class | Disagreements |
|---|---|
| general_discussion | 6 |
| frustrated_demand | 4 |
| high_intent | 1 |
| nostalgia_reactivation | 1 |

Full disagreement rows: `reports/ml/tables/rule_model_disagreements.csv`. These rows are the highest-value candidates for the re-audit queue.

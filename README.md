# Behavioural Intent Classification & Demand Intelligence

**Supervised NLP analysis of public Reddit and YouTube community discussion around Kingdom Hearts IV**

**Live dashboard:** https://kh4-demand-intel.streamlit.app

**Model status: Development / preliminary evaluation** — the labelled
audit corpus is still undergoing expansion and re-audit, so all
supervised-model results are preliminary development benchmarks, not
final held-out performance.

**Key finding:** sentiment is not behavioural intent. Positive comments
can still be low-action lore discussion, while negative or fatigued
comments can reveal demand risk, confusion barriers, or reactivation
opportunities.

**Unofficial portfolio project. Not affiliated with Square Enix, Disney,
or the Kingdom Hearts franchise. Data sourced from public community
discussion.**

---

## 1. Problem

Fan discussion volume and sentiment do not reveal *behaviour*: purchase
intent, reactivation, new-player interest, blocked demand, confusion, or
disengagement. This project classifies public community comments into an
eight-class behavioural intent taxonomy, estimates calibrated
actionable-signal probabilities, ranks comments for efficient human
review, and feeds the results into an explainable heuristic scoring
layer and a Streamlit dashboard.

## 2. Dataset

- ~4,831 processed Reddit + YouTube signals (local pipeline output).
- A **200-row manually corrected audit corpus**
  (`reports/audits/intent_audit_corrected.csv`) used as development
  labelled data. Real discussion-group IDs (YouTube video IDs, Reddit
  submission IDs) are recovered deterministically from permalinks —
  23 groups in total.
- Data-quality report: `python -m src.ml.data validate` →
  `reports/data_quality/labelled_data_report.{json,md}`.
- Full documentation: `docs/DATA_CARD.md`.

Two taxonomy classes (`content_drought_fatigue`, `expectation_decay`)
currently have zero labelled rows and cannot be learned or evaluated
until the corpus is expanded.

## 3. Taxonomy

| Intent | Meaning | Actionable |
|------|--------|---|
| high_intent | Explicit purchase intent | ✔ |
| nostalgia_reactivation | Legacy attachment driving re-engagement | ✔ |
| new_customer_interest | Signals from potential new players | ✔ |
| frustrated_demand | Demand blocked by lack of updates | ✔ |
| content_drought_fatigue | Coping signals during content drought | ✔ |
| confusion_barrier | Narrative complexity reducing accessibility | ✔ |
| expectation_decay | Disengagement after prolonged silence | ✔ |
| general_discussion | Non-actionable engagement | ✘ |

Annotation definitions, boundary cases, and priority rules:
`docs/ANNOTATION_GUIDE.md`.

## 4. ML architecture

```
Reddit / YouTube → Ingestion → Preprocessing → VADER Sentiment
        ↓
Behavioural Intent Modelling
    ├── Rule baseline (regex, interpretable)
    ├── TF-IDF Logistic Regression
    ├── TF-IDF Linear SVM
    ├── Word + character TF-IDF SVM
    ├── Optional sentence-embedding model  (.[embeddings])
    └── Optional transformer interface     (.[transformers])
        ↓
Calibrated probabilities (validation-only Platt scaling)
        ↓
Actionable-signal ranking (Precision@K, Recall@K, Lift@K)
        ↓
Heuristic demand / activation / risk scoring (unchanged)
        ↓
Evaluation + error analysis → Streamlit dashboard
```

The supervised layer (`src/ml/`) is fully separable from the heuristic
scoring layer (`src/score.py`), and the project distinguishes
classification, probability estimation, ranking, and decision-support
scoring. `demand_score` remains an explainable heuristic, not a
forecast.

## 5. Rule baseline

The ordered regex classifier (`src/intent.py`) is retained as the
interpretable baseline, exposed through the common classifier interface
(`src/ml/models.py`). Its fixed 0.7/0.3 outputs are surfaced as a
`rule_confidence_indicator` — never as probabilities. Regression tests
enforce precision floors on the audited corpus: `high_intent` 0.55,
`nostalgia_reactivation` 0.65, `new_customer_interest` 0.70
(`tests/test_intent_audit_regression.py`).

## 6. Supervised models

Benchmark ladder: majority class → rule baseline → TF-IDF Logistic
Regression → TF-IDF Linear SVM → combined word (1–2) + character (3–5)
TF-IDF SVM, with optional sentence-embedding and transformer models
behind extras. The main strategy is **hierarchical**: Stage 1 classifies
actionable vs general, Stage 2 assigns intent among actionable rows; a
flat multiclass benchmark runs alongside for comparison. Class imbalance
is handled with `class_weight="balanced"` and Stage 1 threshold
adjustment (no synthetic augmentation).

## 7. Evaluation methodology

- **Group holdout:** 70/15/15 splits with no discussion group or
  duplicate text crossing splits; automated leakage tests.
- **Validation-only tuning:** calibration and decision thresholds are
  fitted on validation and frozen before the test split is touched.
- **Repeated seeds:** each seed regenerates the split and model
  initialisation; per-seed and mean/std/median outputs are exported.
- **Metrics:** Stage 1 PR-AUC/F1 (prioritised over accuracy), Brier,
  ECE; Stage 2 and end-to-end macro-F1; ranking metrics.

Full rationale: `docs/EXPERIMENT_DESIGN.md`.

## 8. Preliminary results

> **Preliminary development benchmark** — 200 rows, 23 groups, 5 seeds.
> Generated tables: `reports/ml/results_summary.csv`,
> `reports/ml/model_comparison.md`. Numbers are not hardcoded here
> because they regenerate with every benchmark run; on this tiny corpus
> the rule baseline still leads end-to-end macro-F1 while SVM-family
> models lead Stage 1 actionable ranking. Seed variance is large — the
> corpus must be expanded before any number is treated as meaningful.

## 9. Ranking / decision-support metrics

For actionable-signal review efficiency the benchmark reports
Precision@K, Recall@K, and Lift@K (K = 10/25/50/100 and 5/10/20%), plus
cumulative-gains and lift curves (`reports/ml/figures/`). Lift@10%
answers: how many more actionable signals does a reviewer find in the
top 10% model-ranked comments than by random review? Only measured lift
is reported.

## 10. Error analysis

`python -m src.ml.error_analysis` exports false positives, false
negatives, most-confident errors, multiclass errors, and rule-vs-model
disagreements (`reports/ml/tables/`), with Markdown summaries
(`reports/ml/error_analysis.md`, `reports/ml/rule_model_comparison.md`).
Qualitative error categories are a human-review template, never
auto-assigned. An active-learning queue
(`python -m src.ml.active_learning`) prioritises the most informative
rows for the next annotation pass.

## 11. Dashboard

```bash
pip install -e .[ml,app]
streamlit run app.py
```

Tabs: **Demand Intelligence** (original dashboard, unchanged), **Model
Performance**, **Model Comparison**, **Error Analysis**, **Drift**, and
**Live Inference** (development-model disclaimer included). The app
shows the current model status and degrades gracefully when report files
are absent.

## 12. Reproducibility

```bash
pip install -e .[ml,dev]

python -m src.ml.data validate                 # data-quality report
python -m src.ml.benchmark \
  --models majority rules logistic linear_svm char_word_svm \
  --seeds 42 43 44 45 46                       # full benchmark
python -m src.ml.train --model logistic        # persist artefacts
python -m src.ml.error_analysis                # error reports
python -m src.ml.drift                         # distribution-shift report
python -m src.ml.active_learning               # re-audit queue
pytest                                         # full test suite
```

Every run records run ID, git commit, dataset hash, annotation version,
seeds, split/group counts, frozen threshold, and
`evaluation_status = development_preliminary`
(`reports/ml/run_metadata.json`).

## 13. Limitations

- 200 labelled rows across 23 discussion groups: split variance
  dominates; all supervised results are preliminary.
- Two taxonomy classes have no labelled examples.
- 197/200 labelled rows are YouTube; Reddit performance is unmeasured.
- Labels were sampled around rule-classifier behaviour (selection bias).
- Temporal dynamics are unmodelled (timestamps not yet recovered).
- `demand_score` is a heuristic decision-support signal, not a forecast.
- See `docs/MODEL_CARD.md` and `docs/DATA_CARD.md`.

## 14. Roadmap

1. Re-audit existing labels via `reports/annotation/review_queue.csv`
   using `docs/ANNOTATION_GUIDE.md`.
2. Expand the labelled corpus using the active-learning queue.
3. Recover `created_at` timestamps to enable temporal holdout.
4. Freeze a final test set; re-run the benchmark with
   `evaluation_status = final_held_out`.
5. Populate the gated CV metrics in `reports/portfolio_summary.md`.
6. Evaluate the optional embedding/transformer models on the expanded
   corpus.

---

**Author:** Ghobikan Aravindan

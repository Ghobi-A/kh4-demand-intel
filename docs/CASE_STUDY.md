# Case study: how far can 200 labels take a behavioural demand-intelligence system?

**Question.** Can public Kingdom Hearts IV discussion be turned into a reliable signal of
*behavioural intent* (purchase intent, frustration, confusion and similar), and is a
learned model measurably better than a hand-written rule baseline at finding it?

**Short answer.** Not yet, and the evaluation design is what shows it. Under leakage-safe
group cross-validation, every supervised model beats random review (PR-AUC 0.49–0.65
against a 0.30 base rate). **None can be separated from the regex rule baseline**, and
the labelled sample is stratified on that baseline's own predictions, which inflates its
apparent accuracy. The binding constraint is the labels, not the model family. The
annotation protocol in [`ANNOTATION_PROTOCOL.md`](ANNOTATION_PROTOCOL.md) is the next
step.

Everything below is labelled **Executed** (produced in this repository, with the command
given) or **Proposed** (designed, not yet run). No label, sales figure or result is
invented, and discussion volume is never treated as sales.

---

## 1. Data

| | Status | Detail |
|---|---|---|
| Public comments (YouTube, Reddit) | Observed; full set **not in repo** | ~4,831 processed rows when last built locally; regenerating needs a YouTube API key |
| Human-labelled corpus | Observed, committed | 200 rows, 23 discussion groups, 1 annotator, 197 YouTube / 3 Reddit |
| Weekly demand proxy | Derived | Sum of calibrated P(actionable) per week. A discussion proxy, **not** sales |
| Forecasts | Model output | On a 304-row demo extract only (`evaluation_status = demo_only`) |
| Marketing mix data | **Synthetic** | Simulated with known ground truth; not Square Enix data |

**Executed:** `python -m src.ml.dataset_audit` → `reports/data_quality/dataset_audit.md`.
Two classes have zero labels and three have fewer than 10. First-pass vs final labels from
the single annotator agree on 57% of rows (κ 0.30).

### The sampling-design problem (found in this review)

`scripts/sample_intent_audit.py` drew the labelled corpus as 100 comments the rule
classifier called `general_discussion`, plus 20 comments for each class the rule
predicted. The inclusion probabilities were never recorded. Consequences, all
**executed** and measured:

- The rule baseline's class recall is near 1 by construction. 17 of the 18
  `confusion_barrier` rows entered the sample *because the rule fired*, so its
  class-wise F1 (0.94) is an artefact.
- Actionable comments are over-represented. The sample base rate is 0.30; a
  design-weighted estimate, using population rule-label shares from
  `reports/figures/intent_distribution.png`, puts it at **0.15**.

The annotation queue now records every row's stratum and inclusion probability, so the
next batch does not repeat this.

## 2. Evaluation protocol (executed)

`python -m src.ml.model_comparison --with-embedding` →
[`reports/ml/evaluation_report.md`](../reports/ml/evaluation_report.md)

- 5-fold **group** cross-validation, so whole videos and threads are held out, repeated
  over 3 fold seeds. Every model sees identical folds.
- Calibration (Platt) and the decision threshold are fitted on a group-held-out
  validation split *inside* each training fold. TF-IDF and model weights are fitted per
  fold.
- Uncertainty comes from 1,000 **paired** group-bootstrap replicates: each resample of
  groups scores every model, so differences have their own intervals. A model is called
  better only when that interval excludes zero.
- Ranking metrics use the binary actionable target, which is what a review queue needs.
  Ties are broken in expectation. Intent-level ranking is not reported, because eight
  classes have no single relevance order.

Models: majority; regex rules (Platt-calibrated two-level score); TF-IDF logistic
regression; TF-IDF linear SVM; word+char TF-IDF SVM; and a **compact transformer
baseline**, frozen `all-MiniLM-L6-v2` sentence embeddings (22M parameters, pinned
revision) with a class-balanced logistic head.

## 3. Results (executed)

| Model | PR-AUC [95%] | Lift@20 | Brier | Design-weighted PR-AUC [95%] |
|---|---|---|---|---|
| majority | 0.31 [0.23–0.37] | 1.10 | 0.212 | 0.16 [0.10–0.23] |
| rules | 0.62 [0.52–0.72] | 2.27 | 0.145 | 0.31 [0.20–0.49] |
| logistic | 0.57 [0.47–0.67] | 2.11 | 0.176 | 0.37 [0.23–0.54] |
| linear_svm | 0.59 [0.49–0.69] | 2.28 | 0.172 | 0.37 [0.23–0.53] |
| char_word_svm | 0.65 [0.56–0.75] | 2.50 | 0.157 | 0.43 [0.27–0.61] |
| embedding (frozen MiniLM) | 0.49 [0.42–0.58] | 2.00 | 0.206 | 0.32 [0.22–0.49] |

What the paired tests **support**:

- The rule baseline beats the majority baseline (the only pairing tested against
  majority). Every model's PR-AUC interval lies above the base rate, both on the sample
  (0.30) and design-weighted (0.15).
- word+char SVM > logistic, and word+char SVM > word-only SVM, on PR-AUC (both on the
  sample and design-weighted) and on Brier.
- Both TF-IDF SVMs > the frozen-encoder baseline on PR-AUC.

What they **do not support**:

- Any supervised model being better than the rule baseline. On the sample the rules
  rank second; design-weighted, they rank fifth. The intervals overlap in both cases,
  and the closest call (word+char SVM vs rules, design-weighted) has a 95% interval of
  −0.00 to +0.25.
- The frozen-encoder result as a verdict on transformers. Its head uses the same
  untuned C = 1 as the TF-IDF model on normalised 384-dimensional inputs, so it may be
  under-fitted.

**Errors** (`reports/ml/model_comparison/errors.csv`, every error with text):

- Length is a confounder. The word+char SVM's false positives have a median of 28 words
  against 16 for false negatives, and its false-positive rate is 52% on comments over 25
  words against 9% on comments of 8 words or fewer (`reports/ml/group_cv_report.md`).
- The most confident false positives are long, engaged general discussion. The most
  confident false negatives are terse or sarcastic frustration ("See you in another 4
  years.").
- All four supervised models make the same mistake on 25 rows. Some of those look like
  possible label errors (for example, a lore remark labelled `high_intent`). They are
  candidates for adjudication, not relabelling.

**Fixed while measuring:** ranking metrics broke ties by row order, and this corpus is
ordered by sampling stratum. The majority baseline's committed Lift@10% was therefore
0.16 instead of its definitional 1.0. Ties are now broken in expectation; only the
rules and majority numbers moved.

## 4. Forecasting and monitoring (executed, demo data only)

On the 304-row demo extract, the naive forecast has the lowest matched-origin WAPE
(0.82 vs 0.93 Bayesian). That comparison rests on 6 forecasts, and one origin produces
the whole gap (`reports/demo_forecasting/matched_origin_sensitivity.json`). No model is
recommended. Monitoring flags persistent bias, and interval coverage is reported per
model. A full-data run is **blocked** on the missing canonical data
(`scripts/check_full_data_readiness.py` exits 2 and lists the inputs).

## 5. Operational paths (verified)

| Requirement | Status before this review | Now |
|---|---|---|
| Ingestion with provenance | Existed (fetch manifest, UTC timestamps) | Unchanged |
| Preprocessing | Existed (empty, deleted and duplicate-ID removal; URL strip) | Gap noted: no text-level dedupe at ingestion |
| Calibrated probabilities | Existed (fixed in the previous PR) | Unchanged |
| Group-aware evaluation | Existed (holdout + group CV) | **Added** paired bootstrap, design weighting, review-budget ranking, compact-transformer baseline, inspectable errors |
| Annotation workflow | Queue + validator existed | **Added** group cap, inclusion probabilities, double annotation, blind adjudication, protocol |
| Batch inference | **Missing** (pipeline labelled with rules only) | **Added** `src.ml.batch_inference` with provenance and monitoring (coverage, PSI, intent mix). On the demo extract with OOF off, the monitor correctly flags 47% coverage (labelled rows excluded, never scored in-sample; this fixed a leak where `--no-oof` scored them in-sample) and a PSI of 1.15, because every remaining row is rule-predicted actionable |
| Forecast monitoring | Existed (bias, coverage, reliability) | Unchanged |
| Reproducibility | Run metadata and hashes existed | Verified: benchmark, group CV and model comparison regenerate byte-identically |
| CI | Existed | **Added** model-comparison, batch-inference and adjudication tests and smoke steps |
| Dashboard | Overview-first; deploy-ready | **Not publicly reachable**: anonymous requests are redirected to Streamlit sign-in (owner setting) |

## 6. Proposed experiments (not run)

1. **Frozen evaluation tier**: about 300 comments from at least 40 unseen groups, two-stage
   group sample, double-annotated and adjudicated (protocol §Two tiers). This is the only
   route to a `final_held_out` number.
2. **Design-weighted final evaluation** using the recorded inclusion probabilities
   instead of chart-read shares.
3. **Tuned and fine-tuned transformer baselines**: a nested C search for the MiniLM head;
   fine-tuning DistilBERT or MiniLM with `src/ml/transformer.py` once at least 1,000
   labels exist. On 140 training rows per fold, fine-tuning would mostly measure
   variance.
4. **Length-controlled evaluation**: stratify the next batch by length bucket, and report
   metrics per bucket, to separate "long comment" from "actionable".
5. **Full-data forecasting** with the sensitivity analysis, once the canonical signals
   are rebuilt.

## 7. Resources

| Item | Requirement | Measured / estimated |
|---|---|---|
| Classical models + group CV + paired bootstrap | 4-CPU container | **Measured**: ~30 s classical, ~80 s with the encoder |
| Compact transformer baseline | CPU torch (≈0.8 GB installed) + MiniLM weights (88 MB download) | **Measured**: 1.7 s to encode 200 comments; no GPU needed |
| Fine-tuned transformer (proposed) | CPU feasible at this size; GPU advisable above ~5k labels | Estimate |
| Bayesian forecast (demo) and MMM | PyMC, 4 chains | **Measured**: a few minutes each on 4 CPUs |
| Full-data forecasting | Canonical signal set + YouTube API key | **Unavailable** in this environment |
| Annotation round (300 items, double) | 2 annotators + 1 adjudicator | Estimate: ~2.5 h per annotator + ~1 h adjudication + 1 h calibration |

## Reproduce

```bash
pip install -e ".[ml,app,dev,forecast,api]"
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[embeddings]"

python -m src.ml.dataset_audit
python -m src.ml.model_comparison --with-embedding   # reports/ml/evaluation_report.md
python -m src.ml.batch_inference --input data/demo/signals_scored_portfolio.csv \
    --output reports/inference/portfolio_scored.csv
python -m src.ml.annotation_queue build
python scripts/check_full_data_readiness.py          # exits 2: canonical data absent
pytest -m "not pymc_smoke"
```

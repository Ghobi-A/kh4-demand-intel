# Annotation expansion and independent adjudication protocol

Status: **proposed protocol with implemented tooling.** No batch has been run under it
yet, and no label in this repository was produced by it. Label definitions stay in
[`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md); this document defines *which* comments
get labelled, *by whom*, and *how disagreements become gold labels*.

## Why the current corpus cannot support a final evaluation

Measured in `reports/data_quality/dataset_audit.md` and `reports/ml/evaluation_report.md`:

| Problem | Evidence | Protocol response |
|---|---|---|
| The sample is stratified on the rule classifier's own output (100 rule-general + 20 per rule class), and the design weights were never recorded | `scripts/sample_intent_audit.py`; rule class-wise F1 is inflated by construction | Every queued row records its stratum and inclusion probability; a uniform random stratum is always included |
| One annotator; first-pass vs final κ = 0.30 | `dataset_audit.md` | Two independent annotators on every evaluation item; a third person adjudicates |
| Two classes have zero rows; three have fewer than 10 | `labelled_data_report.md` | Rare-class candidate stratum with quotas; stop rules per class (below) |
| 197/200 rows are YouTube; 23 groups; top 3 groups hold 34% of rows | `dataset_audit.md` | Group-level sampling with a per-group cap and a Reddit quota |
| No frozen test set; every model choice has seen every row | — | Evaluation tier sampled by group and frozen before any further model change |

## Two tiers

| Tier | Purpose | Sampling | Annotation | Size (first round) |
|---|---|---|---|---|
| **Evaluation (frozen test)** | Unbiased final performance | Two-stage group sample (below); groups never used for training | 100% double, blind, adjudicated | ≈300 comments from ≥40 groups |
| **Training** | Grow rare classes and new platforms | Stratified queue (`annotation_queue build`) | Single annotator + 15% overlap for drift in agreement | ≈300 comments per round |

The evaluation tier is drawn **first** and its groups are excluded from every later
training batch (`annotation_queue build` already excludes rows and near-duplicates of
labelled rows; pass the frozen-test file through the same exclusion).

## Sampling frame

The unit of sampling is the **discussion group** (a YouTube video or a Reddit
submission), not the comment, because comments in one group share topic, vocabulary
and authors. Splits are by group, so sampling should be too.

**Evaluation tier: two-stage sample.**

1. *Stage 1, groups.* Stratify groups by `source` (YouTube / Reddit) and by period
   relative to the event calendar (before the first announcement, within 4 weeks after an
   event, otherwise). Sample groups within each stratum with probability proportional to
   comment count, capped so that no stratum supplies more than 40% of groups.
2. *Stage 2, comments.* Within each sampled group, draw up to 8 comments uniformly at
   random.
3. Record both stages' inclusion probabilities, so that population metrics can be
   estimated with Horvitz–Thompson weights. This is the step the original sample lacked.

**Training tier: stratified queue.** `python -m src.ml.annotation_queue build` draws
quotas in this order (all recorded per row with `inclusion_probability`):

| Stratum | Default quota | Why |
|---|---|---|
| `random_control` | 25 | Drawn first, uniformly: unbiased prevalence and an error audit of the other strata |
| `missing_class_candidate` | 40 | Rule predicts a class with fewer than 10 labels (including the two empty ones) |
| `reddit` | 30 | Platform currently 3/200 |
| `unseen_group` | 25 | Groups absent from the labelled corpus |
| `model_disagreement` | 30 | Rule/model disagreement or near-threshold probability |

Constraints: at most 5 comments per group per batch (`--max-per-group`), and rows that
duplicate or near-duplicate (TF-IDF cosine ≥ 0.80) a labelled row are excluded. A
stratum the pool cannot fill is reported as a shortfall and is never back-filled.

**Stop rule per class.** Keep adding `missing_class_candidate` quota until every class
has at least 30 adjudicated rows or two consecutive rounds add fewer than 3 positives
for it. In that case, record the class as empirically rare and consider merging it,
rather than continuing to search.

## Annotation procedure

1. **Calibration round.** Both annotators independently label the same 50 items drawn
   from the random-control stratum, then meet to discuss every disagreement. Guide
   changes are made here, in a versioned commit to `ANNOTATION_GUIDE.md`, and *before*
   production labelling. These 50 items are not used for evaluation.
2. **Blind production labelling.** Annotators see only `annotation_queue.csv`: text,
   source and group. They never see rule labels, model labels, model probabilities or
   each other's labels. Those live in `annotation_queue_provenance.csv`.
3. **Fields.** `human_label` (primary, per the guide's priority rules), optional
   `secondary_label` and `multi_intent_flag`, `annotator_id`, `annotated_at`, and free-text
   `review_notes`. "Not about KH4 / spam / non-English" goes in `review_notes` with
   `general_discussion` as the primary label, so it can be filtered later.
4. **QC.** `python -m src.ml.annotation_queue validate --completed <file>` rejects
   labels outside the taxonomy, missing annotator IDs, duplicates, and rows already in the
   corpus.

## Disagreement analysis and adjudication

```bash
python -m src.ml.adjudication agreement --a annotator_a.csv --b annotator_b.csv
# third person fills adjudicated_label, adjudicator_id, rationale in:
#   reports/annotation/adjudication/adjudication_queue.csv
python -m src.ml.adjudication finalise --a annotator_a.csv --b annotator_b.csv \
    --adjudicated reports/annotation/adjudication/adjudication_queue.csv
```

`agreement` reports, with item-bootstrap 95% intervals:

- 8-class raw agreement and Cohen's κ;
- binary actionable κ and PABAK (prevalence-adjusted, because actionable is a minority);
- per-class one-vs-rest κ, flagged for guide review below **0.60**;
- the full confusion matrix and the most frequent disagreement pairs. These are the
  input for the next guide revision; for example, `frustrated_demand` vs
  `general_discussion` was the commonest first-pass revision in the existing corpus.

Adjudication rules:

- The adjudicator is a third person. `finalise` refuses an `adjudicator_id` equal to
  either annotator.
- The adjudication queue shows the two candidate labels in random order without
  annotator identities, so the adjudicator cannot defer to a person.
- The adjudicator may choose a third label; every adjudication needs a written
  `rationale`.
- Output rows carry `label_source` (`agreed` or `adjudicated`), both original labels,
  all three IDs and a new `annotation_version`. Status is `gold_candidate_unmerged`.

Acceptance targets before the evaluation tier is used: binary actionable κ ≥ 0.70, and
no class with κ < 0.60 that has not been through a guide revision.

## What is never done

- No label is generated by a rule, a model, or an LLM and treated as ground truth. Model
  outputs may *choose which rows to annotate*; they never fill `human_label`.
- Existing labels in `reports/audits/intent_audit_corrected.csv` are not overwritten.
  Merging gold candidates is a separate, reviewed commit with a new annotation version.
- Annotators are not shown model suggestions, and the test tier is never used to tune
  anything.

## Resources

- Time: about 20–30 s per comment for a trained annotator. A 300-item double-annotated
  evaluation tier is about 2.5 h per annotator, plus about 1 h of adjudication, plus a
  1 h calibration meeting.
- People: two annotators familiar with the franchise and the guide, and one adjudicator.
- Data: the canonical scored signal set (`data/processed/signals_scored.csv`), which is
  not in the repository. Without it, the queue can only be drawn from the 304-row demo
  extract, and most strata cannot be filled (see
  `reports/annotation/annotation_queue_manifest.json`).
- Compute: negligible (seconds on a laptop CPU).

## After a round

1. `python -m src.ml.dataset_audit`: support by class, source and group.
2. Merge accepted gold candidates in a reviewed commit with a new `annotation_version`.
3. `python -m src.ml.model_comparison --with-embedding` on the training tier, then a
   **single** scoring run on the frozen evaluation tier with `evaluation_status =
   final_held_out`, design-weighted with the recorded inclusion probabilities.

# Data card

## Sources

Public community discussion about Kingdom Hearts IV:

- **YouTube** comments on official/franchise-related videos (comment
  permalinks embed the video ID, used as the discussion-group ID).
- **Reddit** posts and comments from franchise-related subreddits
  (permalinks embed the submission ID).

All content was publicly posted. This is an unofficial portfolio
project, not affiliated with Square Enix or Disney.

## Volumes

- Full processed signal set: ~4,831 rows (local; not required in the repo).
- Manually reviewed labelled corpus: **200 rows**
  (`reports/audits/intent_audit_corrected.csv`), sampled half for
  precision slices and half for recall slices of the rule classifier.

## Composition and imbalance

Current labelled class counts (see
`reports/data_quality/labelled_data_report.md` for the generated
version): general_discussion 140, frustrated_demand 25,
confusion_barrier 18, nostalgia_reactivation 8, high_intent 7,
new_customer_interest 2; `content_drought_fatigue` and
`expectation_decay` have **zero** labelled rows. Actionable vs general
is 60/140. Max/min imbalance ratio is 70:1.

## Known biases

- **Platform bias:** 197 of 200 labelled rows are YouTube; Reddit is
  barely represented in the labelled set.
- **Selection bias:** rows were sampled around rule-classifier
  behaviour (precision/recall slices), so the labelled distribution is
  not a random sample of the full signal set.
- **Topic/creator bias:** only 23 discussion groups; a handful of videos
  dominate.
- **Language bias:** predominantly English; rules and models are not
  evaluated for other languages.

## Timestamps

Each signal carries the creation time supplied by the source API (YouTube
`publishedAt`, Reddit `created_utc`), parsed to timezone-aware UTC in one place
(`src/timestamps.py`). Scrape time is recorded separately and is never
substituted for creation time.

A missing timestamp stays missing. Two earlier defects are fixed: a Reddit
comment with no `created_utc` used to default to 0 and become a real-looking
1970 timestamp, and a YouTube comment with no `publishedAt` used to be dropped
entirely, which biased volume counts downwards. Rows without a usable timestamp
are excluded from weekly aggregation and counted in the output metadata rather
than being placed in an arbitrary bucket.

An optional, network-dependent enrichment path
(`python -m src.enrich_timestamps`) re-fetches creation times for rows collected
before this handling existed. It is never part of the normal pipeline or CI, and
identifiers it cannot resolve are reported as unresolved rather than filled in.

## Derived demand proxies

The weekly table (`data/processed/weekly_demand_signals.csv`) aggregates scored
signals into complete weekly periods. Its headline column,
`actionable_probability_mass`, is the sum of calibrated P(actionable) over the
comments observed in a period: an expected actionable-signal volume, **not**
sales, revenue or units.

Each row records `probability_source`:

| Value | Meaning |
|---|---|
| `oof_supervised` | Out-of-fold, from a model trained without this row or its discussion group |
| `persisted_supervised` | From the persisted trained model, for rows outside the labelled corpus |
| `excluded_training_row` | In the corpus but no out-of-fold value; excluded from the proxy |
| `unavailable` | No usable probability; contributes nothing |

Rows are matched to the labelled corpus by record id, falling back to normalised
text because the portfolio extract strips identifiers for privacy. The rule
baseline's fixed 0.7/0.3 confidence indicator is never used as a probability.

The committed weekly table under `data/demo/` is built from the class-stratified
304-row portfolio extract. That extract samples up to 60 rows per intent class,
so its weekly volumes are **not volume-representative** and it is stamped
`sample_only = true`, `volume_representative = false`,
`evaluation_status = demo_only`.

## Synthetic marketing data

`data/synthetic/` contains simulated marketing data with known ground-truth
parameters and is **not Square Enix data**. It never mixes with the real signals
above: see `docs/MMM_LAB.md`.

## Privacy

The portfolio extract strips user-level metadata; usernames are not
included in labelled data, reports, or the dashboard. Permalinks are
retained only to preserve verifiable provenance and group structure of
public comments.

## Limitations

200 labelled rows across 23 groups is far too small for stable
supervised estimates; two taxonomy classes cannot be learned or
evaluated at all. All model results are development-preliminary until
the re-audit and expansion (see `docs/EXPERIMENT_DESIGN.md`).

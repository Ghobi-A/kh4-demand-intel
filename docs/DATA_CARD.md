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

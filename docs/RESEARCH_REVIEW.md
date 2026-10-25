# Research review of PR #26

Review basis: `main` at `15598b4a16a30ad0dba6050421b8adae265294f2` and PR #26 at
`c80400d75634a5b103176f433bd810936c33990b`, inspected through Git. This follow-up
branch includes the PR's existing work; it does not merge PR #26 into main.
GitHub API access is blocked by the environment network policy, so PR comments,
review threads, checks and formal approval state could not be inspected. This is
a code-and-evidence review, not a GitHub approval.

## Review decision

**Corrections required before merge.** PR #26 substantially improves the project,
but its queue inclusion-probability claim is invalid, and its original folds have
a cross-discussion near-duplicate. The fixes and sensitivity below address those
issues without manufacturing labels, changing the inference model or increasing
model complexity. Results remain `development_preliminary`.

| Area | Already implemented in main / PR #26 | Follow-up review |
|---|---|---|
| Discussion isolation | main: whole-group outer and inner splits | Checked 15 configurations: no shared discussion IDs or exact/normalized duplicate texts |
| Near-duplicate leakage | Not excluded across different discussions | Found one cross-video pair at char-ngram cosine 0.8499; added connected-group sensitivity, linking before both outer and inner splitting |
| Calibration | main: validation-only Platt scaling and threshold | No held-out label use found; small validation groups remain a limitation; report labels ECE as reference-seed rather than seed-averaged |
| Ranking | PR: expected tie handling, fixed review budgets | Retained; no complex model justified |
| Paired uncertainty | PR: identical folds and paired group bootstrap | Reject missing/duplicate/inconsistent predictions instead of silently averaging/dropping them; identify pointwise intervals as exploratory without multiplicity correction |
| Population weighting | PR: chart-read stratum shares | Sensitivity assumptions only; neither verified inclusion probabilities nor corrected population performance |
| Annotation acquisition | PR: caps, rare-class/Reddit/disagreement strata | Record allocation fractions; leave unknown inclusion probabilities blank; add explicit frozen-group and near-text exclusion |
| Human adjudication | PR: independent IDs, blind label order | Reject blank identities, disjoint queues, mismatched text, duplicate adjudications and attempts to override agreed items |
| Dashboard | main overview; PR paired report and inference banner | Added protocol selector, paired interval chart, budget explorer, error filters, marketing decision paths and deployment evidence gates |
| Full forecasting | main leakage tests, matched origins, bias checks, demo refusal | Canonical input still absent; no full-data forecast claimed |

Forecast leakage has an additional boundary: the existing tests condition on a
fixed weekly proxy. Its NLP measurement model is not frozen/refit as of each
origin. Group OOF does not enforce temporal availability of training labels or
calibration. This may introduce future information into a historical proxy;
end-to-end prospective validity is unverified. Freeze the measurement model
before a future window, or rebuild it as of each origin using versioned snapshots
and annotation times. This cannot be repaired by relabeling a demo backtest.

## Empirical findings and scope

The original six-model predictions contain 200 human-labelled rows from 23
public discussion groups, 197 YouTube / 3 Reddit, with one annotator. Two intent
classes have no labels. The labels, rules and model families have already been
used during development; cross-validation cannot turn them into a final test.

- Classical refit on the original folds reproduced PR #26's PR-AUC, Brier and
  Lift@20 exactly for all five classical baselines.
- Replaying saved predictions for all six models reproduced `point_estimates.csv`,
  `paired_differences.csv` and `errors.csv` byte for byte. This validates those
  calculations, **not** an encoder refit. MiniLM downloads are blocked here.
- Original-fold word+char SVM PR-AUC is 0.653, versus rules 0.616 and frozen
  MiniLM 0.493. Its paired PR-AUC difference from rules includes zero. This is
  selected development evidence, not an established population winner.
- One near-duplicate pair connects two videos. The separate
  [`leakage_sensitivity/report.md`](../reports/ml/leakage_sensitivity/report.md)
  refits classical models after linking these discussions (23 actual groups,
  22 splitting/bootstrap units). Do not combine its rows with original-fold
  embedding predictions or compare unmatched intervals as if paired.
- Linked-group point PR-AUC: rules 0.634; word+char SVM 0.598; linear SVM
  0.537; logistic 0.515; majority 0.283. The word+char-minus-rules paired
  95% interval is −0.157 to +0.088. Both group assignment and possible
  repeated-content exposure change, so the point shift cannot be attributed
  solely to leakage. It is a robustness check, not a new final test.
- The bootstrap conditions on fitted OOF predictions and averages fold seeds.
  It does not resample and refit the full training procedure. Forty-four
  pointwise comparisons in the original report are exploratory, without
  family-wise or false-discovery control. No deployment model is switched.
- Rule-stratified sampling changes measured class performance; it does not
  mathematically force recall to one. Chart-read reweighting does not establish
  representative prevalence. Earlier stronger claims are corrected.

## Data recovery and human review

Searched all 26 remote branches, their reachable tracked-data history and this
checkout. No `data/processed/signals_scored.csv`, canonical clean input or raw
comment collection was found. Tracked data consists of demo extracts, video
seeds, reference events and synthetic marketing data. Other repositories in the
owner's account cannot be searched until GitHub API access is available; this
search does not establish that no other repository contains the data.

The new queue at `reports/annotation/research_review/` is explicitly a **demo-pool
training acquisition queue**. It has no human labels. The small pool cannot fill
rare-class, Reddit or unseen-group quotas; its manifest reports the shortfalls.
Do not use it as a representative test set. Earlier queues under
`reports/annotation/` are historical templates; use the new manifest's corrected
selection semantics. A frozen two-stage evaluation sampler and final
population-weighted scorer remain proposed work, not implemented capabilities.

With the canonical pool and an independently frozen test file:

```bash
python -m src.ml.annotation_queue build \
  --pool data/processed/signals_scored.csv \
  --exclude-frozen /secure/path/frozen_test.csv \
  --output-dir /secure/path/annotation-round-1
```

The frozen CSV must have `text`, `source`, `parent_id`. The tool excludes entire
frozen groups, unknown candidate groups and near-duplicate text. Have two actual
humans label the overlap independently, then use `src.ml.adjudication`. Tooling
cannot verify that two IDs represent two people. Never fill labels with model or
LLM output and call that independent human annotation.

To rebuild absent data, provide `YOUTUBE_API_KEY` securely in environment settings,
then use the seed-list ingestion and pipeline commands in
`scripts/check_full_data_readiness.py`. Re-fetching today's comments does not
recover the exact original historical collection. Preserve the new fetch manifest
and timestamps; check scope and volume representativeness before forecasting.

## Public demo and delivery boundaries

No app merge, deploy or visibility setting has been changed. The current public
URL check is blocked by network policy. PR #26's prior report of a sign-in redirect
has not been independently rechecked here. There is no application login flow in
`app.py`; a platform visibility restriction must be changed by the owner in
Streamlit Community Cloud, after approval: set the app visibility to Public, then
verify the URL in a signed-out/incognito session, including interactions. Do not
claim anonymous accessibility based on a local dashboard test.

GitHub API, encoder downloads and the public demo need `api.github.com`,
`huggingface.co`, and `kh4-demand-intel.streamlit.app`; these additions were saved
in the environment draft. A successful draft save does not apply or publish it.
Encoder weight downloads may identify additional exact CDN domains after access
is enabled. No credentials were requested or fabricated.

## Reproduction and validation

Run from the repository root in Python 3.12. The environment uses `.venv` and
writable PyTensor/Numba/Matplotlib caches under `/workspace/.cache/kh4`; see
[`validation/development-versions.txt`](validation/development-versions.txt) for
exact installed package versions. Installation and startup instructions are saved
in the environment draft, left unpublished. Outputs should use fresh directories.

```bash
pip install -e '.[ml,app,dev,forecast,api]'
ruff check src/ tests/ scripts/ app.py
pytest -q -rs -m 'not pymc_smoke'

# Verify the original six-model analysis without pretending to refit the encoder.
python -m src.ml.model_comparison \
  --models majority rules logistic linear_svm char_word_svm embedding \
  --oof reports/ml/model_comparison/group_cv/oof_predictions.csv \
  --output-dir /tmp/kh4-replay --report /tmp/kh4-replay/report.md

# Refit the classical comparison with near-duplicate-connected groups held together.
python -m src.ml.model_comparison --link-near-duplicates \
  --output-dir /tmp/kh4-linked --report /tmp/kh4-linked/report.md

# Once model-download access is available, refit the frozen encoder on linked folds.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[embeddings]'
python -m src.ml.model_comparison --with-embedding --link-near-duplicates \
  --output-dir /tmp/kh4-linked-all --report /tmp/kh4-linked-all/report.md

python scripts/check_full_data_readiness.py --output-dir /tmp/kh4-readiness
# Expected exit 2 until canonical data and provenance are available.
```

Current validation: 377 regression tests passed, 23 `pymc_smoke` tests deselected;
two representative Bayesian integration tests also passed in this environment.
Ruff, dependency consistency, package imports, original-fold classical refits,
six-model saved-prediction replay, linked-group refit and the CI refit/replay
byte-comparison smoke check passed. Full posterior production experiments were
not rerun. Dataset readiness exits 2 as expected for missing canonical input.
The separate clean app environment installs only `requirements.txt` plus pytest,
with no full dataset, trained artefacts, PyMC or torch. Its AppTest exercises all
11 tabs, trained live inference and comparison/error controls. It validates the
local application, not Streamlit Cloud visibility or publication.

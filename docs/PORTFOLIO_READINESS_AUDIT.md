# Portfolio readiness audit

Date: 2026-10-17 · Baseline audited: `main` at `ca7fca9` (after the merged forecasting/MMM work)
· Branch with fixes: `claude/kh4-demand-intel-audit-wkjyal`

This audit is evidence-first. Every finding below is marked **Verified** (reproduced
in this checkout, with the command or file that shows it) or **Suspected** (consistent
with the evidence but not reproducible here). No data, labels or evaluation results
were invented. Where data is missing, the work is listed as blocked.

## Summary

The engineering foundation is strong: group-aware splits with leakage checks,
out-of-fold probability provenance, rolling-origin backtests with announcement-date
gating, explicit demo stamping, and an honest synthetic MMM. The committed ML
benchmark reproduces exactly (max absolute difference 0.0 across all per-seed
metrics), and the demo forecast reproduces to about 1e-11.

The problems were in what the evidence was taken to show, plus four code defects:

1. **The headline "naive beats Bayesian" result is anecdotal.** It rests on 6 forecasts
   over 3 origins of the demo sample, and one origin (208) produces the whole gap.
   Leave that origin out and the two are tied (WAPE 0.961 vs 0.959).
2. **The logistic model's "calibrated" probabilities were not calibrated.** Out of fold,
   their mean was 0.46 against a 0.30 base rate, with ECE 0.21. Those probabilities
   are summed into the forecasting target.
3. **Commenter usernames were published** in the portfolio extract and shown on the dashboard.
4. **The synthetic MMM scenario was not a reallocation.** It added about 267k of spend. As a
   budget-neutral move, its effect is indistinguishable from zero.
5. **A demo extract could be stamped `full_dataset`** if `--demo` was omitted.

All five are fixed on this branch, with regression tests.

## Severity-ranked findings

| ID | Sev | Status | Finding | Evidence | Fix on this branch |
|---|---|---|---|---|---|
| F1 | P0 | Verified, fixed | **Usernames in a public artefact.** `data/demo/signals_scored_portfolio.csv` had YouTube `author_display_name` handles nested in the JSON `metadata` column (296 of 304 rows). The dashboard's "Top 25" tables showed that column. This contradicts `docs/DATA_CARD.md` ("usernames are not included"). The cause: `_strip_identifying_columns` dropped top-level columns only. | `git grep -c author_display_name` on `ca7fca9` → 296 | `scripts/build_portfolio_dataset.py` strips identifying keys inside `metadata`. The committed extract is scrubbed (only those keys changed). The dashboard hides `metadata`/identifier columns. Tests: `tests/test_portfolio_data.py`. **History still contains the handles; rewriting history is the owner's decision.** |
| F2 | P0 | Verified, documented | **The forecast evidence is a demo sample, and the headline ranking is fragile.** All committed forecast results come from the 304-row class-stratified extract (`evaluation_status = demo_only`). The matched-origin comparison behind "naive 0.82 vs Bayesian 0.94 WAPE" is 6 forecasts over 3 origins. Origin 208 (naive's last value 22.3 happened to sit near the next actual, 26.1) produces the whole gap. Dropping it makes Bayesian the leader. On all 99 origins, every baseline has WAPE ≈ 0.99–1.02. The README presented this as "what the evaluation actually found" without the demo caveat. | `reports/demo_forecasting/matched_origin_sensitivity.json`, `matched_origin_per_origin.csv` (`python -m src.forecasting.sensitivity`) | New `src/forecasting/sensitivity.py` runs per-origin and leave-one-origin-out checks and is wired into every forecasting run and report. The README section is rewritten as a demo finding about model-selection discipline. |
| F3 | P1 | Verified, fixed | **Logistic Stage 1 probabilities were uncalibrated.** `train_stage1` skipped Platt scaling when a model emits probabilities, contrary to README §4 and the model card. With `class_weight="balanced"`, out-of-fold P(actionable) averaged 0.46 against a 0.30 base rate, 172 of 200 rows fell in 0.4–0.6, and Brier (0.217) was no better than predicting the base rate (0.21). The demand proxy `actionable_probability_mass` sums these, so it was inflated. | `reports/ml/group_cv/` before/after: Brier 0.22 → 0.18, ECE 0.21 → 0.07, mean p 0.46 → 0.32. Demo proxy total 143.8 → 119.2. | Probability outputs are now Platt-scaled on validation (`ExperimentConfig.calibrate_probability_outputs`, default true; false restores the old behaviour). Stale uncalibrated artefacts are rejected and retrained. The decision used calibration metrics only, before any forecast was re-run. Ranking metrics, thresholded F1 and the forecasting conclusions are unchanged. |
| F4 | P1 | Verified, fixed | **A demo extract could be stamped as the full dataset.** `build_temporal_dataset.py --input data/demo/signals_scored_portfolio.csv` without `--demo` wrote `sample_only=false`, `volume_representative=true`, `evaluation_status=full_dataset`, and the forecasting runner would then write to `reports/forecasting/`. | Reproduced at `ca7fca9` | `src.temporal.is_known_demo_extract` (path plus content hash, so a renamed copy is also caught) refuses this. Tests in `tests/test_temporal.py`. |
| F5 | P1 | Verified, fixed | **The MMM "reallocation" scenario added spend.** `budget_reallocation` scaled the source by `1-f` and the target by `1+f` of their *own* budgets. Social→video therefore added +267,293 total spend, and the reported "96% probability of an increase" mostly measured that extra spend. As a budget-neutral move: **−593 simulated sales (90% interval −5,733 to +3,592), P(increase) 46%.** | `reports/mmm/scenario_example.json` before/after | New budget-neutral `reallocations` in `MarketingScenario`, used by the service layer, the API (`reallocations` field) and the dashboard. The dashboard shows total spend change. MMM artefacts regenerated. |
| F6 | P1 | Verified, fixed | **The README overstated MMM recovery.** It said the model "keeps every true contribution inside its credible interval"; `recovery.csv` shows 2/3 (video's true 112,782 is above the 95th percentile). The same wording was in `docs/MMM_LAB.md`. | `reports/mmm/recovery.csv` | Both documents corrected. |
| F7 | P1 | Verified, documented | **NLP evaluation support is too thin to rank models.** Test splits are 27–42 rows from 2–7 groups. `confusion_barrier` has 0 test rows on 2 of 5 seeds, and `new_customer_interest` has 0–1. Out of fold, `high_intent` F1 is 0.00 for both supervised models. The single annotator's first-pass and final labels agree on 56.6% of rows (Cohen's κ 0.30). | `reports/data_quality/dataset_audit.md`, `reports/ml/group_cv_report.md` | New dataset audit and group-aware CV report (below). No result is relabelled as final. |
| F8 | P1 | Verified, fixed | **The dashboard's first impression was misleading.** The first tab showed KPIs and an intent distribution from the class-stratified extract, labelled by the rule baseline, with no caveat. The intent mix there is a sampling artefact. | `app.py` at `ca7fca9` | New **Overview** tab: status (development / demo / synthetic), what the data is, findings ranked by defensibility, blocked work, links to evidence, all read from artefacts. Caveat banner on the extract tab. Per-tab exception guard. |
| F9 | P1 | Suspected | **The live dashboard may not be publicly viewable.** An anonymous request to https://kh4-demand-intel.streamlit.app returns HTTP 303 to `share.streamlit.io/-/auth/app` (a sign-in page). This cannot be resolved from here; the owner should check sharing settings in Streamlit Community Cloud. | `curl -I` on 2026-10-17 | None (needs the owner). Headless startup is now tested in CI (`tests/test_app_smoke.py`). |
| F10 | P2 | Verified, fixed | **A smoke benchmark overwrote committed split manifests.** `run_benchmark` always wrote to `reports/splits/`, so CI's `--output /tmp/ci-ml` smoke step (or any local run elsewhere) overwrote them. | Reproduced: `git status` after a run with `--output` elsewhere | Manifests go to `<output>/splits` unless the output is `reports/ml`. |
| F11 | P2 | Verified, fixed | **Missing canonical input gave a traceback, not an abstention.** | `python scripts/run_forecasting.py` with no input | Explicit `SystemExit` with the requirements. New `scripts/check_full_data_readiness.py` (exit 2 when blocked). |
| F12 | P2 | Verified, documented | **One cross-group near-duplicate.** One pair (cosine 0.85, different videos, both `general_discussion`) can sit on both sides of a group split. There are no exact duplicates and no shared groups on any seed. | `dataset_audit.md` | Reported; the impact is negligible at one pair. |
| F13 | P2 | Verified, documented | **The previous active-learning queue had weak provenance.** `record_id` was a row position in an extract with identifiers stripped; the queue had no missing-class or unseen-group stratum and showed model labels to the annotator. | `reports/annotation/active_learning_queue.csv` | New stratified blind queue with provenance (below). The old queue is kept unchanged. |
| F14 | P2 | Verified, documented | **There is no annotation-date gating source.** `data/reference/events.csv` has no `announced_date`, so gating falls back to `event_date`. That is conservative and correct, but the gating cannot be exercised on real data. Both events post-date the observed window, so no event effect is identifiable. | `src/events.py:74`, `temporal_coverage.json` | Documented. Add `announced_date` when known. |
| F15 | P3 | Verified, documented | **The new evaluation suggests a better model than the one deployed.** In group-aware CV, `char_word_svm` beats the deployed `logistic` (PR-AUC 0.65 vs 0.57; ECE 0.05 vs 0.07). | `reports/ml/group_cv_report.md` | Not switched. Choose the model under a frozen protocol on the expanded corpus, not here. |

### Checked and sound (no change needed)

- **Group isolation:** no discussion group is shared across train/validation/test on any benchmark seed; exact-duplicate texts are absent (`check_leakage` on seeds 42–46).
- **Out-of-fold provenance:** of the 304 demo rows, 160 match the corpus by normalised text and use OOF values, 144 use the persisted model, and 0 are excluded. No training row is scored in-sample.
- **Calibration and thresholds:** both are fitted on validation only; the test split is never touched before evaluation. (The validation set is about 30 rows, so both are noisy; see F7.)
- **Timestamps:** 304/304 valid, with no epoch-zero, future or naive values (`weekly_demand_signals_demo.meta.json`).
- **Forecast origin isolation:** future values cannot change an earlier origin, and event features are rebuilt per origin from `announced_date <= origin` (tests in `tests/forecasting/test_validation.py`).
- **Synthetic MMM isolation:** every row carries `data_type = synthetic`, the fitter refuses non-synthetic input, convergence is clean (R-hat ≤ 1.006, 0 divergences), and the PPC mean and sd match within 1%.
- **Secrets:** no API keys are committed (`.env.example` holds placeholders only; `git grep` for key patterns finds nothing).
- **CI:** green on `main` (run 72).

## What changed (measurable)

| Area | Before | After |
|---|---|---|
| Usernames in public extract | 296 rows | 0 (test-enforced) |
| Logistic OOF calibration (ECE / Brier / mean p vs 0.30) | 0.21 / 0.22 / 0.46 | 0.07 / 0.18 / 0.32 |
| MMM scenario total-spend change | +267,293 | 0 (budget-neutral) |
| MMM scenario P(increase) | 96% (artefact) | 46% |
| Headline forecast ranking | Stated as a finding | Per-origin + leave-one-out: fragile, origin 208 drives it |
| NLP evaluation rows with a held-out prediction | 27–42 per seed | 200 (pooled group-CV) with group-bootstrap CIs |
| Tests | 317 fast + 23 PyMC | 342 fast + 23 PyMC, plus headless dashboard startup |

### Current development numbers (preliminary; not final held-out)

Group-aware 5-fold CV, 3 fold seeds, 200 rows, 23 groups (`reports/ml/group_cv_report.md`):

| Model | Stage 1 PR-AUC (group-bootstrap 95%) | Lift@10% (95%) | Brier | ECE | e2e macro-F1, observed classes |
|---|---|---|---|---|---|
| char_word_svm | 0.65 (0.55–0.77) | 2.5 (2.1–3.3) | 0.16 | 0.07 | 0.31 |
| logistic | 0.57 (0.47–0.69) | 2.1 (1.7–2.9) | 0.18 | 0.07 | 0.25 |

The base rate (PR-AUC of a random ranking) is 0.30. Out-of-fold `high_intent` F1 is 0 for
both models. Error slices: long comments (>25 words) have a 52% false-positive rate
against 9% for short ones, so length is a confounder worth controlling in the next
annotation batch.

The demo forecast (unchanged conclusion; `reports/demo_forecasting/`): naive 0.819 vs
Bayesian 0.929 WAPE on 6 matched forecasts; leave out origin 208 and the two tie at
0.961 vs 0.959. The Bayesian model converges (R-hat 1.005, 0 divergences), but its PPC sd
is about half the observed sd, so it understates spikes. **No model is reliable on this
history, and none is recommended.**

## End-to-end pass (2026-10-19)

A fresh clone of the PR branch, installed on Python 3.12 exactly as the README says,
with every README command run as written, the dashboard driven in a headless browser,
the API exercised, and the dashboard tested on a `requirements.txt`-only install (what
Streamlit Community Cloud uses).

| ID | Sev | Status | Finding | Fix |
|---|---|---|---|---|
| E1 | P1 | Verified, fixed | **Both Scenario Planner modes and both scenario API endpoints are dead in a deployed checkout.** They need posterior samples (`*.npz`), which are git-ignored, so a visitor clicking *Run* gets "run `scripts/run_mmm.py --save-posterior`" (HTTP 409 from the API). | Every forecasting and MMM run now writes thinned (≤1,000-draw, float32), run-stamped posterior samples that are committed; services refuse a stamp that does not match the report's run ID. Verified in a fresh clone: both dashboard modes and both API endpoints return results. |
| E2 | P1 | Verified, fixed | **Live Inference uses the rule baseline on a deployed checkout**, because `artifacts/` is git-ignored. The tab says it shows a trained development model; the output says `model_name: rules`. | `src.ml.inference.ensure_live_model` trains the same logistic model once per process from the committed corpus when no artefact exists, and the tab states the model's origin, seed and dataset hash. |
| E3 | P1 | Verified, fixed | **A deprecation time bomb.** All 12 tables used `use_container_width`, which Streamlit flags for removal after 2025-12-31, and the requirement allowed any `streamlit<2`. | Switched to `width="stretch"`; floor raised to `streamlit>=1.50` (tested on 1.50.0 and 1.66.0). |
| E4 | P2 | Verified, fixed | **A wrong number in the README:** "16 of 41 (39%)" negative comments actionable; the corpus and the dashboard both give 17 of 41 (41%). Every other quoted number was re-checked against its artefact and matches. | Corrected. |
| E5 | P2 | Verified, fixed | **A stale committed artefact:** `reports/annotation/active_learning_queue.csv` was generated with the pre-calibration model, so a fresh run changed it. All other committed ML reports regenerate exactly (only timestamps and run IDs differ). | Regenerated; it is byte-identical to a fresh-clone run. |
| E6 | P2 | Verified, fixed | **First impression:** the header still read "a lightweight portfolio dashboard … heuristic demand signals", and the Demand tab's filters sat in the sidebar on every tab, doing nothing outside that tab. | New header text; filters moved into the Demand Intelligence tab. A README screenshot was added, since the live link is not public. |
| E7 | P3 | Verified, fixed | `run_temporal_pipeline.py` and `enrich_timestamps` showed raw tracebacks on missing input. | Clean abstention messages. |
| E8 | P2 | Verified, fixed | **The Events tab is always empty in deployment**, because it reads git-ignored `data/processed/` outputs. | The tab is shown only when its outputs exist. |
| F9 | P1 | Re-verified, open | The live URL still redirects anonymous visitors to Streamlit sign-in (HTTP 303). | Owner: check the app's sharing settings in Streamlit Community Cloud. |

## Reproducible commands

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[ml,app,dev,forecast,api]"

ruff check src/ tests/ scripts/ app.py
pytest -m "not pymc_smoke"            # 342 passed
pytest -m pymc_smoke                  # 23 passed

python -m src.ml.data validate
python -m src.ml.dataset_audit        # reports/data_quality/dataset_audit.md
python -m src.ml.benchmark            # reports/ml/ (reproduces exactly)
python -m src.ml.group_cv             # reports/ml/group_cv_report.md
python -m src.ml.train --model logistic
python -m src.ml.annotation_queue build
python scripts/check_full_data_readiness.py   # exits 2: blocked_missing_input
python scripts/build_temporal_dataset.py --demo
python scripts/run_forecasting.py --demo --chains 4
python scripts/run_mmm.py --chains 4
```

## Annotation workflow added

`python -m src.ml.annotation_queue build` writes:

- `reports/annotation/annotation_queue.csv`: blind annotator file (no rule or model outputs), with empty label columns and a `double_annotate` flag on about 15% of rows for a second, independent annotator.
- `annotation_queue_provenance.csv`: pool path and hash, pool row, rule/model outputs, stratum reason and seed, keyed by a stable content hash `queue_id`.
- `annotation_queue_manifest.json`: quotas, eligible counts and **shortfalls**.

Strata, filled in this order: random control (drawn first, so it stays uniform), rare- or missing-class candidates, Reddit, unseen discussion groups, rule/model disagreement or near-threshold. Rows already labelled, duplicates and near-duplicates (cosine ≥ 0.80) of labelled rows are excluded. **No label is generated.** `validate` QC-checks a completed batch (annotator present, labels in the taxonomy, no corpus overlap, no duplicates, inter-annotator κ on the overlap). Accepted rows go to a separate `human_reviewed_unmerged` file; the corpus is never modified.

Built from the only pool available here (the 304-row demo extract), the committed queue has 75 rows and large shortfalls: Reddit 2/30, unseen groups 0/25, missing-class candidates 18/40. **Rebuild it from the canonical signal set before annotating.**

## Blocked by missing data or human work

1. **Full-data forecasting evaluation.** Needs `data/processed/signals_scored.csv`, which is not committed. `scripts/check_full_data_readiness.py` lists the inputs and the exact commands (YouTube API key, scrape, `run_pipeline.py`, build, forecast, sensitivity).
2. **Missing classes** (`content_drought_fatigue`, `expectation_decay`) and **rare classes** (`new_customer_interest` 2, `high_intent` 7, `nostalgia_reactivation` 8). Needs human labels from the stratified queue, ideally rebuilt on the canonical pool.
3. **Reddit performance.** 3 labelled rows. Needs Reddit-heavy annotation.
4. **Inter-annotator agreement.** Needs a second annotator on the `double_annotate` rows.
5. **Final held-out evaluation.** Needs the expanded corpus, then a test set frozen before any further model or threshold change.
6. **Public dashboard access (F9).** The owner needs to check the Streamlit Cloud sharing setting.
7. **Usernames in git history (F1).** Removing them needs a history rewrite and force-push, which is the owner's decision.

## Recommended next steps

1. Rebuild the canonical signal set, run `check_full_data_readiness.py`, then the full forecast with sensitivity.
2. Rebuild the annotation queue on that pool; label about 300 rows with a 15% double-annotated overlap; validate; merge in a reviewed commit with a new `annotation_version`.
3. Freeze the evaluation protocol (group-CV + one frozen group-held-out test set) *before* choosing between `logistic` and `char_word_svm`.
4. Add `announced_date` to `data/reference/events.csv` where known, and extend coverage past the June/August 2026 events so event effects become identifiable.

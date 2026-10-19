# Full-data forecasting readiness

**Status: `blocked_missing_input`**

No forecasting result is reported until the canonical run completes. The committed reports/demo_forecasting/ results are a demo smoke run over a class-stratified sample and must not be quoted as measured.

## What exists

- Canonical signals (`data/processed/signals_scored.csv`): **absent**
- Weekly table: absent (status `None`)
- Full forecasting results: absent
- YouTube fetch manifest: absent

## Problems

- data/processed/signals_scored.csv does not exist in this checkout.
- No YouTube fetch manifest, so volume representativeness cannot be verified; a full run would be stamped full_dataset on trust.

## Missing inputs

- `data/processed/signals_scored.csv`: the full scored signal set produced by `scripts/run_pipeline.py` (~4,831 rows when last built locally). Required columns: `text`, `timestamp`, `source`, `intent_label`.
- Creation timestamps from the source API (YouTube `publishedAt`, Reddit `created_utc`) in `timestamp`; rows without one are excluded, never imputed.
- `YOUTUBE_API_KEY` in `.env` to regenerate YouTube comments (`data/raw/` is git-ignored).
- `reports/tables/youtube_fetch_manifest.csv` from the scraper, so the build can tell whether any video hit the `--max-results` cap (seed-capped collections are stamped `volume_representative = false`).

## Commands

```bash
cp data/raw/youtube/video_seed_list.example.csv data/raw/youtube/video_seed_list.csv
python -m src.scraper_youtube --seed-list --max-results 100 --fetch-metadata
python -m src.scraper_reddit  # optional; see README section 17
python scripts/run_pipeline.py  # preprocess -> sentiment -> intent -> score
python scripts/check_full_data_readiness.py
python scripts/build_temporal_dataset.py --input data/processed/signals_scored.csv
python scripts/run_forecasting.py --chains 4 --save-posterior
python -m src.forecasting.sensitivity --report-dir reports/forecasting
```

# Kingdom Hearts IV: Demand Intelligence

A deployed fan-signal analytics system for *Kingdom Hearts IV*, using Reddit discussion data to surface player sentiment, frustration topics, and a re-engagement scoring model for lapsed franchise players.

> Built as a portfolio piece following a final-round interview at Square Enix's recommendation team. The interview revealed the gap to close was production deployment evidence, not theoretical knowledge — this project is the answer.

**Status:** In development. Live dashboard target: end of Week 4.

## The thesis

Square Enix has been near-silent on KH4 since the 2022 reveal. The hypothesis: this silence translates to **awareness without intent** — players know KH4 exists but aren't actively forming purchase intent. This project quantifies that with social-signal data and proposes a re-engagement scoring framework rooted in the same recommendation-system thinking used by Square Enix's analytics function.

## What this builds

A full pipeline from raw social signal to deployed decision tool:

1. **Ingest** — Reddit discussion across r/KingdomHearts, r/SquareEnix, r/JRPG (multi-source ready: YouTube and Trends in v2)
2. **Process** — Sentiment classification (VADER → DistilBERT) and topic clustering (BERTopic)
3. **Score** — Re-engagement score per topic cluster, intent classification per signal
4. **Serve** — Live Streamlit dashboard backed by a FastAPI prediction endpoint
5. **Document** — Case-study README with architecture, trade-offs, failure modes, and "what I'd do at production scale"

## Roadmap

### MVP (Weeks 1–4) — ship the clickable thing

| Week | Deliverable |
|------|-------------|
| 1 | Reddit scraper + raw dataset + EDA notebook |
| 2 | VADER sentiment + BERTopic clustering + insight charts |
| 3 | Re-engagement scoring + Streamlit dashboard skeleton |
| 4 | **Deploy live + case-study README + LinkedIn announcement** |

### Phase 2 (Weeks 5–8) — make it interview-grade

| Week | Deliverable |
|------|-------------|
| 5 | FastAPI prediction endpoint; Streamlit calls it (service architecture, not notebook) |
| 6 | YouTube ingestion through unified schema (no pipeline rewrite — just plug in) |
| 7 | DistilBERT replaces VADER; MLflow experiment tracking |
| 8 | Cross-platform divergence detection: where Reddit and YouTube diverge in signal |

### Phase 3 (Weeks 9–12) — production polish

| Week | Deliverable |
|------|-------------|
| 9–10 | Docker, basic monitoring, scheduled re-scrape via GitHub Actions cron |
| 11–12 | Full case-study write-up, second smaller portfolio project, applications |

## Repo structure

```
kh4-demand-intel/
├── src/
│   ├── schema.py          # Unified SignalRecord — works for any source
│   ├── scraper_reddit.py  # Reddit ingestion (MVP)
│   └── scraper_youtube.py # Phase 2
├── tests/                 # pytest suite, runs in CI
├── data/
│   ├── raw/
│   │   ├── reddit/        # Reddit scraper output (gitignored)
│   │   └── youtube/       # Phase 2
│   └── processed/         # Cleaned + scored data
├── notebooks/             # EDA and analysis
├── models/                # Saved model artefacts (gitignored)
├── app/                   # Streamlit dashboard (Week 3)
├── api/                   # FastAPI endpoint (Week 5)
└── .github/workflows/     # CI: lint + test on every push
```

## Local setup

```bash
git clone https://github.com/<your-username>/kh4-demand-intel
cd kh4-demand-intel
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your Reddit API credentials
python src/scraper_reddit.py --limit 10 --top-n 5    # quick smoke test
```

## Reddit API credentials

You need a Reddit "script" app to use PRAW. Free, takes ~3 minutes:

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app" at the bottom
3. Choose **script**, fill in name; redirect URI can be `http://localhost:8080`
4. Copy `client_id` (under the app name) and `client_secret` into your `.env`
5. Set `REDDIT_USER_AGENT` to something descriptive: `kh4-demand-intel by u/yourusername`

## Case study (drafted at end of MVP)

The final write-up will cover problem framing, data sources, pipeline design, model trade-offs (VADER vs DistilBERT, BERTopic vs LDA), scoring logic, deployment architecture, failure modes, and what would change at production scale — the last section being the direct answer to the question that decided the Square Enix interview.

## Author

**Ghobi Ara** — MSc Data Science, City University of London. Background in electrical engineering, applied ML in e-commerce and marketing analytics, dissertation on differential privacy.

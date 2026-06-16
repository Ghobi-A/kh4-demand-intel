# Kingdom Hearts IV: Demand Intelligence

**Live dashboard:** https://kh4-demand-intel.streamlit.app

**Key finding:** sentiment is not behavioural intent. Positive comments can still be low-action lore discussion, while negative or fatigued comments can reveal demand risk, confusion barriers, or reactivation opportunities.

An evaluated Python NLP pipeline that transforms public Reddit and YouTube community discussion around Kingdom Hearts IV into structured sentiment, intent, and demand signals for a Streamlit Community Cloud portfolio dashboard.

**Unofficial portfolio project. Not affiliated with Square Enix, Disney, or the Kingdom Hearts franchise. Data sourced from public community discussion.**

Hosted on Streamlit Community Cloud free tier; first load may take 20–30 seconds if the app has been idle.

---

## Project status

**Implemented**
- Reddit + YouTube ingestion utilities
- Preprocessing pipeline
- VADER sentiment analysis
- Rule-based behavioural intent classification
- Demand, risk, and activation scoring layer
- Lightweight Streamlit dashboard
- Manually audited evaluation corpus
- `pytest` regression testing with precision floors
- Portfolio dataset build script for deployment-ready dashboard extracts

**This project is currently an evaluated analytical prototype, not a production demand model.**

---

## Pipeline

```
Ingest → Clean → Sentiment → Intent → Score → Audit → Portfolio Dashboard
```

- **Ingest**: Reddit and YouTube community comments collected by local utilities
- **Clean**: Text normalization and schema alignment
- **Sentiment**: VADER polarity scoring
- **Intent**: Rule-based behavioural intent classification
- **Score**: Heuristic demand, activation, and risk scoring for decision support
- **Audit**: Manual corpus review and precision-floor validation with regression checks
- **Portfolio extract**: A privacy-conscious dashboard dataset sampled from local processed outputs

---

## Empirical audit and methodology

The intent layer was evaluated with a **200-row manually labelled audit corpus**. That audit showed that emotional polarity and behavioural demand signals are different modelling targets: a comment can be positive but non-actionable, or negative while still revealing important demand risk.

The audit process was used to:
- inspect intent-class precision in ambiguous community phrasing,
- identify recall-slice contamination across neighboring categories,
- refine taxonomy boundaries for behaviourally meaningful interpretation,
- enforce regression-tested precision floors for high-value classes.

Regression tests enforce these minimum acceptable precision floors:

- `high_intent`: **0.55**
- `nostalgia_reactivation`: **0.65**
- `new_customer_interest`: **0.70**

These are not probability-calibrated classifier confidence thresholds. The `intent_confidence` field emitted by the rule-based baseline is a simple baseline confidence indicator for matched versus unmatched rules, not a calibrated probability.

---

## Intent taxonomy

| Intent | Meaning |
|------|--------|
| high_intent | Explicit purchase intent |
| nostalgia_reactivation | Legacy attachment driving re-engagement |
| new_customer_interest | Signals from potential new players |
| frustrated_demand | Demand blocked by lack of updates |
| content_drought_fatigue | Coping signals during content drought |
| confusion_barrier | Narrative complexity reducing accessibility |
| expectation_decay | Disengagement after prolonged silence |
| general_discussion | Non-actionable engagement |

---

## Connection to recommendation systems (conceptual)

This repository is a **demand-intelligence / NLP signal analysis pipeline**, not a production recommender.

Its relevance to recommendation-system practice is conceptual:
- **Implicit behavioural signals**: social discussion is treated as weak feedback
- **Intent modelling**: language patterns are abstracted into behavioural states
- **Weak-signal prioritisation**: low-frequency but high-value demand cues are surfaced
- **Re-engagement framing**: friction and latent demand are emphasized
- **Decision-support parallels**: outputs can inform prioritisation choices in product/marketing workflows

---

## Current dataset (~4.8k local signals)

Combined Reddit + YouTube local sample:

- **Total signals:** 4,831
- **Positive sentiment:** 2,748
- **Negative sentiment:** 1,025
- **Neutral:** 1,058

### Intent distribution

- general_discussion: 4,273
- nostalgia_reactivation: 313
- frustrated_demand: 121
- new_customer_interest: 49
- high_intent: 46
- confusion_barrier: 29

The full processed scored dataset may exist only locally as `data/processed/signals_scored.csv` and is not required to be present when cloning the repository.

---

## Visual outputs

### Sentiment distribution
![Sentiment distribution](reports/figures/sentiment_distribution.png)

### Intent distribution
![Intent distribution](reports/figures/intent_distribution.png)

### Actionable demand signals
![Actionable intent distribution](reports/figures/actionable_intent_distribution.png)

---

## Dashboard

A lightweight Streamlit dashboard is implemented in `app.py` for reviewing portfolio-ready demand intelligence outputs. By default it loads:

- `data/demo/signals_scored_portfolio.csv`
- `data/demo/video_scores_portfolio.csv`

Run locally:

```bash
streamlit run app.py
```

For Streamlit Community Cloud, install the runtime dependencies and run the app with the default portfolio data paths:

```bash
pip install -r requirements.txt
streamlit run app.py
```

If dashboard input files are missing, the app shows a clear in-page warning instead of crashing.

---

## Building the portfolio dataset

The portfolio CSVs are generated from the local scored dataset:

```bash
python scripts/build_portfolio_dataset.py
```

The script:
- requires `data/processed/signals_scored.csv`,
- fails clearly if that local processed file is missing,
- strips identifying/user-level metadata columns when present,
- samples up to 60 rows per intent class,
- patches missing intent classes from the corrected audit labels when possible,
- re-scores patched audit rows with the same scoring code used by the pipeline,
- writes dashboard-ready signal and video-score CSVs under `data/demo/`.

It does **not** scrape new data or fabricate replacement portfolio rows when the processed input is absent.

---

## Scoring note

`demand_score` is an explainable prioritisation heuristic, not a forecast or personalised recommendation score. It uses an intent/sentiment base signal amplified by logged non-negative engagement:

```text
base_signal = intent_weight + sentiment_weight
engagement_multiplier = 1 + (log1p(clipped_engagement) / 5)
demand_score = base_signal * engagement_multiplier
```

This keeps engagement useful as supporting evidence while preventing raw popularity from being added directly on top of negative or low-intent signals.

---

## Limitations

- The rule-based intent classifier is an evaluated baseline, not a learned model.
- Minimum acceptable precision floors were set on a relatively small audit corpus and enforced by regression tests.
- `intent_confidence` is a simple baseline confidence indicator, not a probability-calibrated classifier output.
- Community-source selection introduces platform/video/subreddit bias.
- Temporal demand dynamics are not modelled yet.
- Demand scores are heuristic decision-support signals, not forecasts.
- The hosted dashboard uses a portfolio extract rather than requiring the full local processed dataset.

---

## Author

**Ghobikan Aravindan**

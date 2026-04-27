# Kingdom Hearts IV: Demand Intelligence

A deployed fan-signal analytics system for **Kingdom Hearts IV**, designed to translate community behaviour into decision relevant insight.

Built following a final-stage interview with Square Enix’s recommendation team — where the identified gap was not modelling knowledge, but **production level execution and decision framing**.

---

## Overview

This project models **player demand under uncertainty**, using social signals as a proxy for intent, engagement, and disengagement.

**The core problem:**

Square Enix has maintained near silence on KH4 since its 2022 reveal.

**Hypothesis:**

> Silence creates awareness without conversion — players know KH4 exists, but may not be forming strong purchase intent.

This system quantifies that gap.

---

## What this builds

A full pipeline from raw community signals to actionable insight:

```
Ingest → Clean → Sentiment → Intent → Score → Serve
```

- **Ingest** — Reddit (PullPush) + YouTube comments  
- **Process** — Cleaning + VADER sentiment scoring  
- **Interpret** — Intent classification layer (behavioural signals)  
- **Score** — Re-engagement scoring (lapsed player recovery potential)  
- **Serve** — Streamlit dashboard + FastAPI endpoint (planned)  

---

## Methodology

### 1. Multi-source signal ingestion
- Reddit discussions (long-form reasoning, community sentiment)  
- YouTube comments (high-volume, reactive sentiment)  
- Unified into a single schema (`SignalRecord`)  

---

### 2. Sentiment layer (VADER)

Each signal is scored using:

- neg / neu / pos probabilities  
- compound score  

Mapped into:

- positive  
- negative  
- neutral  

This provides baseline emotional polarity — but not actionability.

---

### 3. Intent classification (MVP layer)

To bridge that gap, signals are mapped into behavioural intent categories:

| Intent | Meaning |
|------|--------|
| high_intent | Explicit purchase intent |
| frustrated_demand | Demand blocked by lack of updates |
| nostalgia_reactivation | Returning players driven by legacy attachment |
| new_customer_interest | Signals from potential new players |
| confusion_barrier | Narrative complexity reducing accessibility |
| general_discussion | Non-actionable engagement |

This reframes sentiment into **decision relevant demand states**.

---

### 4. Re-engagement framing

The system is designed to answer:

```
Not: “Are players positive?”
But: “What is preventing conversion?”
```

This aligns with recommendation-system thinking:

- identifying latent demand  
- quantifying conversion friction  
- prioritising intervention strategies  

---

## Current dataset (~4.8k signals)

Combined Reddit + YouTube sample:

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

---

## Key insights

### 1. Demand exists, but is friction-constrained

High presence of *frustrated_demand* relative to *high_intent*:

> Interest is present, but not converting into action.

---

### 2. Nostalgia dominates engagement

Strong signals tied to KH1/KH2:

> Engagement is anchored in legacy identity rather than new narrative.

---

### 3. Weak new-player acquisition signal

Low *new_customer_interest*:

> KH4 currently behaves as a **retention-driven product**, not a growth-driven one.

---

### 4. Core risk = communication gap

Sentiment is largely positive, but:

- repeated references to “no news”  
- declining trust in updates  

> The primary risk is not negativity — it is **uncertainty**.

---

## Example: signal interpretation

| Pattern | Meaning | Action |
|------|--------|--------|
| “4 years no news” | Frustrated demand | Increase communication cadence |
| KH2 nostalgia spikes | Reactivation anchor | Leverage legacy callbacks |
| “Story too confusing” | Barrier to entry | Improve onboarding / recap content |
| Low high_intent signals | Weak conversion | Strengthen marketing clarity |

---

## Roadmap

### MVP (Weeks 1–4)
- Data ingestion (Reddit + YouTube)  
- Sentiment + intent pipeline  
- Initial dashboard  
- Deployment  

### Phase 2
- DistilBERT sentiment (replace VADER)  
- BERTopic clustering  
- FastAPI service layer  
- Google Trends integration  

### Phase 3
- Scheduled data refresh  
- Monitoring + Dockerisation  
- Cross-platform divergence analysis  

---

## Limitations

- Rule-based intent classification (no ML generalisation yet)  
- Dataset bias based on selected videos  
- No temporal modelling of demand evolution  
- Early-stage scoring framework  

---

## Positioning

This is not a sentiment analysis project.

It is a **demand intelligence system** designed to bridge:

```
community signals → product & marketing decisions
```

---

## Author

**Ghobi Ara**  
MSc Data Science — City, University of London  

Built as a direct response to a Square Enix recommendation systems interview, focusing on **execution, deployment, and decision relevance over theory**.

# Kingdom Hearts IV: Demand Intelligence

A deployed fan-signal analytics system for **Kingdom Hearts IV**, designed to translate community behaviour into **decision relevant insights**.

Built to explore how player sentiment and behaviour can be translated into actionable signals within live game ecosystems.

---

## Overview

This project models **player demand under uncertainty**, using social signals as a proxy for intent, engagement, and disengagement.

This project applies **recommendation-system thinking** to unstructured community data, treating social signals as implicit feedback for demand modelling.

**The core problem:**

Square Enix has maintained near silence on KH4 since its 2022 reveal.

**Hypothesis:**

> Silence creates awareness without conversion, players know KH4 exists, but are not forming strong purchase intent.

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

This reframes sentiment into **decision-relevant demand states**.

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

## Connection to recommendation systems

This system mirrors core recommendation-system principles, reframed for demand intelligence:

- **User intent modelling** → Signals classified into behavioural intent states  
- **Implicit feedback modelling** → Social signals used as behavioural proxies  
- **Cold-start handling** → Works without direct gameplay or purchase data  
- **Re-ranking logic** → Prioritises signals based on re-engagement potential  
- **Conversion optimisation** → Focus on influencing behaviour, not just measuring it  

Each signal can be treated as a weak feedback signal:

```
user sentiment + intent → proxy for likelihood of engagement or conversion
```

This positions the system as an upstream layer to recommendation or marketing decision pipelines.

---

## Future recommendation system extensions

This system can be extended into a full recommendation or decision-ranking framework:

- **User-level embeddings**  
  Aggregate signals by user or cohort to model player personas and behavioural profiles  

- **Content/topic embeddings**  
  Represent themes (e.g. nostalgia, confusion, hype) as vectorised features for ranking  

- **Re-ranking layer**  
  Rank topics or signals based on expected impact on engagement or conversion  

- **Temporal modelling**  
  Track how intent evolves over time (e.g. decay of hype, spikes after announcements)  

- **Intervention optimisation**  
  Use signals to prioritise actions such as trailers, recaps, or marketing beats  

In this form, the pipeline transitions from descriptive analytics into a **decision-support system aligned with recommendation system architectures**.

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

## Decision framing (example)

Given the current signal distribution:

- High frustrated demand relative to high intent  
- Strong nostalgia-driven engagement  
- Weak new player acquisition signals  

A likely intervention strategy would be:

- Increase communication cadence to reduce uncertainty  
- Leverage legacy callbacks to activate nostalgia segments  
- Introduce onboarding or recap content to reduce narrative barriers  

This demonstrates how community signals can be translated into concrete product and marketing actions.

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

**Ghobi Aravindan**  
MSc Data Science — City, University of London  

Focus: applied analytics, behavioural signal modelling, and decision systems.

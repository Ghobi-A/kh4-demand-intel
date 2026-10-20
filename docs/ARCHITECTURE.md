# Architecture

Four kinds of data move through this system, and the diagram keeps them visually apart:
**observed public discussion** (blue), **derived behavioural demand proxies** (green),
**forecasts and model outputs** (amber), and **synthetic marketing data** (grey). The
synthetic lane shares no inputs with the observed lanes; they meet only in the interface
layer, where every synthetic view is labelled as simulated. Nothing here is sales data, and
discussion volume is never presented as sales.

```mermaid
flowchart TB
    classDef observed fill:#dbeafe,stroke:#1d4ed8,color:#0b1d4d
    classDef proxy fill:#dcfce7,stroke:#15803d,color:#0b3d1d
    classDef model fill:#fef3c7,stroke:#b45309,color:#4a2a00
    classDef synthetic fill:#e5e7eb,stroke:#4b5563,color:#1f2937,stroke-dasharray: 5 3
    classDef human fill:#fce7f3,stroke:#be185d,color:#4a0a26

    subgraph ingest["Ingestion and preprocessing (observed)"]
        YT["YouTube Data API<br/>src/scraper_youtube.py<br/>+ fetch manifest"]:::observed
        RD["Reddit / PullPush<br/>src/scraper_reddit.py"]:::observed
        PRE["Preprocess: drop empty/deleted, dedupe ids,<br/>strip URLs, parse UTC timestamps<br/>src/preprocess.py, src/timestamps.py"]:::observed
        SENT["VADER sentiment<br/>src/sentiment.py"]:::observed
        RULE["Regex intent rules (baseline)<br/>src/intent.py"]:::model
        YT --> PRE
        RD --> PRE
        PRE --> SENT --> RULE
    end

    subgraph labels["Human labels and annotation loop"]
        CORPUS["200-row audit corpus<br/>1 annotator, rule-stratified sample"]:::human
        QUEUE["Stratified blind queue<br/>group cap, inclusion probabilities<br/>src/ml/annotation_queue.py"]:::human
        ADJ["Double annotation and adjudication<br/>kappa, PABAK, blind adjudication<br/>src/ml/adjudication.py"]:::human
        QUEUE --> ADJ -. "reviewed merge,<br/>new annotation_version" .-> CORPUS
    end

    subgraph nlp["Supervised NLP (development-preliminary)"]
        TRAIN["Hierarchical classifier<br/>Stage 1 actionable / Stage 2 intent<br/>TF-IDF LR, SVMs, frozen MiniLM"]:::model
        CAL["Validation-only Platt calibration<br/>+ validation-selected threshold"]:::model
        EVAL["Group CV, paired group bootstrap,<br/>design-weighted sensitivity, errors<br/>src/ml/group_cv.py, model_comparison.py"]:::model
        TRAIN --> CAL --> EVAL
    end

    subgraph serve["Scoring and temporal proxy"]
        OOF["Out-of-fold P(actionable) for labelled rows;<br/>persisted model for the rest<br/>src/ml/probabilities.py"]:::model
        BATCH["Batch inference + monitoring<br/>coverage, PSI, intent mix<br/>src/ml/batch_inference.py"]:::model
        WEEK["Weekly demand proxies<br/>sum of P(actionable), counts<br/>src/temporal.py"]:::proxy
        OOF --> BATCH
        OOF --> WEEK
    end

    subgraph fc["Forecasting and monitoring"]
        BT["Rolling-origin backtest<br/>baselines + hurdle-Gamma / NegBin<br/>announcement-date event gating"]:::model
        SENS["Matched-origin sensitivity<br/>src/forecasting/sensitivity.py"]:::model
        MON["Bias, interval coverage,<br/>reliability flags<br/>src/forecasting/monitoring.py"]:::model
        BT --> SENS
        BT --> MON
    end

    subgraph mmm["Synthetic marketing lab (simulated, not Square Enix data)"]
        SIM["Simulated spend, price, promotion, sales<br/>with known ground truth<br/>src/mmm/synthetic.py"]:::synthetic
        FIT["Bayesian MMM: adstock + Hill saturation<br/>recovery checks, identifiability notes"]:::synthetic
        SCN["Budget-neutral scenarios"]:::synthetic
        SIM --> FIT --> SCN
    end

    subgraph ui["Interfaces (read generated artefacts; no Bayesian fitting at request time)"]
        SVC["src/services.py<br/>run-stamped thinned posteriors"]:::model
        APP["Streamlit dashboard<br/>Overview first"]:::model
        API["FastAPI service"]:::model
        SVC --> APP
        SVC --> API
    end

    RULE --> CORPUS
    RULE --> QUEUE
    CORPUS --> TRAIN
    CAL --> OOF
    RULE --> OOF
    WEEK --> BT
    MON --> SVC
    SENS --> SVC
    EVAL --> APP
    BATCH --> APP
    SCN --> SVC
```

## Data-type boundaries enforced in code

| Boundary | Where it is enforced |
|---|---|
| A demo extract is never stamped as the full dataset | `src/temporal.py::is_known_demo_extract` (path and content hash) |
| Labelled rows are never scored in-sample for the proxy | `src/ml/probabilities.py` (`oof_supervised` / `excluded_training_row`) |
| The rule 0.7/0.3 indicator is never a probability | `src/ml/models.py` score types; Platt-calibrated only on validation |
| The MMM fitter refuses observed data | `src/mmm/model.py` raises when `data_type` is present and not `synthetic` (a frame with no `data_type` column is not checked) |
| Scenarios use only samples from the run they describe | `src/posterior_store.py` (run-ID stamp checked by `src/services.py`) |
| Model outputs never become labels | `src/ml/annotation_queue.py` (blind file), `src/ml/adjudication.py` |

## Batch versus interactive paths

| Path | Command | Model | Notes |
|---|---|---|---|
| Batch scoring | `python -m src.ml.batch_inference --input X.csv --output Y.csv` | persisted or trained-on-demand logistic | writes `Y.monitoring.json` |
| Weekly proxy | `python scripts/build_temporal_dataset.py` | same, out of fold for labelled rows | provenance per row |
| Interactive | dashboard *Live Inference*, `src.ml.inference.predict_text` | same; trained once per process if no artefact | rule fallback only if training is impossible |

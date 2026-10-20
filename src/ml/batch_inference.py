"""Batch scoring of a signal file with the supervised model, plus a monitoring summary.

    python -m src.ml.batch_inference --input data/demo/signals_scored_portfolio.csv \
        --output reports/inference/portfolio_scored.csv

Probabilities come from ``src.ml.probabilities.build_probabilities``, the same
path the weekly demand proxy uses, so provenance is identical. Rows in the
labelled corpus get *out-of-fold* values, rows the model trained on but could
not score out of fold are marked ``excluded_training_row`` (never scored
in-sample), and every other row uses the persisted model. Intent labels come
from the same hierarchical model.

Next to the scored file, ``<output stem>.monitoring.json`` records what a
batch monitor should check:

* coverage: rows, scorable rows, provenance counts, unscorable reasons;
* prediction distribution: mean P(actionable), predicted-actionable share,
  intent mix;
* input shift against a reference: the population stability index (PSI) of
  P(actionable) and of text length, and the platform mix;
* alerts when PSI > 0.2 or coverage < 95%.

The default reference is the model's out-of-fold probabilities on the
labelled corpus. That corpus is rule-stratified, so the reference itself is
not population-representative: a PSI alert means "this batch looks different
from what the model was evaluated on", not "the model is wrong". Pass
``--reference`` with a previous scored batch for batch-to-batch monitoring.

Outputs describe public discussion. P(actionable) is the probability that a
comment expresses an actionable behavioural signal; it is not a purchase
probability.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.probabilities import PROBABILITY_COLUMN, PROVENANCE_COLUMN, build_probabilities

DEFAULT_REFERENCE = Path("reports/ml/group_cv/oof_predictions.csv")
PSI_ALERT = 0.20
COVERAGE_ALERT = 0.95


def population_stability_index(reference, current, bins: int = 10) -> float | None:
    """PSI of ``current`` against quantile bins of ``reference``."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < bins or len(current) == 0:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    eps = 1e-4
    ref_share, cur_share = np.clip(ref_share, eps, None), np.clip(cur_share, eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _reference_frame(path: Path) -> tuple[pd.DataFrame | None, str]:
    path = Path(path)
    if not path.exists():
        return None, f"no reference at {path}"
    frame = pd.read_csv(path)
    if "model" in frame.columns:  # group-CV OOF file: one model, one seed
        frame = frame[(frame["model"] == "logistic") & (frame["seed"] == frame["seed"].min())]
        frame = frame.rename(columns={"p_actionable": PROBABILITY_COLUMN})
        from src.ml.data import load_labelled_data

        texts = load_labelled_data().set_index("record_id")["text"]
        frame["text"] = frame["record_id"].map(texts)
        return frame, f"{path} (logistic out-of-fold, labelled corpus; rule-stratified)"
    return frame, str(path)


def monitoring_summary(scored: pd.DataFrame, reference: pd.DataFrame | None, reference_name: str) -> dict:
    probabilities = pd.to_numeric(scored[PROBABILITY_COLUMN], errors="coerce")
    scorable = probabilities.notna()
    words = scored["text"].fillna("").astype(str).str.split().map(len)
    summary: dict = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": int(len(scored)),
        "scored_rows": int(scorable.sum()),
        "coverage": float(scorable.mean()) if len(scored) else 0.0,
        "probability_source_counts": scored[PROVENANCE_COLUMN].value_counts().to_dict(),
        "mean_p_actionable": float(probabilities.mean()) if scorable.any() else None,
        "predicted_actionable_share": float(scored.loc[scorable, "is_actionable"].mean())
        if scorable.any() else None,
        "intent_mix": scored.loc[scorable, "predicted_intent"].value_counts(normalize=True).round(4).to_dict(),
        "source_mix": scored["source"].value_counts(normalize=True).round(4).to_dict()
        if "source" in scored else {},
        "median_words": float(words.median()) if len(words) else None,
        "reference": reference_name,
        "alerts": [],
    }
    if reference is not None:
        ref_p = pd.to_numeric(reference.get(PROBABILITY_COLUMN), errors="coerce")
        ref_words = reference.get("text", pd.Series(dtype=str)).fillna("").astype(str).str.split().map(len)
        summary["psi_p_actionable"] = population_stability_index(ref_p, probabilities[scorable])
        summary["psi_text_length"] = population_stability_index(ref_words, words)
        summary["reference_mean_p_actionable"] = float(ref_p.mean())
        for key in ("psi_p_actionable", "psi_text_length"):
            value = summary.get(key)
            if value is not None and value > PSI_ALERT:
                summary["alerts"].append(
                    f"{key} = {value:.2f} > {PSI_ALERT}: this batch differs from the reference"
                )
    if summary["coverage"] < COVERAGE_ALERT:
        summary["alerts"].append(
            f"coverage {summary['coverage']:.1%} < {COVERAGE_ALERT:.0%}: rows left unscored"
        )
    return summary


def score_batch(
    signals: pd.DataFrame,
    reference_path: Path = DEFAULT_REFERENCE,
    **probability_kwargs,
) -> tuple[pd.DataFrame, dict]:
    from src.ml.probabilities import ensure_hierarchical_model

    scored, provenance = build_probabilities(signals, **probability_kwargs)
    model, model_meta = ensure_hierarchical_model(
        **{k: v for k, v in probability_kwargs.items() if k in ("artifact_dir", "seed", "model_name")}
    )
    texts = scored["text"].fillna("").astype(str)
    has_text = texts.str.strip() != ""
    scored["predicted_intent"] = pd.NA
    scored.loc[has_text, "predicted_intent"] = model.predict(texts[has_text].tolist())
    scored["is_actionable"] = pd.to_numeric(scored[PROBABILITY_COLUMN], errors="coerce") > model.threshold
    scored["model_name"] = model_meta.get("model_name")
    scored["model_dataset_hash"] = model_meta.get("dataset_hash")
    scored["model_threshold"] = float(model.threshold)

    reference, reference_name = _reference_frame(reference_path)
    summary = monitoring_summary(scored, reference, reference_name)
    summary["probability_provenance"] = provenance
    summary["notice"] = (
        "P(actionable) is the probability that a public comment expresses an actionable "
        "behavioural signal. It is not a purchase probability, and the mean is not sales."
    )
    return scored, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score a signal file with monitoring")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--no-oof", action="store_true",
                        help="Skip out-of-fold scoring of labelled rows (faster). They are then "
                             "marked excluded_training_row, never scored in-sample")
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    signals = pd.read_csv(args.input)
    if "text" not in signals.columns:
        raise SystemExit(f"{args.input} has no 'text' column")
    scored, summary = score_batch(signals, reference_path=args.reference, use_oof=not args.no_oof)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in signals.columns if c not in {"metadata"}] + [
        PROBABILITY_COLUMN, PROVENANCE_COLUMN, "is_actionable", "predicted_intent",
        "model_name", "model_dataset_hash", "model_threshold",
    ]
    scored[[c for c in dict.fromkeys(keep) if c in scored.columns]].to_csv(args.output, index=False)
    monitoring_path = args.output.with_suffix(".monitoring.json")
    monitoring_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Scored {summary['scored_rows']}/{summary['rows']} rows -> {args.output}")
    print(f"Monitoring -> {monitoring_path}")
    for alert in summary["alerts"]:
        print(f"ALERT: {alert}")


if __name__ == "__main__":
    main()

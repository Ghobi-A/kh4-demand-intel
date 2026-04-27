"""Demand and risk scoring pipeline for KH4 signals."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/processed/signals_intent.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/signals_scored.csv")
DEFAULT_VIDEO_OUTPUT_PATH = Path("reports/tables/video_scores.csv")

INTENT_WEIGHTS: dict[str, float] = {
    "high_intent": 2.0,
    "nostalgia_reactivation": 1.5,
    "new_customer_interest": 1.2,
    "frustrated_demand": 1.0,
    "confusion_barrier": -1.0,
    "general_discussion": 0.1,
}

SENTIMENT_WEIGHTS: dict[str, float] = {
    "positive": 1.0,
    "neutral": 0.3,
    "negative": -0.5,
}

RISK_INTENTS = {"frustrated_demand", "confusion_barrier"}
ACTIVATION_INTENTS = {
    "high_intent",
    "nostalgia_reactivation",
    "new_customer_interest",
}


def _dominant_intent(series: pd.Series) -> str:
    """Return the most frequent intent label in a group."""
    if series.empty:
        return ""

    counts = series.astype(str).value_counts()
    max_count = counts.max()
    candidates = sorted(counts[counts == max_count].index.tolist())
    return candidates[0]


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return row-level demand/risk/activation scoring columns."""
    required_columns = {"intent_label", "sentiment_label", "engagement"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns {sorted(required_columns)}; missing {sorted(missing)}")

    scored = df.copy()

    scored["intent_weight"] = scored["intent_label"].map(INTENT_WEIGHTS).fillna(0.0)
    scored["sentiment_weight"] = scored["sentiment_label"].map(SENTIMENT_WEIGHTS).fillna(0.0)
    scored["engagement_weight"] = np.log1p(
        pd.to_numeric(scored["engagement"], errors="coerce").fillna(0.0)
    )

    scored["demand_score"] = (
        scored["intent_weight"] + scored["sentiment_weight"] + scored["engagement_weight"]
    )

    scored["risk_score"] = scored["intent_label"].isin(RISK_INTENTS).astype(int)
    scored["activation_score"] = scored["intent_label"].isin(ACTIVATION_INTENTS).astype(int)

    return scored


def aggregate_video_scores(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level scores to video-level metrics by parent_id."""
    if "parent_id" not in scored_df.columns:
        raise ValueError("Expected column 'parent_id' in scored data")

    grouped = scored_df.groupby("parent_id", dropna=False)

    summary = grouped.agg(
        total_signals=("parent_id", "size"),
        mean_demand_score=("demand_score", "mean"),
        total_risk_score=("risk_score", "sum"),
        total_activation_score=("activation_score", "sum"),
        positive_share=("sentiment_label", lambda s: (s == "positive").mean()),
        negative_share=("sentiment_label", lambda s: (s == "negative").mean()),
    )

    dominant = grouped["intent_label"].agg(_dominant_intent).rename("dominant_intent")

    result = summary.join(dominant).reset_index()
    ordered_columns = [
        "parent_id",
        "total_signals",
        "mean_demand_score",
        "total_risk_score",
        "total_activation_score",
        "dominant_intent",
        "positive_share",
        "negative_share",
    ]
    return result[ordered_columns]


def run_scoring_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    video_output_path: Path = DEFAULT_VIDEO_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run row-level scoring and video-level aggregation outputs."""
    df = pd.read_csv(input_path)

    scored_df = score_dataframe(df)
    video_scores = aggregate_video_scores(scored_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_output_path.parent.mkdir(parents=True, exist_ok=True)

    scored_df.to_csv(output_path, index=False)
    video_scores.to_csv(video_output_path, index=False)

    LOGGER.info("Rows scored: %s", len(scored_df))
    LOGGER.info("Unique parent_id count: %s", video_scores["parent_id"].nunique())
    LOGGER.info("Wrote scored rows to %s", output_path)
    LOGGER.info("Wrote video-level scores to %s", video_output_path)

    return scored_df, video_scores


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score KH4 demand and risk signals")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--video-output", type=Path, default=DEFAULT_VIDEO_OUTPUT_PATH)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_parser().parse_args()
    run_scoring_pipeline(
        input_path=args.input,
        output_path=args.output,
        video_output_path=args.video_output,
    )


if __name__ == "__main__":
    main()

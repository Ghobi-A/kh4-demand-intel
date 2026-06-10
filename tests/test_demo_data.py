"""Tests for committed dashboard demo data."""

from pathlib import Path

import pandas as pd

from src.score import aggregate_video_scores, score_dataframe


SIGNALS_DEMO_PATH = Path("data/demo/signals_scored_demo.csv")
VIDEO_DEMO_PATH = Path("data/demo/video_scores_demo.csv")

REQUIRED_SIGNAL_COLUMNS = {
    "text",
    "source",
    "parent_id",
    "intent_label",
    "sentiment_label",
    "engagement",
    "intent_weight",
    "sentiment_weight",
    "engagement_weight",
    "demand_score",
    "risk_score",
    "activation_score",
}
REQUIRED_VIDEO_COLUMNS = {"parent_id", "mean_demand_score"}
KEY_INTENTS = {
    "high_intent",
    "nostalgia_reactivation",
    "new_customer_interest",
    "frustrated_demand",
    "content_drought_fatigue",
    "confusion_barrier",
    "expectation_decay",
    "general_discussion",
}
KEY_SENTIMENTS = {"positive", "neutral", "negative"}


def test_demo_data_is_committed_and_representative() -> None:
    signals = pd.read_csv(SIGNALS_DEMO_PATH)
    video_scores = pd.read_csv(VIDEO_DEMO_PATH)

    assert not signals.empty
    assert not video_scores.empty
    assert REQUIRED_SIGNAL_COLUMNS.issubset(signals.columns)
    assert REQUIRED_VIDEO_COLUMNS.issubset(video_scores.columns)
    assert KEY_INTENTS.issubset(set(signals["intent_label"]))
    assert KEY_SENTIMENTS.issubset(set(signals["sentiment_label"]))


def test_demo_scores_match_current_scoring_formula() -> None:
    signals = pd.read_csv(SIGNALS_DEMO_PATH)

    input_columns = [
        "text",
        "source",
        "parent_id",
        "intent_label",
        "sentiment_label",
        "engagement",
    ]
    rescored = score_dataframe(signals[input_columns])

    pd.testing.assert_series_equal(
        signals["engagement_weight"],
        rescored["engagement_weight"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        signals["demand_score"],
        rescored["demand_score"],
        check_names=False,
    )


def test_demo_video_scores_match_current_aggregation() -> None:
    signals = pd.read_csv(SIGNALS_DEMO_PATH)
    video_scores = pd.read_csv(VIDEO_DEMO_PATH).sort_values("parent_id").reset_index(drop=True)

    expected_video_scores = (
        aggregate_video_scores(signals)
        .sort_values("parent_id")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(video_scores, expected_video_scores)

"""Tests for demand/risk scoring and video-level aggregation."""

import math

import pandas as pd

from src.score import aggregate_video_scores, score_dataframe


def test_score_calculation() -> None:
    df = pd.DataFrame(
        {
            "parent_id": ["v1"],
            "intent_label": ["high_intent"],
            "sentiment_label": ["positive"],
            "engagement": [9],
        }
    )

    scored = score_dataframe(df)

    assert scored.loc[0, "intent_weight"] == 2.0
    assert scored.loc[0, "sentiment_weight"] == 1.0
    assert scored.loc[0, "engagement_weight"] == math.log1p(9)
    assert scored.loc[0, "demand_score"] == 2.0 + 1.0 + math.log1p(9)


def test_risk_score() -> None:
    df = pd.DataFrame(
        {
            "parent_id": ["v1", "v1", "v1"],
            "intent_label": ["frustrated_demand", "confusion_barrier", "high_intent"],
            "sentiment_label": ["negative", "neutral", "positive"],
            "engagement": [0, 0, 0],
        }
    )

    scored = score_dataframe(df)

    assert scored["risk_score"].tolist() == [1, 1, 0]


def test_activation_score() -> None:
    df = pd.DataFrame(
        {
            "parent_id": ["v1", "v1", "v1", "v1"],
            "intent_label": [
                "high_intent",
                "nostalgia_reactivation",
                "new_customer_interest",
                "general_discussion",
            ],
            "sentiment_label": ["positive", "positive", "neutral", "neutral"],
            "engagement": [0, 0, 0, 0],
        }
    )

    scored = score_dataframe(df)

    assert scored["activation_score"].tolist() == [1, 1, 1, 0]


def test_new_intent_weights_and_flags() -> None:
    df = pd.DataFrame(
        {
            "parent_id": ["v1", "v1"],
            "intent_label": ["expectation_decay", "content_drought_fatigue"],
            "sentiment_label": ["negative", "neutral"],
            "engagement": [0, 0],
        }
    )

    scored = score_dataframe(df)

    assert scored.loc[0, "intent_weight"] == -1.5
    assert scored.loc[1, "intent_weight"] == 0.6
    assert scored["risk_score"].tolist() == [1, 1]
    assert scored["activation_score"].tolist() == [0, 0]


def test_video_level_aggregation() -> None:
    df = pd.DataFrame(
        {
            "parent_id": ["v1", "v1", "v2"],
            "intent_label": ["high_intent", "frustrated_demand", "general_discussion"],
            "sentiment_label": ["positive", "negative", "neutral"],
            "engagement": [9, 0, 0],
        }
    )

    scored = score_dataframe(df)
    aggregated = aggregate_video_scores(scored)

    by_parent = aggregated.set_index("parent_id")

    assert by_parent.loc["v1", "total_signals"] == 2
    assert by_parent.loc["v1", "total_risk_score"] == 1
    assert by_parent.loc["v1", "total_activation_score"] == 1
    assert by_parent.loc["v1", "dominant_intent"] == "frustrated_demand"
    assert by_parent.loc["v1", "positive_share"] == 0.5
    assert by_parent.loc["v1", "negative_share"] == 0.5

    expected_v1_mean = (
        (2.0 + 1.0 + math.log1p(9))
        + (1.0 + (-0.5) + math.log1p(0))
    ) / 2
    assert by_parent.loc["v1", "mean_demand_score"] == expected_v1_mean

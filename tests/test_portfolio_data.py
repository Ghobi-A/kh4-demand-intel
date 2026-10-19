from pathlib import Path

import pandas as pd

SIGNALS_PATH = Path("data/demo/signals_scored_portfolio.csv")
VIDEO_SCORES_PATH = Path("data/demo/video_scores_portfolio.csv")

TAXONOMY_CLASSES = {
    "high_intent",
    "nostalgia_reactivation",
    "new_customer_interest",
    "frustrated_demand",
    "content_drought_fatigue",
    "confusion_barrier",
    "expectation_decay",
    "general_discussion",
}

EXPECTED_OBSERVED_CLASSES = {
    "high_intent",
    "nostalgia_reactivation",
    "new_customer_interest",
    "frustrated_demand",
    "confusion_barrier",
    "general_discussion",
}

REQUIRED_DASHBOARD_COLUMNS = {
    "intent_label",
    "sentiment_label",
    "demand_score",
    "activation_score",
    "risk_score",
    "parent_id",
}

REQUIRED_VIDEO_COLUMNS = {
    "parent_id",
    "mean_demand_score",
}


def test_portfolio_signals_csv_exists() -> None:
    assert SIGNALS_PATH.exists()


def test_portfolio_video_scores_csv_exists() -> None:
    assert VIDEO_SCORES_PATH.exists()


def test_portfolio_signals_are_dashboard_ready() -> None:
    signals_df = pd.read_csv(SIGNALS_PATH)
    observed_classes = set(signals_df["intent_label"].dropna().astype(str))

    assert len(signals_df) >= 300
    assert signals_df["parent_id"].nunique() >= 10
    assert observed_classes.issubset(TAXONOMY_CLASSES)
    assert EXPECTED_OBSERVED_CLASSES.issubset(observed_classes)
    assert len(observed_classes) >= 6
    assert not signals_df[
        ["demand_score", "risk_score", "activation_score"]
    ].isna().all().all()
    assert REQUIRED_DASHBOARD_COLUMNS.issubset(signals_df.columns)


def test_portfolio_video_scores_are_dashboard_ready() -> None:
    video_scores_df = pd.read_csv(VIDEO_SCORES_PATH)

    assert len(video_scores_df) >= 10
    assert REQUIRED_VIDEO_COLUMNS.issubset(video_scores_df.columns)


def test_portfolio_extract_carries_no_commenter_identifiers() -> None:
    """The public extract must not expose usernames, even nested in JSON metadata."""
    import json
    import sys

    sys.path.insert(0, "scripts")
    from build_portfolio_dataset import IDENTIFYING_COLUMNS, IDENTIFYING_METADATA_KEYS

    signals_df = pd.read_csv(SIGNALS_PATH)
    assert not IDENTIFYING_COLUMNS.intersection(signals_df.columns)
    if "metadata" in signals_df.columns:
        for value in signals_df["metadata"].dropna():
            payload = json.loads(value)
            assert not IDENTIFYING_METADATA_KEYS.intersection(payload), payload


def test_metadata_stripping_removes_nested_author_fields() -> None:
    import sys

    sys.path.insert(0, "scripts")
    from build_portfolio_dataset import _strip_identifying_columns

    frame = pd.DataFrame(
        {
            "author": ["someone"],
            "text": ["hi"],
            "metadata": ['{"video_id": "abc", "author_display_name": "@someone"}'],
        }
    )
    cleaned = _strip_identifying_columns(frame)
    assert "author" not in cleaned.columns
    assert "someone" not in cleaned["metadata"].iloc[0]
    assert "abc" in cleaned["metadata"].iloc[0]

from pathlib import Path

import pandas as pd

SIGNALS_PATH = Path("data/demo/signals_scored_portfolio.csv")
VIDEO_SCORES_PATH = Path("data/demo/video_scores_portfolio.csv")
INTENT_CLASSES = {
    "high_intent",
    "nostalgia_reactivation",
    "new_customer_interest",
    "frustrated_demand",
    "content_drought_fatigue",
    "confusion_barrier",
    "expectation_decay",
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


def test_portfolio_signals_csv_exists() -> None:
    assert SIGNALS_PATH.exists()


def test_portfolio_video_scores_csv_exists() -> None:
    assert VIDEO_SCORES_PATH.exists()


def test_portfolio_signals_are_dashboard_ready() -> None:
    signals_df = pd.read_csv(SIGNALS_PATH)

    assert len(signals_df) >= 300
    assert signals_df["parent_id"].nunique() >= 10
    assert INTENT_CLASSES.issubset(set(signals_df["intent_label"].dropna()))
    assert not signals_df[["demand_score", "risk_score", "activation_score"]].isna().all().all()
    assert REQUIRED_DASHBOARD_COLUMNS.issubset(signals_df.columns)

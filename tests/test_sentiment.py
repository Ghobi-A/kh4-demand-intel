"""Tests for VADER sentiment scoring."""

import pandas as pd

from src.sentiment import score_dataframe


def test_score_dataframe_adds_vader_and_labels() -> None:
    df = pd.DataFrame(
        {
            "text": [
                "This is amazing, I love it",
                "This is terrible, I hate it",
                "The game has nine worlds",
            ]
        }
    )

    scored = score_dataframe(df)

    required_cols = {
        "vader_neg",
        "vader_neu",
        "vader_pos",
        "vader_compound",
        "sentiment_label",
    }
    assert required_cols.issubset(set(scored.columns))
    assert scored.loc[0, "sentiment_label"] == "positive"
    assert scored.loc[1, "sentiment_label"] == "negative"

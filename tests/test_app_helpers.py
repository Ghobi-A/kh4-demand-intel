"""Tests for lightweight dashboard helper functions."""

import importlib
import sys
from types import SimpleNamespace

import pandas as pd


def _import_app_with_streamlit_stub():
    """Import app helpers without requiring a Streamlit runtime in tests."""
    sys.modules.setdefault(
        "streamlit",
        SimpleNamespace(cache_data=lambda function: function),
    )
    return importlib.import_module("app")


def test_sentiment_share_helpers() -> None:
    app = _import_app_with_streamlit_stub()
    df = pd.DataFrame(
        {
            "sentiment_label": ["positive", "negative", "positive", "neutral"],
        }
    )

    assert app.positive_share(df) == 0.5
    assert app.negative_share(df) == 0.25


def test_filter_dataframe_applies_selected_values() -> None:
    app = _import_app_with_streamlit_stub()
    df = pd.DataFrame(
        {
            "source": ["reddit", "youtube", "reddit"],
            "intent_label": ["high_intent", "general_discussion", "high_intent"],
            "demand_score": [3.5, 0.4, 2.1],
        }
    )

    filtered = app.filter_dataframe(
        df,
        {
            "source": ["reddit"],
            "intent_label": ["high_intent"],
        },
    )

    assert filtered["demand_score"].tolist() == [3.5, 2.1]

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


def test_overview_data_reads_generated_artefacts() -> None:
    app = _import_app_with_streamlit_stub()
    data = app.overview_data()

    assert data["labelled_rows"] == 200
    assert data["forecast_sample_only"] is True  # committed forecast is a demo run
    assert data["sentiment_vs_intent"]["positive"] > 0
    assert data["ranking"]["pr_auc"] > data["ranking"]["base_rate"]
    passed, total = data["mmm_checks"]
    assert 0 < passed <= total


def test_display_columns_hide_raw_metadata() -> None:
    app = _import_app_with_streamlit_stub()
    frame = pd.DataFrame({"text": ["x"], "metadata": ["{}"], "author": ["a"]})
    assert list(app.display_columns(frame).columns) == ["text"]


def test_overview_reports_the_rules_comparison() -> None:
    app = _import_app_with_streamlit_stub()
    ranking = app.overview_data()["ranking"]
    assert ranking["supervised_compared_with_rules"] >= 4
    assert ranking["supervised_beating_rules"] == 0  # committed comparison: none supported

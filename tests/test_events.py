"""Tests for event-relative annotation and window-level sentiment deltas."""

import math

import pandas as pd
import pytest

from src.events import (
    PRE_EVENT_LABEL,
    aggregate_event_deltas,
    annotate_events_dataframe,
    load_events,
    run_events_pipeline,
)


def _write_events_csv(path, rows: list[str]) -> None:
    header = "event_date,event_name,event_type,platform,notes"
    path.write_text("\n".join([header, *rows]) + "\n")


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-06-09", "2026-08-15"], utc=True),
            "event_name": ["KH4 Nintendo Direct trailer", "D23 2026 Showcase KH4 trailer"],
            "event_window": [
                "post_kh4_nintendo_direct_trailer",
                "post_d23_2026_showcase_kh4_trailer",
            ],
        }
    )


def test_load_events_valid(tmp_path) -> None:
    path = tmp_path / "events.csv"
    _write_events_csv(
        path,
        [
            "2026-08-15,D23 2026 Showcase KH4 trailer,trailer,D23,note",
            "2026-06-09,KH4 Nintendo Direct trailer,trailer,Nintendo Direct,note",
        ],
    )

    events = load_events(path)

    assert events["event_name"].tolist() == [
        "KH4 Nintendo Direct trailer",
        "D23 2026 Showcase KH4 trailer",
    ]
    assert events["event_window"].tolist() == [
        "post_kh4_nintendo_direct_trailer",
        "post_d23_2026_showcase_kh4_trailer",
    ]


def test_load_events_missing_column_raises(tmp_path) -> None:
    path = tmp_path / "events.csv"
    path.write_text("event_date,notes\n2026-06-09,x\n")

    with pytest.raises(ValueError):
        load_events(path)


def test_load_events_drops_bad_dates_and_raises_when_empty(tmp_path) -> None:
    path = tmp_path / "events.csv"
    _write_events_csv(path, ["not-a-date,Bad event,trailer,X,note"])

    with pytest.raises(ValueError):
        load_events(path)


def test_annotate_windows_and_days_since_event() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-05-01T00:00:00Z",
                "2026-07-01T00:00:00Z",
                "2026-08-20T00:00:00Z",
            ],
            "sentiment_label": ["neutral", "positive", "positive"],
        }
    )

    annotated = annotate_events_dataframe(df, _sample_events())

    assert annotated["event_window"].tolist() == [
        PRE_EVENT_LABEL,
        "post_kh4_nintendo_direct_trailer",
        "post_d23_2026_showcase_kh4_trailer",
    ]
    assert math.isnan(annotated.loc[0, "days_since_event"])
    assert annotated.loc[1, "days_since_event"] == 22.0
    assert annotated.loc[2, "days_since_event"] == 5.0
    assert annotated.loc[0, "last_event_name"] == PRE_EVENT_LABEL


def test_annotate_missing_timestamp_raises() -> None:
    with pytest.raises(ValueError):
        annotate_events_dataframe(pd.DataFrame({"text": ["hi"]}), _sample_events())


def test_annotate_handles_unsorted_input_rows() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2026-08-20T00:00:00Z", "2026-05-01T00:00:00Z"],
            "sentiment_label": ["positive", "neutral"],
        }
    )

    annotated = annotate_events_dataframe(df, _sample_events())

    assert annotated["event_window"].tolist() == [
        "post_d23_2026_showcase_kh4_trailer",
        PRE_EVENT_LABEL,
    ]


def test_aggregate_event_deltas_ordering_and_deltas() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-05-01T00:00:00Z",
                "2026-07-01T00:00:00Z",
                "2026-08-20T00:00:00Z",
                "2026-08-21T00:00:00Z",
            ],
            "sentiment_label": ["negative", "positive", "positive", "neutral"],
            "vader_compound": [-0.5, 0.4, 0.8, 0.0],
        }
    )
    annotated = annotate_events_dataframe(df, _sample_events())

    deltas = aggregate_event_deltas(annotated)

    assert deltas["event_window"].tolist() == [
        PRE_EVENT_LABEL,
        "post_kh4_nintendo_direct_trailer",
        "post_d23_2026_showcase_kh4_trailer",
    ]
    assert deltas["n_comments"].tolist() == [1, 1, 2]
    assert math.isnan(deltas.loc[0, "delta_vs_prev_window"])
    assert deltas.loc[1, "delta_vs_prev_window"] == pytest.approx(0.9)
    assert deltas.loc[2, "delta_vs_prev_window"] == pytest.approx(0.4 - 0.4)
    assert deltas.loc[2, "positive_share"] == 0.5


def test_run_events_pipeline_end_to_end(tmp_path) -> None:
    events_path = tmp_path / "events.csv"
    _write_events_csv(
        events_path,
        [
            "2026-06-09,KH4 Nintendo Direct trailer,trailer,Nintendo Direct,note",
            "2026-08-15,D23 2026 Showcase KH4 trailer,trailer,D23,note",
        ],
    )
    input_path = tmp_path / "signals_scored.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-05-01T00:00:00Z", "2026-08-20T00:00:00Z"],
            "sentiment_label": ["neutral", "positive"],
            "vader_compound": [0.0, 0.7],
            "demand_score": [0.1, 2.0],
        }
    ).to_csv(input_path, index=False)

    output_path = tmp_path / "signals_events.csv"
    delta_output_path = tmp_path / "event_sentiment_deltas.csv"

    annotated, deltas = run_events_pipeline(
        input_path=input_path,
        output_path=output_path,
        events_path=events_path,
        delta_output_path=delta_output_path,
    )

    assert output_path.exists()
    assert delta_output_path.exists()
    saved = pd.read_csv(output_path)
    for column in ["event_window", "days_since_event", "last_event_name"]:
        assert column in saved.columns
    saved_deltas = pd.read_csv(delta_output_path)
    for column in ["event_window", "n_comments", "mean_vader_compound", "delta_vs_prev_window"]:
        assert column in saved_deltas.columns
    assert len(annotated) == 2
    assert len(deltas) == 2

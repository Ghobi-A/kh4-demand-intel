"""Tests for weekly aggregation and the behavioural demand proxy."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.events import event_features_for_periods, load_events
from src.ml.probabilities import PROBABILITY_COLUMN, PROVENANCE_COLUMN, SOURCE_UNAVAILABLE
from src.temporal import aggregate_weekly, build_temporal_dataset


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-05T10:00:00Z",
                "2026-01-07T10:00:00Z",
                "2026-01-11T10:00:00Z",
                # week of 2026-01-12 is empty
                "2026-01-19T10:00:00Z",
                None,
            ],
            "source": ["youtube", "youtube", "reddit", "youtube", "youtube"],
            "text": ["a", "b", "c", "d", "e"],
            "intent_label": [
                "high_intent",
                "general_discussion",
                "frustrated_demand",
                "general_discussion",
                "high_intent",
            ],
            "vader_compound": [0.5, 0.0, -0.4, 0.2, 0.9],
            "demand_score": [3.0, 0.1, 1.0, 0.1, 3.0],
            PROBABILITY_COLUMN: [0.9, 0.2, 0.7, 0.1, 0.8],
            PROVENANCE_COLUMN: [
                "oof_supervised",
                "oof_supervised",
                "persisted_supervised",
                "persisted_supervised",
                "persisted_supervised",
            ],
        }
    )


def test_weekly_bins_start_on_monday_and_have_no_gaps() -> None:
    weekly, meta = aggregate_weekly(_signals())

    assert weekly["period_start"].dt.dayofweek.unique().tolist() == [0]
    assert weekly["period_start"].tolist() == [
        pd.Timestamp("2026-01-05T00:00:00Z"),
        pd.Timestamp("2026-01-12T00:00:00Z"),
        pd.Timestamp("2026-01-19T00:00:00Z"),
    ]
    assert meta["periods"] == 3
    assert meta["non_empty_periods"] == 2


def test_rows_without_timestamps_are_excluded_and_reported() -> None:
    weekly, meta = aggregate_weekly(_signals())

    assert meta["rows_in"] == 5
    assert meta["rows_aggregated"] == 4
    assert meta["rows_without_timestamp"] == 1
    assert weekly["total_comments"].sum() == 4


def test_empty_week_keeps_zero_counts_but_undefined_rates() -> None:
    weekly, _ = aggregate_weekly(_signals())
    empty_week = weekly.loc[weekly["period_start"] == pd.Timestamp("2026-01-12T00:00:00Z")].iloc[0]

    assert empty_week["total_comments"] == 0
    assert empty_week["actionable_count"] == 0
    # A week with no comments observed nothing about the actionable rate;
    # recording 0% would invent an observation.
    assert np.isnan(empty_week["actionable_rate"])


def test_demand_proxy_is_the_sum_of_probabilities_in_the_period() -> None:
    weekly, _ = aggregate_weekly(_signals())
    first = weekly.iloc[0]

    assert first["total_comments"] == 3
    assert first["actionable_probability_mass"] == pytest.approx(0.9 + 0.2 + 0.7)
    assert first["mean_actionable_probability"] == pytest.approx((0.9 + 0.2 + 0.7) / 3)
    assert first["n_with_probability"] == 3
    assert first["probability_coverage"] == pytest.approx(1.0)


def test_unavailable_probabilities_are_left_out_of_the_proxy() -> None:
    signals = _signals()
    signals.loc[0, PROVENANCE_COLUMN] = SOURCE_UNAVAILABLE
    signals.loc[0, PROBABILITY_COLUMN] = np.nan

    weekly, _ = aggregate_weekly(signals)
    first = weekly.iloc[0]

    assert first["actionable_probability_mass"] == pytest.approx(0.2 + 0.7)
    assert first["n_with_probability"] == 2
    assert first["probability_coverage"] == pytest.approx(2 / 3)


def test_counts_split_by_platform_and_intent() -> None:
    weekly, _ = aggregate_weekly(_signals())
    first = weekly.iloc[0]

    assert first["youtube_comments"] == 2
    assert first["reddit_comments"] == 1
    assert first["actionable_count"] == 2
    assert first["high_intent_count"] == 1
    assert first["actionable_rate"] == pytest.approx(2 / 3)


def test_aggregation_without_probability_columns_keeps_count_targets() -> None:
    signals = _signals().drop(columns=[PROBABILITY_COLUMN, PROVENANCE_COLUMN])

    weekly, _ = aggregate_weekly(signals)

    assert "actionable_probability_mass" not in weekly.columns
    assert weekly["actionable_count"].sum() == 2
    assert weekly["total_comments"].sum() == 4


def test_event_features_attach_and_decay_after_the_event() -> None:
    events = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-12"], utc=True),
            "announced_date": pd.to_datetime(["2026-01-12"], utc=True),
            "event_name": ["Trailer"],
            "event_window": ["post_trailer"],
            "event_type": ["trailer"],
            "major_announcement_flag": [1],
            "trailer_flag": [1],
        }
    )

    weekly, _ = aggregate_weekly(_signals(), events=events)

    assert weekly.loc[0, "event_window"] == "pre_announcement"
    assert weekly.loc[0, "event_active"] == 0
    assert weekly.loc[1, "event_active"] == 1
    assert weekly.loc[1, "trailer_flag"] == 1
    # Influence decays as the event recedes.
    assert weekly.loc[2, "event_decay"] < weekly.loc[1, "event_decay"]
    assert (weekly["event_annotation"] == "retrospective").all()


def test_event_features_hide_events_not_yet_announced_at_the_origin() -> None:
    events = load_events()
    periods = pd.date_range("2026-05-25", "2026-08-31", freq="7D", tz="UTC")

    origin = pd.Timestamp("2026-06-20", tz="UTC")
    limited = event_features_for_periods(periods, events, origin=origin)
    full = event_features_for_periods(periods, events)

    # D23 (2026-08-15) was not public at a 2026-06-20 origin, so no period may
    # reference it; the retrospective view does.
    assert "post_d23_2026_showcase_kh4_trailer" not in set(limited["event_window"])
    assert "post_d23_2026_showcase_kh4_trailer" in set(full["event_window"])
    assert limited["event_known_at_origin"].all()


def test_build_temporal_dataset_demo_stamps_outputs(tmp_path, monkeypatch) -> None:
    signals = _signals()
    signals["id"] = ["a", "b", "c", "d", "e"]
    input_path = tmp_path / "signals.csv"
    signals.to_csv(input_path, index=False)
    output_path = tmp_path / "weekly.csv"

    def _fake_probabilities(df, **kwargs):
        return df, {"probability_generation": "stub", "oof_rows": 0}

    monkeypatch.setattr("src.temporal.build_probabilities", _fake_probabilities)

    weekly, meta = build_temporal_dataset(
        input_path=input_path, output_path=output_path, demo=True
    )

    assert output_path.exists()
    assert meta["sample_only"] is True
    assert meta["volume_representative"] is False
    assert meta["evaluation_status"] == "demo_only"
    assert "not volume-representative" in meta["notice"]

    saved_meta = json.loads((output_path.with_suffix(".meta.json")).read_text())
    assert saved_meta["evaluation_status"] == "demo_only"
    assert "not expected purchases or sales" in saved_meta["primary_target_definition"]
    assert len(weekly) == 3


def test_full_dataset_run_is_not_marked_as_a_sample(tmp_path, monkeypatch) -> None:
    signals = _signals()
    input_path = tmp_path / "signals.csv"
    signals.to_csv(input_path, index=False)

    monkeypatch.setattr(
        "src.temporal.build_probabilities", lambda df, **kwargs: (df, {"oof_rows": 0})
    )

    _, meta = build_temporal_dataset(
        input_path=input_path, output_path=tmp_path / "weekly.csv", demo=False
    )

    assert meta["sample_only"] is False
    assert meta["volume_representative"] is True
    assert meta["evaluation_status"] == "full_dataset"

"""Tests for canonical timestamp parsing, serialisation, and coverage audit."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.timestamps import (
    STATUS_INVALID,
    STATUS_MISSING,
    STATUS_VALID,
    audit_temporal_coverage,
    complexity_tier,
    parse_timestamps,
    period_range,
    period_start,
    serialise_timestamps,
    timestamp_status,
)


def test_parses_iso_and_epoch_values() -> None:
    parsed, report = parse_timestamps(["2026-01-01T12:00:00Z", 1700000000])

    assert parsed.dt.tz is not None
    assert parsed.iloc[0] == pd.Timestamp("2026-01-01T12:00:00Z")
    assert parsed.iloc[1] == pd.Timestamp(1700000000, unit="s", tz="UTC")
    assert report.valid == 2
    assert report.invalid == 0


def test_missing_values_stay_missing_and_are_counted() -> None:
    parsed, report = parse_timestamps([None, "", "  ", "nan"])

    assert parsed.isna().all()
    assert report.missing == 4
    assert report.invalid == 0


def test_epoch_zero_is_treated_as_fabricated_not_as_1970_data() -> None:
    parsed, report = parse_timestamps([0, "0"])

    assert parsed.isna().all()
    assert report.epoch_zero_fabricated == 2
    assert report.invalid == 2
    assert report.valid == 0


def test_unparseable_values_are_invalid_not_silently_dropped() -> None:
    parsed, report = parse_timestamps(["not-a-date"])

    assert parsed.isna().all()
    assert report.invalid == 1
    assert report.missing == 0


def test_timezone_naive_rejected_unless_explicitly_assumed_utc() -> None:
    rejected, report = parse_timestamps(["2026-01-05 10:00:00"])
    assert rejected.isna().all()
    assert report.naive_rejected == 1

    accepted, ok_report = parse_timestamps(["2026-01-05 10:00:00"], assume_utc=True)
    assert accepted.iloc[0] == pd.Timestamp("2026-01-05T10:00:00Z")
    assert ok_report.valid == 1


def test_future_timestamps_are_rejected() -> None:
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=30)).isoformat()
    parsed, report = parse_timestamps([future])

    assert parsed.isna().all()
    assert report.future == 1


def test_status_separates_missing_from_invalid() -> None:
    raw = pd.Series(["2026-01-01T00:00:00Z", None, "garbage"])
    parsed, _ = parse_timestamps(raw)

    status = timestamp_status(parsed, raw)

    assert status.tolist() == [STATUS_VALID, STATUS_MISSING, STATUS_INVALID]


def test_serialisation_is_deterministic_and_round_trips() -> None:
    parsed, _ = parse_timestamps(["2026-01-01T12:00:00Z", None])

    serialised = serialise_timestamps(parsed)

    assert serialised.tolist() == ["2026-01-01T12:00:00+00:00", ""]
    reparsed, report = parse_timestamps(serialised)
    assert reparsed.iloc[0] == parsed.iloc[0]
    assert report.missing == 1
    assert serialise_timestamps(reparsed).tolist() == serialised.tolist()


def test_period_start_buckets_to_monday_and_keeps_timezone() -> None:
    values = pd.to_datetime(
        ["2026-01-01T23:00:00Z", "2026-01-05T00:00:00Z", "2026-01-11T23:59:00Z"], utc=True
    )

    starts = period_start(pd.Series(values))

    assert starts.dt.tz is not None
    assert starts.dt.dayofweek.tolist() == [0, 0, 0]
    assert starts.tolist() == [
        pd.Timestamp("2025-12-29T00:00:00Z"),
        pd.Timestamp("2026-01-05T00:00:00Z"),
        pd.Timestamp("2026-01-05T00:00:00Z"),
    ]


def test_period_range_is_gap_free() -> None:
    index = period_range(pd.Timestamp("2026-01-05", tz="UTC"), pd.Timestamp("2026-02-02", tz="UTC"))

    assert len(index) == 5
    assert (index.to_series().diff().dropna() == pd.Timedelta(days=7)).all()


def test_coverage_audit_reports_gaps_and_event_usability() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-05T00:00:00Z",
                "2026-01-06T00:00:00Z",
                # four-week gap here
                "2026-02-09T00:00:00Z",
                None,
            ],
            "source": ["youtube", "youtube", "reddit", "youtube"],
        }
    )
    events = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-19"], utc=True),
            "event_name": ["Mid-window event"],
        }
    )

    audit = audit_temporal_coverage(df, events=events, min_event_periods=1)

    assert audit.rows_with_timestamp == 3
    assert audit.rows_without_timestamp == 1
    assert audit.non_empty_periods == 2
    assert audit.observed_periods == 6
    assert audit.source_coverage == {"youtube": 1, "reddit": 1}
    assert audit.gaps and audit.gaps[0]["periods"] == 4
    assert audit.usable_event_periods == 1


@pytest.mark.parametrize(
    ("periods", "expected"),
    [(0, "minimal"), (29, "minimal"), (30, "moderate"), (60, "moderate"), (61, "rich")],
)
def test_complexity_tier_follows_measured_history(periods: int, expected: str) -> None:
    assert complexity_tier(periods) == expected

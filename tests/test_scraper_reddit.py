"""Tests for Reddit ingestion timestamp handling."""

from __future__ import annotations

import datetime as dt

from src.scraper_reddit import _created_utc_to_datetime


def test_valid_epoch_becomes_utc() -> None:
    parsed = _created_utc_to_datetime(1700000000)

    assert parsed is not None
    assert parsed.tzinfo is dt.timezone.utc
    assert parsed.year == 2023


def test_missing_created_utc_stays_missing() -> None:
    """A missing created_utc must not default to 0 and become a 1970 timestamp."""
    assert _created_utc_to_datetime(None) is None
    assert _created_utc_to_datetime("") is None
    assert _created_utc_to_datetime("not-a-number") is None


def test_epoch_zero_is_not_treated_as_real_data() -> None:
    assert _created_utc_to_datetime(0) is None
    assert _created_utc_to_datetime(-5) is None

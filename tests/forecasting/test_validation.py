"""Tests for rolling-origin validation and temporal leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.baselines import NaiveForecaster
from src.forecasting.data import ExogBuilder
from src.forecasting.validation import RollingOriginSplit, backtest, rolling_origin_splits


def test_splits_walk_forward_with_the_configured_geometry() -> None:
    splits = rolling_origin_splits(20, min_train=12, horizon=2, step=2)

    assert [split.origin for split in splits] == [12, 14, 16, 18]
    assert splits[0].train_index.tolist() == list(range(12))
    assert splits[0].test_index.tolist() == [12, 13]
    assert splits[-1].test_index.tolist() == [18, 19]


def test_no_split_trains_on_data_from_after_the_forecast_period() -> None:
    for split in rolling_origin_splits(40, min_train=10, horizon=3, step=5):
        assert split.train_index.max() < split.test_index.min()
        assert split.train_index.min() == 0


def test_a_split_that_trains_on_the_future_is_rejected() -> None:
    with pytest.raises(ValueError):
        RollingOriginSplit(
            origin=5, train_index=np.arange(0, 10), test_index=np.arange(5, 7)
        )


def test_configuration_is_honoured_and_validated() -> None:
    assert len(rolling_origin_splits(30, min_train=20, horizon=5, step=5)) == 2
    assert rolling_origin_splits(10, min_train=12, horizon=2) == []
    assert len(rolling_origin_splits(40, min_train=10, horizon=2, step=2, max_origins=3)) == 3

    for bad in [{"min_train": 0}, {"horizon": 0}, {"step": 0}]:
        with pytest.raises(ValueError):
            rolling_origin_splits(20, **bad)


def test_backtest_scores_each_origin_against_its_own_future() -> None:
    y = pd.Series(np.arange(20, dtype=float))
    splits = rolling_origin_splits(20, min_train=10, horizon=2, step=2)

    results = backtest(NaiveForecaster, y, splits, model_name="naive")

    assert len(results) == len(splits) * 2
    assert (results["actual"] == results["period_index"].astype(float)).all()
    # A naive forecast repeats the last training value, which is origin - 1.
    assert (results["forecast"] == (results["origin"] - 1).astype(float)).all()


def test_future_values_cannot_change_an_earlier_origin_forecast() -> None:
    """The core leakage check: mutating later weeks must not move earlier forecasts."""
    y = pd.Series(np.arange(30, dtype=float))
    splits = rolling_origin_splits(30, min_train=10, horizon=2, step=2)
    baseline = backtest(NaiveForecaster, y, splits, model_name="naive")

    tampered = y.copy()
    tampered.iloc[20:] = 999.0
    after = backtest(NaiveForecaster, tampered, splits, model_name="naive")

    early = baseline[baseline["origin"] <= 18][["origin", "horizon_step", "forecast"]]
    early_after = after[after["origin"] <= 18][["origin", "horizon_step", "forecast"]]
    pd.testing.assert_frame_equal(early.reset_index(drop=True), early_after.reset_index(drop=True))


def test_exog_builder_never_reveals_periods_after_the_origin() -> None:
    periods = pd.Series(pd.date_range("2026-01-05", periods=20, freq="7D", tz="UTC"))
    events = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-03-02"], utc=True),
            # Announced only one day before it happened.
            "announced_date": pd.to_datetime(["2026-03-01"], utc=True),
            "event_name": ["Surprise trailer"],
            "event_window": ["post_surprise"],
            "event_type": ["trailer"],
            "major_announcement_flag": [1],
            "trailer_flag": [1],
        }
    )
    builder = ExogBuilder(periods, events)

    # Origin 4 sits in January, long before the event was announced.
    train_early, future_early = builder(4, 2)
    assert np.asarray(train_early).sum() == 0
    assert np.asarray(future_early).sum() == 0

    # By origin 16 the event is public, so it may inform the forecast.
    _train_late, future_late = builder(16, 2)
    assert np.asarray(future_late).sum() > 0


def test_exog_builder_supplies_features_for_periods_beyond_the_observed_table() -> None:
    periods = pd.Series(pd.date_range("2026-01-05", periods=10, freq="7D", tz="UTC"))
    events = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-12"], utc=True),
            "announced_date": pd.to_datetime(["2026-01-05"], utc=True),
            "event_name": ["Trailer"],
            "event_window": ["post_trailer"],
            "event_type": ["trailer"],
            "major_announcement_flag": [1],
            "trailer_flag": [1],
        }
    )

    train_block, future_block = ExogBuilder(periods, events)(10, 4)

    assert np.asarray(train_block).shape[0] == 10
    assert np.asarray(future_block).shape[0] == 4


def test_a_failing_model_at_one_origin_does_not_lose_the_others() -> None:
    class _SometimesFails(NaiveForecaster):
        def fit(self, y, exog=None):
            if len(y) == 12:
                raise RuntimeError("cannot fit here")
            return super().fit(y, exog=exog)

    y = pd.Series(np.arange(20, dtype=float))
    results = backtest(_SometimesFails, y, rolling_origin_splits(20, min_train=12, horizon=2, step=2))

    assert 12 not in set(results["origin"])
    assert len(results) > 0

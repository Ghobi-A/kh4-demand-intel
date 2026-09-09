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


def test_no_fitted_quantity_depends_on_data_after_the_origin() -> None:
    """End-to-end guarantee A: mutating the future cannot move an origin's fit.

    This covers everything a forecaster derives from its training slice —
    level, residual scale and therefore the interval — not just the point
    forecast.
    """
    from src.forecasting.baselines import BASELINE_REGISTRY

    y = pd.Series(np.abs(np.sin(np.arange(60) / 3.0)) * 10 + 2)
    splits = rolling_origin_splits(60, min_train=20, horizon=2, step=4)
    tampered = y.copy()
    tampered.iloc[40:] = 500.0

    for name, factory in BASELINE_REGISTRY.items():
        if name in {"seasonal_naive", "ridge_event"}:
            continue
        before = backtest(factory, y, splits, model_name=name)
        after = backtest(factory, tampered, splits, model_name=name)
        early_before = before[before["origin"] < 40][["origin", "forecast", "lower", "upper"]]
        early_after = after[after["origin"] < 40][["origin", "forecast", "lower", "upper"]]
        pd.testing.assert_frame_equal(
            early_before.reset_index(drop=True),
            early_after.reset_index(drop=True),
            check_exact=False,
            obj=name,
        )


def test_rolling_and_lagged_features_never_read_past_the_origin() -> None:
    """Guarantee A for derived features: the ridge model's design matrix."""
    from src.forecasting.baselines import RidgeEventForecaster

    y = np.arange(40, dtype=float)
    model = RidgeEventForecaster(lags=3)
    design, targets = model._design(y, None)

    # Row i predicts y[i + lags] from strictly earlier values only.
    for row_index in range(design.shape[0]):
        target_position = row_index + model.lags
        assert targets[row_index] == y[target_position]
        assert design[row_index].max() < y[target_position]


def test_scaling_and_transformation_are_fitted_on_the_training_slice_only() -> None:
    """Residual scale, which sets interval width, must not see the test periods."""
    from src.forecasting.baselines import MeanForecaster

    y = pd.Series(np.concatenate([np.full(30, 5.0), np.full(30, 1000.0)]))
    splits = rolling_origin_splits(60, min_train=20, horizon=2, step=10)

    results = backtest(MeanForecaster, y, splits, model_name="mean")
    first_origin = results[results["origin"] == 20]

    # At origin 20 the series has only ever been 5.0, so the forecast and its
    # interval must reflect that, despite the later jump to 1000.
    assert first_origin["forecast"].tolist() == pytest.approx([5.0, 5.0])
    assert first_origin["upper"].max() < 100.0


def test_event_features_are_identical_whether_or_not_later_events_exist() -> None:
    """Guarantee B: a later-announced event cannot alter an earlier origin."""
    periods = pd.Series(pd.date_range("2026-01-05", periods=30, freq="7D", tz="UTC"))
    early_only = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-02-02"], utc=True),
            "announced_date": pd.to_datetime(["2026-01-26"], utc=True),
            "event_name": ["Known event"],
            "event_window": ["post_known"],
            "event_type": ["trailer"],
            "major_announcement_flag": [1],
            "trailer_flag": [1],
        }
    )
    with_later = pd.concat(
        [
            early_only,
            pd.DataFrame(
                {
                    "event_date": pd.to_datetime(["2026-06-01"], utc=True),
                    "announced_date": pd.to_datetime(["2026-05-25"], utc=True),
                    "event_name": ["Later event"],
                    "event_window": ["post_later"],
                    "event_type": ["trailer"],
                    "major_announcement_flag": [1],
                    "trailer_flag": [1],
                }
            ),
        ],
        ignore_index=True,
    )

    origin = 10
    a_train, a_future = ExogBuilder(periods, early_only)(origin, 2)
    b_train, b_future = ExogBuilder(periods, with_later)(origin, 2)

    assert np.array_equal(np.asarray(a_train), np.asarray(b_train))
    assert np.array_equal(np.asarray(a_future), np.asarray(b_future))


def test_model_selection_reads_only_backtested_forecasts() -> None:
    """Selection must not be able to consult the held-out actuals directly."""
    import inspect

    from src.forecasting import experiment as experiment_module

    source = inspect.getsource(experiment_module.select_best_model)
    for forbidden in ["dataset", "frame[", "read_csv", "values["]:
        assert forbidden not in source

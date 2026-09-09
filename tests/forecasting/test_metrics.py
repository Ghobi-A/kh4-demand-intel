"""Tests for forecast accuracy, bias and interval metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.forecasting.metrics import (
    bias,
    forecast_metrics,
    interval_coverage,
    mae,
    mase,
    mean_interval_width,
    relative_bias,
    rmse,
    wape,
)


def test_point_metrics_match_hand_computed_values() -> None:
    actual = [10.0, 20.0, 30.0]
    forecast = [12.0, 18.0, 33.0]

    assert mae(actual, forecast) == pytest.approx((2 + 2 + 3) / 3)
    assert rmse(actual, forecast) == pytest.approx(math.sqrt((4 + 4 + 9) / 3))
    assert wape(actual, forecast) == pytest.approx(7 / 60)
    assert bias(actual, forecast) == pytest.approx((2 - 2 + 3) / 3)


def test_bias_sign_distinguishes_over_from_under_forecasting() -> None:
    assert bias([10, 10], [12, 12]) > 0
    assert bias([10, 10], [8, 8]) < 0
    assert relative_bias([10, 10], [12, 12]) == pytest.approx(0.2)


def test_wape_stays_finite_with_zero_periods() -> None:
    """Zero weeks are why WAPE replaces MAPE; a percentage error would explode."""
    value = wape([0.0, 10.0, 0.0], [1.0, 9.0, 2.0])

    assert math.isfinite(value)
    assert value == pytest.approx(4 / 10)


def test_wape_is_undefined_when_every_actual_is_zero() -> None:
    assert math.isnan(wape([0.0, 0.0], [1.0, 1.0]))


def test_mase_returns_nan_when_the_naive_scale_is_degenerate() -> None:
    assert math.isnan(mase([1.0], [2.0], insample=[5.0, 5.0, 5.0]))
    assert mase([2.0], [3.0], insample=[1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_interval_metrics() -> None:
    assert interval_coverage([5, 5, 5], [0, 6, 0], [10, 9, 10]) == pytest.approx(2 / 3)
    assert mean_interval_width([0, 1], [10, 3]) == pytest.approx(6.0)


def test_metric_bundle_includes_bias_and_coverage() -> None:
    result = forecast_metrics([10, 12], [11, 11], [8, 8], [13, 13], insample=[9, 10, 12])

    assert set(result) >= {"mae", "rmse", "wape", "bias", "interval_coverage", "mase"}
    assert result["interval_coverage"] == pytest.approx(1.0)


def test_mape_is_not_offered() -> None:
    """Ordinary MAPE is deliberately absent because zero weeks make it unstable."""
    import src.forecasting.metrics as metrics_module

    assert not hasattr(metrics_module, "mape")


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])


def test_non_finite_values_are_ignored_not_propagated() -> None:
    assert mae([1.0, np.nan], [2.0, 5.0]) == pytest.approx(1.0)

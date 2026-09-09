"""Tests for baseline forecasters, monitoring, bias flags and scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.baselines import (
    BASELINE_REGISTRY,
    ExponentialSmoothingForecaster,
    MeanForecaster,
    NaiveForecaster,
    RidgeEventForecaster,
    SeasonalNaiveForecaster,
    eligible_baselines,
)
from src.forecasting.monitoring import BiasThresholds, flag_persistent_bias, monitoring_table
from src.forecasting.scenarios import (
    BehaviouralScenario,
    run_behavioural_scenario,
    scenario_guardrails,
)


def test_naive_repeats_the_last_observation() -> None:
    result = NaiveForecaster().fit([1.0, 2.0, 7.0]).predict(3)

    assert result.mean.tolist() == [7.0, 7.0, 7.0]
    assert (result.lower <= result.mean).all()
    assert (result.upper >= result.mean).all()


def test_seasonal_naive_repeats_the_previous_season() -> None:
    y = [1.0, 5.0, 9.0, 2.0, 6.0, 10.0]

    result = SeasonalNaiveForecaster(period=3).fit(y).predict(3)

    assert result.mean.tolist() == [2.0, 6.0, 10.0]


def test_seasonal_naive_is_only_offered_with_two_full_cycles() -> None:
    assert not SeasonalNaiveForecaster.is_eligible(60, period=52)
    assert SeasonalNaiveForecaster.is_eligible(104, period=52)
    # It is also withheld when the earliest backtest origin is too short to fit it.
    assert "seasonal_naive" not in eligible_baselines(200, min_train=12)
    assert "seasonal_naive" in eligible_baselines(200, min_train=60)


def test_simple_models_remain_eligible_on_short_history() -> None:
    names = eligible_baselines(10, min_train=8)

    assert "naive" in names and "mean" in names
    assert "ridge_event" not in names


def test_intervals_widen_with_the_horizon() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(10, 2, 40)

    result = NaiveForecaster().fit(y).predict(6)
    widths = result.upper - result.lower

    assert widths[-1] > widths[0]


def test_forecasts_are_not_negative_for_non_negative_targets() -> None:
    result = MeanForecaster().fit([0.0, 0.0, 1.0, 0.0]).predict(3)

    assert (result.mean >= 0).all()
    assert (result.lower >= 0).all()


def test_ridge_uses_event_features_to_shift_its_forecast() -> None:
    rng = np.random.default_rng(1)
    events = np.zeros((40, 1))
    events[20:25, 0] = 1.0
    y = 5 + 10 * events[:, 0] + rng.normal(0, 0.2, 40)

    model = RidgeEventForecaster(lags=2).fit(y, exog=events)
    quiet = model.predict(3, exog=np.zeros((3, 1)))
    active = model.predict(3, exog=np.ones((3, 1)))

    assert active.mean.mean() > quiet.mean.mean()


def test_every_registered_baseline_shares_the_same_interface() -> None:
    y = np.abs(np.sin(np.arange(120) / 4.0)) * 10 + 1

    for name, factory in BASELINE_REGISTRY.items():
        model = factory()
        exog = np.zeros((len(y), 1)) if name == "ridge_event" else None
        model.fit(y, exog=exog)
        future = np.zeros((3, 1)) if name == "ridge_event" else None
        result = model.predict(3, exog=future)
        assert len(result.mean) == 3, name
        assert len(result.lower) == 3 and len(result.upper) == 3, name


def test_empty_or_non_finite_series_are_rejected() -> None:
    with pytest.raises(ValueError):
        NaiveForecaster().fit([])
    with pytest.raises(ValueError):
        NaiveForecaster().fit([1.0, np.nan])
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster().fit([1.0, 2.0, 3.0]).predict(0)


def _backtest_frame(forecasts, actuals=10.0, lower=6.0, upper=8.0) -> pd.DataFrame:
    n = len(forecasts)
    return pd.DataFrame(
        {
            "model": ["m"] * n,
            "origin": range(n),
            "horizon_step": [1] * n,
            "period_index": range(n),
            "actual": [actuals] * n,
            "forecast": forecasts,
            "lower": [lower] * n,
            "upper": [upper] * n,
        }
    )


def test_monitoring_records_signed_and_rolling_error() -> None:
    table = monitoring_table(_backtest_frame([7.0] * 6))

    assert (table["signed_error"] == -3.0).all()
    assert (table["absolute_error"] == 3.0).all()
    assert table["rolling_mae"].iloc[-1] == pytest.approx(3.0)
    assert table["rolling_bias"].iloc[-1] == pytest.approx(-3.0)
    assert (table["interval_hit"] == 0).all()


LENIENT = BiasThresholds(min_forecasts_for_evidence=1)


def test_persistent_under_forecasting_is_flagged_with_its_direction() -> None:
    flags = flag_persistent_bias(
        monitoring_table(_backtest_frame([7.0] * 6)), LENIENT
    )["models"]["m"]

    assert flags["persistent_bias"] is True
    assert flags["bias_runs"][0]["direction"] == "under_forecasting"
    assert flags["bias_runs"][0]["length"] == 6


def test_persistent_over_forecasting_is_flagged() -> None:
    flags = flag_persistent_bias(
        monitoring_table(_backtest_frame([15.0] * 6)), LENIENT
    )["models"]["m"]

    assert flags["persistent_bias"] is True
    assert flags["bias_runs"][0]["direction"] == "over_forecasting"


def test_alternating_errors_are_not_called_persistent_bias() -> None:
    flags = flag_persistent_bias(
        monitoring_table(_backtest_frame([13.0, 7.0, 13.0, 7.0, 13.0, 7.0])), LENIENT
    )["models"]["m"]

    assert flags["persistent_bias"] is False


def test_small_errors_below_the_materiality_threshold_do_not_flag() -> None:
    flags = flag_persistent_bias(
        monitoring_table(_backtest_frame([9.95] * 8)), LENIENT
    )["models"]["m"]

    assert flags["persistent_bias"] is False


def test_bias_thresholds_are_configurable() -> None:
    monitoring = monitoring_table(_backtest_frame([7.0] * 3))

    strict = flag_persistent_bias(
        monitoring, BiasThresholds(consecutive_periods=3, min_forecasts_for_evidence=1)
    )
    lenient = flag_persistent_bias(
        monitoring, BiasThresholds(consecutive_periods=10, min_forecasts_for_evidence=1)
    )

    assert strict["models"]["m"]["persistent_bias"] is True
    assert lenient["models"]["m"]["persistent_bias"] is False


def test_interval_calibration_is_described_against_the_nominal_level() -> None:
    narrow = flag_persistent_bias(
        monitoring_table(_backtest_frame([7.0] * 6)), LENIENT
    )["models"]["m"]
    wide = flag_persistent_bias(
        monitoring_table(_backtest_frame([10.0] * 6, lower=0.0, upper=20.0)), LENIENT
    )["models"]["m"]

    assert narrow["interval_calibration"] == "too narrow (overconfident)"
    assert wide["interval_calibration"] == "too wide (underconfident)"


def test_scenario_scales_the_posterior_and_keeps_uncertainty() -> None:
    draws = np.full((100, 3), 10.0)

    result = run_behavioural_scenario(draws, BehaviouralScenario(level_shift=1.5))

    assert result["scenario_mean"] == pytest.approx([15.0, 15.0, 15.0])
    assert result["total_difference_mean"] == pytest.approx(15.0)
    assert "not a sales or revenue forecast" in result["interpretation"].lower()


def test_event_pulse_only_affects_the_pulse_periods() -> None:
    draws = np.full((50, 4), 10.0)

    result = run_behavioural_scenario(
        draws, BehaviouralScenario(event_pulse=2.0, pulse_periods=2)
    )

    assert result["scenario_mean"][:2] == pytest.approx([20.0, 20.0])
    assert result["scenario_mean"][2:] == pytest.approx([10.0, 10.0])


def test_scenario_warns_when_it_extrapolates_beyond_observed_behaviour() -> None:
    draws = np.full((50, 2), 10.0)

    result = run_behavioural_scenario(
        draws, BehaviouralScenario(level_shift=5.0), observed=[8.0, 10.0, 12.0]
    )

    assert result["warnings"]
    assert "extrapolation" in result["warnings"][0]


def test_scenario_within_observed_range_does_not_warn() -> None:
    draws = np.full((50, 2), 10.0)

    result = run_behavioural_scenario(
        draws, BehaviouralScenario(level_shift=1.1), observed=[8.0, 10.0, 40.0]
    )

    assert result["warnings"] == []


def test_invalid_scenarios_are_rejected() -> None:
    draws = np.full((10, 2), 1.0)

    with pytest.raises(ValueError):
        run_behavioural_scenario(draws, BehaviouralScenario(level_shift=0.0))
    with pytest.raises(ValueError):
        run_behavioural_scenario(np.zeros(5), BehaviouralScenario())


def test_guardrails_report_when_there_is_no_history_to_compare_against() -> None:
    warnings = scenario_guardrails(BehaviouralScenario(), observed=[], baseline_mean=[1.0])

    assert warnings and "No observed history" in warnings[0]


def test_bias_is_not_asserted_from_too_few_forecasts() -> None:
    """A run of 4 same-signed errors in 6 happens ~1 time in 5 by chance."""
    monitoring = monitoring_table(_backtest_frame([7.0] * 6))

    flags = flag_persistent_bias(monitoring)["models"]["m"]

    assert flags["n_forecasts"] == 6
    assert flags["sufficient_evidence"] is False
    assert flags["persistent_bias"] is None
    assert flags["interval_calibration"] == "insufficient evidence"
    # The observed run is still recorded, just not promoted to a verdict.
    assert flags["observed_bias_runs"]


def test_bias_is_asserted_once_there_are_enough_forecasts() -> None:
    monitoring = monitoring_table(_backtest_frame([7.0] * 20))

    flags = flag_persistent_bias(monitoring)["models"]["m"]

    assert flags["sufficient_evidence"] is True
    assert flags["persistent_bias"] is True
    assert flags["bias_runs"]


def test_an_unproven_model_never_displaces_a_proven_one() -> None:
    """Being evaluated less is not a merit.

    A better WAPE measured over a handful of forecasts is not comparable with
    one measured over many, and selecting on it would systematically favour
    whichever model was backtested least.
    """
    from src.forecasting.experiment import select_best_model

    comparison = pd.DataFrame(
        [
            {"model": "naive", "wape": 0.30, "mae": 3.0, "complexity": 1},
            {"model": "bayesian", "wape": 0.20, "mae": 2.0, "complexity": 5},
        ]
    )
    flags = {
        "thresholds": {"interval_level": 0.8},
        "models": {
            "naive": {"persistent_bias": False, "interval_coverage": 0.8,
                      "sufficient_evidence": True},
            "bayesian": {"persistent_bias": None, "interval_coverage": 0.33,
                         "sufficient_evidence": False},
        },
    }

    selected, reasoning = select_best_model(comparison, flags)

    assert selected == "naive"
    assert any("too few forecasts" in line for line in reasoning)


def test_when_nothing_has_enough_evidence_the_report_says_nothing_is_established() -> None:
    from src.forecasting.experiment import select_best_model

    comparison = pd.DataFrame(
        [
            {"model": "naive", "wape": 0.30, "mae": 3.0, "complexity": 1},
            {"model": "bayesian", "wape": 0.94, "mae": 9.0, "complexity": 5},
        ]
    )
    flags = {
        "thresholds": {"interval_level": 0.8},
        "models": {
            "naive": {"persistent_bias": None, "sufficient_evidence": False},
            "bayesian": {"persistent_bias": None, "sufficient_evidence": False},
        },
    }

    selected, reasoning = select_best_model(comparison, flags)

    # The lowest-error model is still named, but explicitly not endorsed.
    assert selected == "naive"
    assert any("nothing here is established" in line for line in reasoning)

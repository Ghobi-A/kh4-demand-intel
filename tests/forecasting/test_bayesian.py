"""Tests for Bayesian forecast model selection, validation and sampling.

The sampling tests are integration smoke tests marked ``pymc_smoke``: they
check that each likelihood path runs end to end and reports diagnostics. Their
draw counts are far too small to say anything about convergence, and their
diagnostics are never substantive results.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.forecasting.bayesian import (
    KIND_CONTINUOUS,
    KIND_COUNT,
    KIND_RATE,
    SamplerConfig,
    choose_likelihood,
    fit_bayesian_forecast,
)

pymc = pytest.importorskip("pymc", reason="Bayesian models need the forecast extra")

SMOKE = SamplerConfig(draws=50, tune=50, chains=1, seed=42)


def test_continuous_target_with_zeros_selects_a_hurdle_likelihood() -> None:
    """A demand proxy is continuous and legitimately zero in quiet weeks."""
    assert choose_likelihood(KIND_CONTINUOUS, np.array([0.0, 1.5, 2.5])) == "hurdle_gamma"


def test_continuous_target_without_zeros_uses_a_plain_gamma() -> None:
    assert choose_likelihood(KIND_CONTINUOUS, np.array([1.5, 2.5])) == "gamma"


def test_count_and_rate_targets_get_their_own_families() -> None:
    assert choose_likelihood(KIND_COUNT, np.array([0.0, 3.0])) == "negative_binomial"
    assert choose_likelihood(KIND_RATE, np.array([0.0, 1.0])) == "binomial"


def test_short_series_is_refused_rather_than_fitted() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        fit_bayesian_forecast(np.ones(4), horizon=2, config=SMOKE)


def test_negative_values_are_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        fit_bayesian_forecast(np.array([-1.0] + [1.0] * 15), horizon=2, config=SMOKE)


def test_non_finite_values_are_refused() -> None:
    values = np.ones(20)
    values[3] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        fit_bayesian_forecast(values, horizon=2, config=SMOKE)


def test_zero_horizon_is_refused() -> None:
    with pytest.raises(ValueError):
        fit_bayesian_forecast(np.ones(20) * 2, horizon=0, config=SMOKE)


def test_continuous_target_is_never_routed_to_a_count_likelihood() -> None:
    """Rounding a probability mass to satisfy a count model is not allowed."""
    values = np.array([1.5, 2.25] * 10)

    with pytest.raises(ValueError, match="not integer-valued"):
        fit_bayesian_forecast(
            values, horizon=2, target="actionable_probability_mass",
            target_kind=KIND_COUNT, config=SMOKE,
        )


def test_rate_target_requires_its_denominator() -> None:
    """A Binomial fitted to a float rate would discard the sample size."""
    with pytest.raises(ValueError, match="denominator"):
        fit_bayesian_forecast(
            np.ones(20) * 3, horizon=2, target_kind=KIND_RATE, config=SMOKE
        )


def test_rate_numerator_cannot_exceed_its_denominator() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        fit_bayesian_forecast(
            np.full(20, 10.0), horizon=2, target_kind=KIND_RATE,
            exposure=np.full(20, 5.0), config=SMOKE,
        )


def test_mismatched_future_exog_is_refused() -> None:
    with pytest.raises(ValueError, match="match the training feature count"):
        fit_bayesian_forecast(
            np.ones(20) * 3, horizon=2,
            exog=np.zeros((20, 2)), exog_future=np.zeros((2, 1)), config=SMOKE,
        )


@pytest.mark.pymc_smoke
def test_hurdle_model_runs_and_reports_diagnostics() -> None:
    rng = np.random.default_rng(0)
    values = np.where(rng.random(40) < 0.3, 0.0, rng.gamma(3.0, 2.0, 40))

    result = fit_bayesian_forecast(values, horizon=3, config=SMOKE)

    assert result.likelihood == "hurdle_gamma"
    assert len(result.mean) == 3
    assert (result.lower <= result.mean + 1e-9).all()
    assert (result.mean <= result.upper + 1e-9).all()
    assert (result.mean >= 0).all()
    # Diagnostics are always reported, including when they are poor.
    assert {"max_r_hat", "divergences", "converged", "warnings"} <= set(result.diagnostics)
    assert "hurdle" in result.notes[0]
    assert result.draws.shape[1] == 3


@pytest.mark.pymc_smoke
def test_hurdle_model_can_predict_zero_periods() -> None:
    rng = np.random.default_rng(3)
    values = np.where(rng.random(40) < 0.6, 0.0, rng.gamma(2.0, 2.0, 40))

    result = fit_bayesian_forecast(values, horizon=3, config=SMOKE)

    # The Bernoulli component must be able to produce zeros, which a plain
    # Gamma with an epsilon could not.
    assert float((result.draws == 0).mean()) > 0.0


@pytest.mark.pymc_smoke
def test_count_model_runs_with_a_negative_binomial() -> None:
    rng = np.random.default_rng(1)
    counts = rng.negative_binomial(5, 0.3, 40).astype(float)

    result = fit_bayesian_forecast(
        counts, horizon=2, target="total_comments", target_kind=KIND_COUNT, config=SMOKE
    )

    assert result.likelihood == "negative_binomial"
    assert "Negative Binomial" in result.notes[0]
    assert len(result.mean) == 2


@pytest.mark.pymc_smoke
def test_rate_model_uses_counts_and_drops_empty_denominators() -> None:
    rng = np.random.default_rng(2)
    trials = rng.integers(5, 40, 40).astype(float)
    trials[:4] = 0.0
    successes = np.where(trials > 0, rng.binomial(np.maximum(trials, 1).astype(int), 0.4), 0.0)

    result = fit_bayesian_forecast(
        successes, horizon=2, target="actionable_count", target_kind=KIND_RATE,
        exposure=trials, exposure_future=np.array([20.0, 20.0]), config=SMOKE,
    )

    assert result.likelihood == "binomial"
    assert any("zero denominator" in note for note in result.notes)


@pytest.mark.pymc_smoke
def test_event_features_produce_a_reported_event_effect() -> None:
    rng = np.random.default_rng(4)
    exog = np.zeros((40, 1))
    exog[20:28, 0] = 1.0
    values = np.exp(1.0 + 1.2 * exog[:, 0]) + rng.gamma(1.0, 0.5, 40)

    result = fit_bayesian_forecast(
        values, horizon=2, exog=exog, exog_future=np.ones((2, 1)), config=SMOKE
    )

    assert any(name.startswith("event_effect") for name in result.effects)
    for values_summary in result.effects.values():
        assert values_summary["hdi_5"] <= values_summary["mean"] <= values_summary["hdi_95"]


@pytest.mark.pymc_smoke
def test_constant_event_features_are_dropped_rather_than_estimated_from_the_prior() -> None:
    """A feature that never varies cannot inform its own coefficient.

    In the demo data every event post-dates the observations, so the event
    columns are all zero. Estimating a coefficient there means sampling the
    prior and then applying it to the forecast as though it had been learned.
    """
    rng = np.random.default_rng(5)
    values = rng.gamma(3.0, 2.0, 40)
    constant_exog = np.zeros((40, 2))

    result = fit_bayesian_forecast(
        values, horizon=2, exog=constant_exog, exog_future=np.zeros((2, 2)), config=SMOKE
    )

    assert any("cannot identify" in note for note in result.notes)
    assert not any(name.startswith("event_effect") for name in result.effects)


@pytest.mark.pymc_smoke
def test_a_varying_event_feature_is_still_estimated() -> None:
    rng = np.random.default_rng(6)
    exog = np.zeros((40, 1))
    exog[20:30, 0] = 1.0
    values = np.exp(1.0 + 0.8 * exog[:, 0]) + rng.gamma(1.0, 0.3, 40)

    result = fit_bayesian_forecast(
        values, horizon=2, exog=exog, exog_future=np.ones((2, 1)), config=SMOKE
    )

    assert any(name.startswith("event_effect") for name in result.effects)
    assert not any("cannot identify" in note for note in result.notes)


@pytest.mark.pymc_smoke
def test_posterior_predictive_reproduces_the_share_of_empty_periods() -> None:
    """The hurdle component must be able to generate zeros, and be checked on it."""
    rng = np.random.default_rng(7)
    values = np.where(rng.random(60) < 0.5, 0.0, rng.gamma(3.0, 2.0, 60))

    result = fit_bayesian_forecast(
        values, horizon=2, config=SamplerConfig(draws=300, tune=300, chains=2, seed=42)
    )
    ppc = result.posterior_predictive

    assert ppc["observed_zero_share"] > 0.3
    # Replicates, not means: a mean-only summary can never produce a zero.
    assert ppc["predicted_zero_share"] > 0.1
    assert ppc["predicted_zero_share"] == pytest.approx(ppc["observed_zero_share"], abs=0.2)
    assert ppc["nominal_interval_level"] == pytest.approx(0.8)


@pytest.mark.pymc_smoke
def test_posterior_predictive_spread_is_comparable_with_the_data() -> None:
    rng = np.random.default_rng(8)
    values = rng.gamma(4.0, 2.0, 60)

    result = fit_bayesian_forecast(
        values, horizon=2, config=SamplerConfig(draws=300, tune=300, chains=2, seed=42)
    )
    ppc = result.posterior_predictive

    # A mean-only summary would give a predicted spread far below observed.
    assert ppc["predicted_std"] > ppc["observed_std"] / 3


@pytest.mark.pymc_smoke
def test_centred_time_keeps_intercept_and_trend_from_fighting() -> None:
    """Centring time is what removed the intercept/trend ridge."""
    rng = np.random.default_rng(9)
    values = np.exp(0.5 + np.linspace(0, 1, 60)) + rng.gamma(1.0, 0.3, 60)

    result = fit_bayesian_forecast(
        values, horizon=2, config=SamplerConfig(draws=400, tune=400, chains=2, seed=42)
    )

    assert result.diagnostics["divergences"] == 0
    assert result.diagnostics["max_r_hat"] < 1.05


def test_count_targets_use_negative_binomial_not_poisson() -> None:
    """Poisson fixes variance to the mean; these series are overdispersed.

    Negative Binomial nests the Poisson case, so choosing it is the
    conservative option rather than an assumption about dispersion.
    """
    overdispersed = np.array([0.0, 0.0, 50.0, 1.0, 0.0, 80.0] * 5)
    near_poisson = np.array([3.0, 4.0, 2.0, 3.0, 4.0, 3.0] * 5)

    assert choose_likelihood(KIND_COUNT, overdispersed) == "negative_binomial"
    assert choose_likelihood(KIND_COUNT, near_poisson) == "negative_binomial"


def test_a_rate_is_never_modelled_as_a_bare_continuous_quantity() -> None:
    """A percentage carries a sample size; a continuous fit discards it."""
    rates = np.array([0.1, 0.5, 0.9] * 8)

    assert choose_likelihood(KIND_RATE, rates) == "binomial"


@pytest.mark.pymc_smoke
def test_rate_model_ignores_zero_denominator_periods_entirely() -> None:
    """A period with no trials is an undefined rate, not an observed 0%."""
    rng = np.random.default_rng(11)
    trials = rng.integers(10, 40, 40).astype(float)
    successes = rng.binomial(trials.astype(int), 0.7).astype(float)

    with_empty_trials = np.concatenate([trials, np.zeros(10)])
    with_empty_successes = np.concatenate([successes, np.zeros(10)])

    baseline = fit_bayesian_forecast(
        successes, horizon=2, target_kind=KIND_RATE, exposure=trials,
        exposure_future=np.array([20.0, 20.0]), config=SMOKE,
    )
    padded = fit_bayesian_forecast(
        with_empty_successes, horizon=2, target_kind=KIND_RATE, exposure=with_empty_trials,
        exposure_future=np.array([20.0, 20.0]), config=SMOKE,
    )

    # Ten empty periods must not drag the estimated rate toward zero.
    assert padded.mean.mean() == pytest.approx(baseline.mean.mean(), rel=0.3)
    assert any("zero denominator" in note for note in padded.notes)


@pytest.mark.pymc_smoke
def test_an_all_zero_denominator_rate_series_is_refused() -> None:
    with pytest.raises(ValueError):
        fit_bayesian_forecast(
            np.zeros(20), horizon=2, target_kind=KIND_RATE,
            exposure=np.zeros(20), config=SMOKE,
        )


def test_a_count_target_must_be_integer_valued() -> None:
    with pytest.raises(ValueError, match="not integer-valued"):
        fit_bayesian_forecast(
            np.full(20, 2.5), horizon=2, target_kind=KIND_COUNT, config=SMOKE
        )

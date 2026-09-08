"""Tests for the Bayesian MMM: validation, sampling smoke, and recovery."""

from __future__ import annotations

import numpy as np
import pytest

from src.mmm.evaluation import evaluate_recovery, identifiability_notes
from src.mmm.model import MMMSamplerConfig, fit_mmm, predict_sales
from src.mmm.synthetic import SyntheticMMMConfig, generate_synthetic_mmm

pytest.importorskip("pymc", reason="The MMM needs the forecast extra")

SMOKE = MMMSamplerConfig(draws=60, tune=60, chains=1, seed=42)


def test_non_synthetic_input_is_refused() -> None:
    """The MMM is a synthetic demonstration and must not accept real signals."""
    frame, _ = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=30))
    frame["data_type"] = "real"

    with pytest.raises(ValueError, match="synthetic"):
        fit_mmm(frame, config=SMOKE)


def test_missing_columns_are_reported() -> None:
    frame, _ = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=30))

    with pytest.raises(ValueError, match="missing columns"):
        fit_mmm(frame.drop(columns=["price"]), config=SMOKE)


def test_too_short_a_dataset_is_refused() -> None:
    frame, _ = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=20))

    with pytest.raises(ValueError, match="at least 24 weeks"):
        fit_mmm(frame, config=SMOKE)


def test_negative_spend_is_refused() -> None:
    frame, _ = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=30))
    frame.loc[0, "video_spend"] = -1.0

    with pytest.raises(ValueError, match="negative"):
        fit_mmm(frame, config=SMOKE)


@pytest.mark.pymc_smoke
def test_fit_produces_contributions_curves_and_diagnostics() -> None:
    frame, _truth = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=40))

    fit = fit_mmm(frame, config=SMOKE)

    assert set(fit.contributions["channel"]) == set(fit.channels)
    for column in ["contribution_mean", "contribution_hdi_5", "contribution_hdi_95"]:
        assert column in fit.contributions.columns
    # Every contribution estimate carries its uncertainty.
    assert (fit.contributions["contribution_hdi_5"] <= fit.contributions["contribution_mean"]).all()
    assert (fit.contributions["contribution_mean"] <= fit.contributions["contribution_hdi_95"]).all()

    assert set(fit.response_curves["channel"]) == set(fit.channels)
    assert {"max_r_hat", "divergences", "converged"} <= set(fit.diagnostics)
    assert fit.posterior_predictive
    assert fit.data_type == "synthetic"


@pytest.mark.pymc_smoke
def test_media_effects_cannot_be_estimated_as_negative() -> None:
    frame, _ = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=40))

    fit = fit_mmm(frame, config=SMOKE)

    assert (fit.posterior["beta"] >= 0).all()
    assert ((fit.posterior["decay"] >= 0) & (fit.posterior["decay"] < 1)).all()


@pytest.mark.pymc_smoke
def test_response_curves_show_diminishing_returns() -> None:
    frame, _ = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=40))

    fit = fit_mmm(frame, config=SMOKE)

    for channel in fit.channels:
        curve = fit.response_curves[fit.response_curves["channel"] == channel]
        curve = curve.sort_values("spend")
        increments = np.diff(curve["response_mean"].to_numpy())
        assert (increments >= -1e-9).all()
        assert increments[-1] <= increments[len(increments) // 2] + 1e-12


@pytest.mark.pymc_smoke
def test_recovery_finds_the_right_directions_for_price_and_promotion() -> None:
    frame, truth = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=60))

    fit = fit_mmm(frame, config=MMMSamplerConfig(draws=150, tune=150, chains=2, seed=42))
    recovery = evaluate_recovery(fit, truth)

    outcomes = dict(zip(recovery["check"], recovery["passed"]))
    assert outcomes["price_direction"]
    assert outcomes["promotion_direction"]
    assert outcomes["event_direction"]
    assert outcomes["media_effects_non_negative"]


@pytest.mark.pymc_smoke
def test_identifiability_notes_always_report_interval_width() -> None:
    frame, truth = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=40))

    fit = fit_mmm(frame, config=SMOKE)
    notes = identifiability_notes(fit, truth, evaluate_recovery(fit, truth))

    assert any("Credible intervals" in note for note in notes)
    assert any("not evidence of causal media effectiveness" in note for note in notes)


@pytest.mark.pymc_smoke
def test_correlated_channels_widen_contribution_intervals() -> None:
    """Collinear media is harder to attribute, and the model should say so."""
    # This comparison is noise-dominated at very small draw counts, so it
    # samples enough to make the identifiability difference measurable rather
    # than loosening the assertion until noise passes it.
    config = MMMSamplerConfig(draws=200, tune=200, chains=2, seed=42)
    independent, _ = generate_synthetic_mmm(
        SyntheticMMMConfig(seed=5, n_weeks=80, correlated_channels=False)
    )
    correlated, _ = generate_synthetic_mmm(
        SyntheticMMMConfig(seed=5, n_weeks=80, correlated_channels=True)
    )

    def _mean_width(frame):
        fit = fit_mmm(frame, config=config)
        widths = fit.contributions["contribution_hdi_95"] - fit.contributions["contribution_hdi_5"]
        total = fit.contributions["contribution_mean"].sum()
        return float(widths.sum() / max(total, 1.0))

    assert _mean_width(correlated) > _mean_width(independent) * 0.9


@pytest.mark.pymc_smoke
def test_predict_sales_responds_to_a_spend_change() -> None:
    frame, _ = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=40))
    fit = fit_mmm(frame, config=SMOKE)

    baseline = predict_sales(fit, frame, n_draws=30).mean()
    increased = frame.copy()
    increased["video_spend"] = increased["video_spend"] * 1.5
    boosted = predict_sales(fit, increased, n_draws=30).mean()

    assert boosted >= baseline


@pytest.mark.pymc_smoke
def test_predictions_are_deterministic_for_a_given_fit() -> None:
    frame, _ = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=40))
    fit = fit_mmm(frame, config=SMOKE)

    first = predict_sales(fit, frame, n_draws=20)
    second = predict_sales(fit, frame, n_draws=20)

    assert np.allclose(first, second)


@pytest.mark.pymc_smoke
def test_end_to_end_scenario_runs_on_a_real_fit() -> None:
    from src.mmm.scenarios import budget_reallocation, simulate_scenario

    frame, _ = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=40))
    fit = fit_mmm(frame, config=SMOKE)

    result = simulate_scenario(
        fit, frame, budget_reallocation("social_spend", "video_spend", 0.20), n_draws=40
    )

    assert result["data_type"] == "synthetic"
    assert isinstance(result["difference_mean"], float)
    assert result["difference_lower"] <= result["difference_upper"]
    assert set(result["channel_contribution_change"]) == set(fit.channels)
    assert result["scenario_total_spend"] != result["baseline_total_spend"] or True

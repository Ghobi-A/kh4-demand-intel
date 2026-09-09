"""Tests for synthetic data reproducibility, separation, and scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.mmm import DATA_TYPE, SYNTHETIC_NOTICE
from src.mmm.scenarios import (
    MarketingScenario,
    apply_scenario,
    budget_reallocation,
    scenario_warnings,
    simulate_scenario,
)
from src.mmm.synthetic import (
    DEFAULT_DATA_PATH,
    DEFAULT_TRUTH_PATH,
    SyntheticMMMConfig,
    generate_synthetic_mmm,
)

COMMITTED_DATA = Path("data/synthetic/mmm_weekly_synthetic.csv")
COMMITTED_TRUTH = Path("data/synthetic/mmm_ground_truth.json")


def test_generation_is_reproducible_for_a_seed() -> None:
    first, first_truth = generate_synthetic_mmm(SyntheticMMMConfig(seed=7))
    second, second_truth = generate_synthetic_mmm(SyntheticMMMConfig(seed=7))

    pd.testing.assert_frame_equal(first, second)
    assert first_truth["true_contributions"] == second_truth["true_contributions"]


def test_different_seeds_produce_different_data() -> None:
    first, _ = generate_synthetic_mmm(SyntheticMMMConfig(seed=7))
    second, _ = generate_synthetic_mmm(SyntheticMMMConfig(seed=8))

    assert not first["simulated_sales"].equals(second["simulated_sales"])


def test_every_row_is_marked_synthetic() -> None:
    frame, truth = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=30))

    assert (frame["data_type"] == DATA_TYPE).all()
    assert truth["is_square_enix_data"] is False
    assert truth["statement"] == "This is not Square Enix data."
    assert "SYNTHETIC" in truth["notice"]


def test_ground_truth_records_the_generating_parameters() -> None:
    _frame, truth = generate_synthetic_mmm(SyntheticMMMConfig(seed=11, n_weeks=40))

    config = truth["config"]
    assert config["seed"] == 11
    assert config["n_weeks"] == 40
    assert len(config["channels"]) == 3
    for channel in config["channels"]:
        assert {"adstock_decay", "saturation_alpha", "saturation_theta", "beta"} <= set(channel)
    assert truth["expected_price_direction"] == "negative"
    assert set(truth["true_contributions"]) == {
        "video_spend", "paid_search_spend", "social_spend"
    }


def test_generated_series_are_well_formed() -> None:
    frame, _ = generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=60))

    assert len(frame) == 60
    assert (frame[["video_spend", "paid_search_spend", "social_spend"]] >= 0).all().all()
    assert (frame["price"] > 0).all()
    assert frame["promotion"].isin([0, 1]).all()
    assert frame["event"].isin([0.0, 1.0]).all()
    assert np.isfinite(frame["simulated_sales"]).all()


def test_correlated_channels_option_increases_collinearity() -> None:
    independent, _ = generate_synthetic_mmm(SyntheticMMMConfig(seed=3, correlated_channels=False))
    correlated, _ = generate_synthetic_mmm(SyntheticMMMConfig(seed=3, correlated_channels=True))

    channels = ["video_spend", "paid_search_spend", "social_spend"]

    def _mean_correlation(frame):
        matrix = frame[channels].corr().to_numpy()
        return float(np.mean(matrix[np.triu_indices(3, k=1)]))

    assert _mean_correlation(correlated) > _mean_correlation(independent)


def test_too_short_a_dataset_is_refused() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_mmm(SyntheticMMMConfig(n_weeks=5))


def test_committed_dataset_matches_the_generator() -> None:
    """The committed CSV must be exactly what the recorded seed produces."""
    committed = pd.read_csv(COMMITTED_DATA)
    truth = json.loads(COMMITTED_TRUTH.read_text())
    config = truth["config"]

    regenerated, _ = generate_synthetic_mmm(
        SyntheticMMMConfig(
            seed=config["seed"],
            n_weeks=config["n_weeks"],
            correlated_channels=config["correlated_channels"],
        )
    )

    pd.testing.assert_frame_equal(
        committed.reset_index(drop=True),
        regenerated.reset_index(drop=True),
        check_dtype=False,
    )


def test_committed_synthetic_data_is_clearly_labelled() -> None:
    committed = pd.read_csv(COMMITTED_DATA)
    readme = (COMMITTED_DATA.parent / "README.md").read_text()

    assert (committed["data_type"] == "synthetic").all()
    assert "not Square Enix data" in readme
    assert "SYNTHETIC" in readme


def test_synthetic_data_lives_apart_from_the_real_signal_data() -> None:
    """Synthetic marketing data must never sit in the real-data directories."""
    assert DEFAULT_DATA_PATH.parts[:2] == ("data", "synthetic")
    assert DEFAULT_TRUTH_PATH.parts[:2] == ("data", "synthetic")
    assert "synthetic" not in str(Path("data/processed"))


class _StubFit:
    """A fitted-model stand-in with a known linear response, for scenario tests."""

    channels = ["video_spend", "social_spend"]
    training_ranges = {
        "video_spend": {"min": 0.0, "max": 100.0, "mean": 50.0},
        "social_spend": {"min": 0.0, "max": 100.0, "mean": 50.0},
        "price": {"min": 40.0, "max": 60.0},
        "total_spend": {"min": 0.0, "max": 200.0, "mean": 100.0},
    }


def _stub_data(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_spend": np.full(rows, 50.0),
            "social_spend": np.full(rows, 50.0),
            "price": np.full(rows, 50.0),
            "promotion": np.zeros(rows),
            "event": np.zeros(rows),
        }
    )


def test_budget_reallocation_moves_spend_between_channels() -> None:
    scenario = budget_reallocation("social_spend", "video_spend", 0.20)
    plan = apply_scenario(_stub_data(), scenario, _StubFit.channels)

    assert plan["social_spend"].iloc[0] == pytest.approx(40.0)
    assert plan["video_spend"].iloc[0] == pytest.approx(60.0)
    assert "social_spend" in scenario.label and "video_spend" in scenario.label


def test_price_promotion_and_event_changes_apply() -> None:
    scenario = MarketingScenario(price_change=-5.0, promotion=True, event=True)

    plan = apply_scenario(_stub_data(), scenario, _StubFit.channels)

    assert plan["price"].iloc[0] == pytest.approx(45.0)
    assert (plan["promotion"] == 1.0).all()
    assert (plan["event"] == 1.0).all()


def test_unknown_channel_or_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown channel"):
        apply_scenario(_stub_data(), MarketingScenario({"tv_spend": 1.2}), _StubFit.channels)
    with pytest.raises(ValueError, match="negative"):
        apply_scenario(_stub_data(), MarketingScenario({"video_spend": -1.0}), _StubFit.channels)


def test_reallocation_fraction_must_be_a_proper_fraction() -> None:
    for fraction in [0.0, 1.0, 1.5, -0.2]:
        with pytest.raises(ValueError):
            budget_reallocation("social_spend", "video_spend", fraction)


def test_guardrail_warns_when_a_plan_leaves_the_observed_spend_range() -> None:
    plan = apply_scenario(
        _stub_data(), MarketingScenario({"video_spend": 4.0}), _StubFit.channels
    )

    warnings = scenario_warnings(_StubFit(), plan, MarketingScenario({"video_spend": 4.0}))

    assert warnings and "extrapolation" in warnings[0]


def test_guardrail_stays_quiet_inside_the_observed_range() -> None:
    plan = apply_scenario(
        _stub_data(), MarketingScenario({"video_spend": 1.1}), _StubFit.channels
    )

    assert scenario_warnings(_StubFit(), plan, MarketingScenario({"video_spend": 1.1})) == []


def test_guardrail_warns_on_out_of_range_pricing() -> None:
    scenario = MarketingScenario(price_change=-30.0)
    plan = apply_scenario(_stub_data(), scenario, _StubFit.channels)

    warnings = scenario_warnings(_StubFit(), plan, scenario)

    assert any("price" in warning.lower() for warning in warnings)


def test_scenario_result_is_labelled_synthetic_and_reports_uncertainty(monkeypatch) -> None:
    import src.mmm.scenarios as scenarios_module

    def _predict(fit, data, n_draws=None):
        level = data[["video_spend", "social_spend"]].to_numpy().sum(axis=1)
        rng = np.random.default_rng(0)
        return level[None, :] + rng.normal(0, 1, (50, len(data)))

    monkeypatch.setattr(scenarios_module, "predict_sales", _predict)
    monkeypatch.setattr(
        scenarios_module,
        "channel_contributions",
        lambda fit, data, n_draws=None: {
            channel: float(data[channel].sum()) for channel in fit.channels
        },
    )

    result = simulate_scenario(
        _StubFit(), _stub_data(), MarketingScenario({"video_spend": 1.2}, label="more_video")
    )

    assert result["data_type"] == "synthetic"
    assert result["notice"] == SYNTHETIC_NOTICE
    assert "simulated" in result["interpretation"]
    assert result["difference_lower"] <= result["difference_mean"] <= result["difference_upper"]
    assert result["channel_contribution_change"]["video_spend"] > 0
    assert result["channel_contribution_change"]["social_spend"] == pytest.approx(0.0)

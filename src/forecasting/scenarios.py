"""Behavioural-demand scenario analysis on the real-data Bayesian forecast.

This is a *behavioural* scenario tool: it asks what the observable demand
proxy would look like if public discussion behaved differently. It is not a
marketing-budget simulator, and it does not model spend, price or promotion —
that lives in the separate synthetic MMM lab and uses synthetic data.

Scenarios operate on the posterior predictive draws, so every answer carries
its uncertainty rather than a single point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

#: Beyond this multiple of the observed range, a scenario is extrapolation.
DEFAULT_TOLERANCE = 1.5


@dataclass
class BehaviouralScenario:
    """A hypothetical change to future behavioural demand.

    Attributes:
        level_shift: Multiplier on the whole forecast (1.2 = 20% more).
        event_pulse: Extra multiplier applied to the first ``pulse_periods``.
        pulse_periods: How many periods the event pulse lasts.
        trajectory: Per-period compounding growth multiplier.
    """

    level_shift: float = 1.0
    event_pulse: float = 1.0
    pulse_periods: int = 1
    trajectory: float = 1.0
    label: str = "scenario"

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.level_shift <= 0 or self.event_pulse <= 0 or self.trajectory <= 0:
            raise ValueError("Scenario multipliers must be positive")
        if self.pulse_periods < 0:
            raise ValueError("pulse_periods cannot be negative")


def _multipliers(scenario: BehaviouralScenario, horizon: int) -> np.ndarray:
    steps = np.arange(horizon)
    multiplier = np.full(horizon, scenario.level_shift, dtype=float)
    multiplier *= scenario.trajectory**steps
    if scenario.pulse_periods > 0 and scenario.event_pulse != 1.0:
        pulse = np.ones(horizon)
        pulse[: min(scenario.pulse_periods, horizon)] = scenario.event_pulse
        multiplier *= pulse
    return multiplier


def scenario_guardrails(
    scenario: BehaviouralScenario,
    observed,
    baseline_mean,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[str]:
    """Warn when a scenario pushes outside the range the data can speak to."""
    warnings: list[str] = []
    observed_array = np.asarray(observed, dtype=float)
    observed_array = observed_array[np.isfinite(observed_array)]
    if observed_array.size == 0:
        return ["No observed history available to judge whether this scenario extrapolates."]

    observed_max = float(np.max(observed_array))
    scenario_peak = float(np.max(np.asarray(baseline_mean, dtype=float)))
    if observed_max > 0 and scenario_peak > observed_max * tolerance:
        warnings.append(
            f"Scenario peak ({scenario_peak:.2f}) exceeds {tolerance:.1f}x the largest "
            f"observed period ({observed_max:.2f}). The model has never seen behaviour "
            "at this level, so the result is extrapolation, not evidence."
        )
    for name, value in [
        ("level_shift", scenario.level_shift),
        ("event_pulse", scenario.event_pulse),
    ]:
        if value > 3.0 or value < 0.33:
            warnings.append(
                f"{name} of {value:.2f} is a large behavioural change; treat the "
                "result as illustrative."
            )
    return warnings


def run_behavioural_scenario(
    baseline_draws,
    scenario: BehaviouralScenario,
    observed=None,
    interval_level: float = 0.8,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Compare baseline and scenario posterior predictive forecasts.

    ``baseline_draws`` has shape (draws, horizon). The scenario is applied to
    every draw, so the reported difference keeps its uncertainty.
    """
    scenario.validate()
    draws = np.asarray(baseline_draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError("baseline_draws must have shape (draws, horizon)")
    horizon = draws.shape[1]

    scenario_draws = draws * _multipliers(scenario, horizon)[None, :]
    difference = scenario_draws - draws
    tail = (1.0 - interval_level) / 2.0

    result = {
        "scenario": scenario.to_dict(),
        "horizon": horizon,
        "interval_level": interval_level,
        "baseline_mean": draws.mean(axis=0).tolist(),
        "baseline_lower": np.quantile(draws, tail, axis=0).tolist(),
        "baseline_upper": np.quantile(draws, 1 - tail, axis=0).tolist(),
        "scenario_mean": scenario_draws.mean(axis=0).tolist(),
        "scenario_lower": np.quantile(scenario_draws, tail, axis=0).tolist(),
        "scenario_upper": np.quantile(scenario_draws, 1 - tail, axis=0).tolist(),
        "difference_mean": difference.mean(axis=0).tolist(),
        "difference_lower": np.quantile(difference, tail, axis=0).tolist(),
        "difference_upper": np.quantile(difference, 1 - tail, axis=0).tolist(),
        "total_baseline": float(draws.sum(axis=1).mean()),
        "total_scenario": float(scenario_draws.sum(axis=1).mean()),
        "total_difference_mean": float(difference.sum(axis=1).mean()),
        "total_difference_lower": float(np.quantile(difference.sum(axis=1), tail)),
        "total_difference_upper": float(np.quantile(difference.sum(axis=1), 1 - tail)),
        "interpretation": (
            "Change in the expected actionable-signal volume (behavioural demand "
            "proxy) under the stated hypothetical. Not a sales or revenue forecast."
        ),
    }
    result["warnings"] = (
        scenario_guardrails(scenario, observed, result["scenario_mean"], tolerance)
        if observed is not None
        else []
    )
    return result


__all__ = [
    "BehaviouralScenario",
    "DEFAULT_TOLERANCE",
    "run_behavioural_scenario",
    "scenario_guardrails",
]

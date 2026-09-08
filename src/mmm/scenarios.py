"""Budget-reallocation scenarios on the fitted synthetic MMM.

SYNTHETIC MARKETING SCIENCE DEMONSTRATION. Every number produced here refers
to simulated sales in a simulated market. It is not a forecast of Square Enix
performance, and it must not be read as one.

A scenario changes spend, price, promotion or event status, pushes the new
plan back through the model's own adstock and saturation curves, and reports
the difference with its uncertainty. Because saturation is non-linear, moving
budget between channels does not move sales proportionally — which is the
point of asking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.mmm import SYNTHETIC_NOTICE
from src.mmm.model import channel_contributions, predict_sales

#: Beyond this multiple of the observed spend range, a plan is extrapolation.
DEFAULT_TOLERANCE = 1.25


@dataclass
class MarketingScenario:
    """A hypothetical marketing plan expressed as changes to the baseline.

    Attributes:
        budget_changes: Per-channel multipliers, e.g. {"social_spend": 0.8,
            "video_spend": 1.2} to move a fifth of social budget into video.
        price_change: Additive change to price in currency units.
        promotion: Force promotion on (True) or off (False); None keeps it.
        event: Force event weeks on or off; None keeps them.
    """

    budget_changes: dict = field(default_factory=dict)
    price_change: float = 0.0
    promotion: bool | None = None
    event: bool | None = None
    label: str = "scenario"

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self, channels) -> None:
        for channel, multiplier in self.budget_changes.items():
            if channel not in channels:
                raise ValueError(f"Unknown channel {channel!r}; known channels: {list(channels)}")
            if multiplier < 0:
                raise ValueError(f"Budget multiplier for {channel} cannot be negative")


def apply_scenario(data: pd.DataFrame, scenario: MarketingScenario, channels) -> pd.DataFrame:
    """Return a copy of the plan with the scenario's changes applied."""
    scenario.validate(channels)
    plan = data.copy()
    for channel, multiplier in scenario.budget_changes.items():
        plan[channel] = plan[channel] * float(multiplier)
    if scenario.price_change:
        plan["price"] = np.clip(plan["price"] + scenario.price_change, 0.01, None)
    if scenario.promotion is not None:
        plan["promotion"] = 1.0 if scenario.promotion else 0.0
    if scenario.event is not None:
        plan["event"] = 1.0 if scenario.event else 0.0
    return plan


def scenario_warnings(
    fit, plan: pd.DataFrame, scenario: MarketingScenario, tolerance: float = DEFAULT_TOLERANCE
) -> list[str]:
    """Warn when a plan leaves the spend or price range the model was fitted on."""
    warnings: list[str] = []
    ranges = fit.training_ranges
    for channel in fit.channels:
        if channel not in ranges:
            continue
        observed_max = ranges[channel]["max"]
        planned_max = float(plan[channel].max())
        if observed_max > 0 and planned_max > observed_max * tolerance:
            warnings.append(
                f"{channel} reaches {planned_max:,.0f}, beyond {tolerance:.2f}x the "
                f"largest level in the training data ({observed_max:,.0f}). The "
                "saturation curve is unconstrained out there, so this is "
                "extrapolation rather than an estimate."
            )
    price_range = ranges.get("price", {})
    if price_range:
        planned_price = float(plan["price"].mean())
        if planned_price < price_range["min"] * 0.8 or planned_price > price_range["max"] * 1.2:
            warnings.append(
                f"Average price {planned_price:,.2f} sits outside the observed range "
                f"({price_range['min']:,.2f}-{price_range['max']:,.2f}); the price "
                "effect is being extrapolated."
            )
    return warnings


def simulate_scenario(
    fit,
    data: pd.DataFrame,
    scenario: MarketingScenario,
    interval_level: float = 0.9,
    n_draws: int = 200,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Compare baseline and scenario expected simulated sales, with uncertainty."""
    plan = apply_scenario(data, scenario, fit.channels)

    baseline_draws = predict_sales(fit, data, n_draws=n_draws).sum(axis=1)
    scenario_draws = predict_sales(fit, plan, n_draws=n_draws).sum(axis=1)
    difference = scenario_draws - baseline_draws

    tail = (1.0 - interval_level) / 2.0
    baseline_contributions = channel_contributions(fit, data, n_draws=min(n_draws, 100))
    scenario_contributions = channel_contributions(fit, plan, n_draws=min(n_draws, 100))

    baseline_spend = float(data[list(fit.channels)].to_numpy().sum())
    scenario_spend = float(plan[list(fit.channels)].to_numpy().sum())

    return {
        "notice": SYNTHETIC_NOTICE,
        "data_type": "synthetic",
        "scenario": scenario.to_dict(),
        "interval_level": interval_level,
        "baseline_total_spend": baseline_spend,
        "scenario_total_spend": scenario_spend,
        "spend_change": scenario_spend - baseline_spend,
        "baseline_expected_sales": float(np.mean(baseline_draws)),
        "scenario_expected_sales": float(np.mean(scenario_draws)),
        "difference_mean": float(np.mean(difference)),
        "difference_lower": float(np.quantile(difference, tail)),
        "difference_upper": float(np.quantile(difference, 1 - tail)),
        "difference_percent": float(np.mean(difference) / np.mean(baseline_draws) * 100),
        "probability_of_increase": float(np.mean(difference > 0)),
        "channel_contributions_baseline": baseline_contributions,
        "channel_contributions_scenario": scenario_contributions,
        "channel_contribution_change": {
            channel: scenario_contributions[channel] - baseline_contributions[channel]
            for channel in fit.channels
        },
        "warnings": scenario_warnings(fit, plan, scenario, tolerance),
        "interpretation": (
            "Change in expected simulated sales under the stated plan, in a "
            "simulated market. Not a forecast of real commercial performance."
        ),
    }


def budget_reallocation(
    from_channel: str, to_channel: str, fraction: float, label: str | None = None
) -> MarketingScenario:
    """Build the canonical 'move X% of one channel's budget into another' scenario."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    return MarketingScenario(
        budget_changes={from_channel: 1.0 - fraction, to_channel: 1.0 + fraction},
        label=label or f"move_{int(fraction * 100)}pct_{from_channel}_to_{to_channel}",
    )


__all__ = [
    "DEFAULT_TOLERANCE",
    "MarketingScenario",
    "apply_scenario",
    "budget_reallocation",
    "scenario_warnings",
    "simulate_scenario",
]

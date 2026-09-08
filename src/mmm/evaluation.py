"""Does the MMM recover the process that generated the data?

The synthetic dataset exists so this question can be asked at all. On real
marketing data the truth is unobservable and a plausible-looking fit cannot be
checked; here the generator's parameters are known, so the model can be scored
on direction, ordering and shape — and its failures can be reported honestly
rather than tuned away.

Recovering a coefficient is not proof of causal media effectiveness. It only
shows the estimator can invert a known process under known conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _spearman(first, second) -> float:
    """Rank correlation without a SciPy dependency."""
    first_ranks = pd.Series(first).rank().to_numpy()
    second_ranks = pd.Series(second).rank().to_numpy()
    if first_ranks.size < 2 or np.std(first_ranks) == 0 or np.std(second_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def evaluate_recovery(fit, truth: dict) -> pd.DataFrame:
    """Score the fit against the known data-generating process."""
    checks: list[dict] = []
    posterior = fit.posterior

    def _add(name, passed, detail, estimated=None, expected=None):
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "estimated": estimated,
                "expected": expected,
                "detail": detail,
            }
        )

    price = float(np.mean(posterior["price_coefficient"]))
    _add(
        "price_direction",
        price < 0,
        "Sales should fall as price rises.",
        round(price, 2),
        truth.get("expected_price_direction", "negative"),
    )

    promotion = float(np.mean(posterior["promotion_coefficient"]))
    _add(
        "promotion_direction",
        promotion > 0,
        "A promotion should lift sales.",
        round(promotion, 2),
        truth.get("expected_promotion_direction", "positive"),
    )

    event = float(np.mean(posterior["event_coefficient"]))
    _add(
        "event_direction",
        event > 0,
        "An event week should lift sales.",
        round(event, 2),
        truth.get("expected_event_direction", "positive"),
    )

    _add(
        "media_effects_non_negative",
        bool((np.mean(posterior["beta"], axis=0) >= 0).all()),
        "Spending on a channel should not be estimated to reduce sales.",
        np.mean(posterior["beta"], axis=0).round(1).tolist(),
        ">= 0",
    )

    true_contributions = truth.get("true_contributions", {})
    estimated = fit.contributions.set_index("channel")["contribution_mean"].to_dict()
    shared = [channel for channel in fit.channels if channel in true_contributions]
    if len(shared) >= 2:
        correlation = _spearman(
            [true_contributions[c] for c in shared], [estimated[c] for c in shared]
        )
        _add(
            "contribution_ranking",
            correlation >= 0.5,
            "Broad ordering of channel contributions should be recovered.",
            round(correlation, 3),
            ">= 0.5 rank correlation",
        )

        true_top = max(shared, key=lambda c: true_contributions[c])
        estimated_top = max(shared, key=lambda c: estimated[c])
        _add(
            "largest_channel_identified",
            true_top == estimated_top,
            "The largest contributing channel should be identified.",
            estimated_top,
            true_top,
        )

        within_interval = 0
        for channel in shared:
            row = fit.contributions.set_index("channel").loc[channel]
            if row["contribution_hdi_5"] <= true_contributions[channel] <= row["contribution_hdi_95"]:
                within_interval += 1
        _add(
            "true_contribution_in_interval",
            within_interval == len(shared),
            "The true contribution should fall inside the 90% credible interval.",
            f"{within_interval}/{len(shared)}",
            f"{len(shared)}/{len(shared)}",
        )

    # Hill saturation with alpha > 1 is S-shaped: it accelerates from zero
    # before bending over, so global concavity is the wrong test — the true
    # generating process does not satisfy it either. The economic claim being
    # checked is that returns *diminish*: the curve rises throughout, and the
    # marginal return at high spend is below the marginal return mid-range.
    curves = fit.response_curves
    monotone = True
    saturating = True
    for channel in fit.channels:
        subset = curves[curves["channel"] == channel].sort_values("spend")
        response = subset["response_mean"].to_numpy()
        if response.size < 6:
            continue
        increments = np.diff(response)
        if not bool((increments >= -1e-9).all()):
            monotone = False
        mid = increments[len(increments) // 2]
        late = increments[-1]
        if not late <= mid + 1e-12:
            saturating = False
    _add(
        "diminishing_returns",
        monotone and saturating,
        "Response should rise with spend while its marginal return falls as "
        "spend grows (saturation).",
        f"monotone={monotone}, marginal return falls={saturating}",
        "monotone with falling marginal return",
    )

    true_decays = {
        channel["name"]: channel["adstock_decay"]
        for channel in truth.get("config", {}).get("channels", [])
    }
    if true_decays:
        recovered = 0
        for index, channel in enumerate(fit.channels):
            if channel not in true_decays:
                continue
            draws = posterior["decay"][:, index]
            low, high = np.quantile(draws, [0.05, 0.95])
            if low <= true_decays[channel] <= high:
                recovered += 1
        _add(
            "adstock_decay_in_interval",
            recovered >= max(1, len(true_decays) // 2),
            "Carryover decay should be recovered for most channels.",
            f"{recovered}/{len(true_decays)}",
            "majority",
        )

    return pd.DataFrame(checks)


def identifiability_notes(fit, truth: dict, recovery: pd.DataFrame) -> list[str]:
    """Explain what the recovery table says about what can and cannot be told apart."""
    notes: list[str] = []

    failed = recovery[~recovery["passed"]]["check"].tolist()
    if "largest_channel_identified" in failed or "contribution_ranking" in failed:
        notes.append(
            "Channel contributions are only partially identified. Channels whose "
            "spend moves together, or whose adstock and saturation shapes are "
            "similar, trade off against one another: the model can fit the data "
            "well while assigning credit between them differently from the truth."
        )

    contributions = fit.contributions
    widest = contributions.assign(
        width=contributions["contribution_hdi_95"] - contributions["contribution_hdi_5"]
    ).sort_values("width", ascending=False)
    top = widest.iloc[0]
    notes.append(
        f"Credible intervals on channel contribution are wide (widest: "
        f"{top['channel']}, {top['contribution_hdi_5']:,.0f} to "
        f"{top['contribution_hdi_95']:,.0f}). Point estimates of contribution "
        "should never be quoted without them."
    )

    if truth.get("config", {}).get("correlated_channels"):
        notes.append(
            "This run used deliberately correlated channel spends. Collinear media "
            "is the standard identifiability problem in marketing mix modelling, "
            "and the widened, overlapping intervals are the model correctly "
            "reporting that it cannot separate the channels."
        )

    notes.append(
        "Recovering a parameter from simulated data shows the estimator works on "
        "this process. It is not evidence of causal media effectiveness, which "
        "observational marketing data cannot establish on its own."
    )
    return notes


__all__ = ["evaluate_recovery", "identifiability_notes"]

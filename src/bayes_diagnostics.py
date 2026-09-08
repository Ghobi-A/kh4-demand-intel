"""Shared posterior diagnostics for the PyMC models in this project.

Diagnostics are always reported, including when they are bad. A model that
did not converge is more useful to a reader as a flagged failure than as a
clean-looking number.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

#: Conventional thresholds for flagging a fit as unreliable.
RHAT_WARNING = 1.01
ESS_WARNING = 400


def summarise_trace(trace, var_names: list[str] | None = None) -> dict:
    """Return R-hat, ESS, divergences and warnings for a fitted trace."""
    import arviz as az

    summary = az.summary(trace, var_names=var_names, round_to=6)
    numeric = summary.reset_index().rename(columns={"index": "parameter"})

    max_rhat = float(numeric["r_hat"].max()) if "r_hat" in numeric else float("nan")
    min_ess_bulk = float(numeric["ess_bulk"].min()) if "ess_bulk" in numeric else float("nan")
    min_ess_tail = float(numeric["ess_tail"].min()) if "ess_tail" in numeric else float("nan")

    divergences = 0
    if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
        divergences = int(trace.sample_stats["diverging"].values.sum())

    warnings: list[str] = []
    if max_rhat == max_rhat and max_rhat > RHAT_WARNING:
        warnings.append(
            f"Max R-hat {max_rhat:.3f} exceeds {RHAT_WARNING}: chains have not mixed, "
            "so posterior summaries are unreliable."
        )
    if min_ess_bulk == min_ess_bulk and min_ess_bulk < ESS_WARNING:
        warnings.append(
            f"Minimum bulk ESS {min_ess_bulk:.0f} is below {ESS_WARNING}: "
            "posterior estimates are noisy."
        )
    if divergences:
        warnings.append(
            f"{divergences} divergent transitions: the sampler could not explore "
            "part of the posterior, so estimates may be biased."
        )

    for message in warnings:
        LOGGER.warning("%s", message)

    return {
        "max_r_hat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "divergences": divergences,
        "converged": not warnings,
        "warnings": warnings,
        "parameters": numeric.to_dict(orient="records"),
    }


def posterior_predictive_summary(observed, predictive_samples) -> dict:
    """Compare observed data with posterior predictive draws."""
    import numpy as np

    observed_array = np.asarray(observed, dtype=float)
    samples = np.asarray(predictive_samples, dtype=float).reshape(-1, observed_array.size)
    if samples.size == 0:
        return {}
    predicted_mean = samples.mean(axis=0)
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    return {
        "observed_mean": float(np.nanmean(observed_array)),
        "predicted_mean": float(np.nanmean(predicted_mean)),
        "observed_std": float(np.nanstd(observed_array)),
        "predicted_std": float(np.nanstd(predicted_mean)),
        "ppc_interval_coverage": float(
            np.nanmean((observed_array >= lower) & (observed_array <= upper))
        ),
        "observed_zero_share": float(np.nanmean(observed_array <= 0)),
        "predicted_zero_share": float(np.nanmean(samples <= 0)),
    }


__all__ = ["ESS_WARNING", "RHAT_WARNING", "posterior_predictive_summary", "summarise_trace"]

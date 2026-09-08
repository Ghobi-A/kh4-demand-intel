"""Bayesian Marketing Mix Model in PyMC, fitted to synthetic data.

SYNTHETIC MARKETING SCIENCE DEMONSTRATION. Everything fitted here is
simulated; no coefficient describes real media effectiveness.

Structure:

    simulated_sales_t = baseline + trend
                        + sum over channels of beta_c * hill(adstock(spend_c))
                        + price_coefficient * centred_price_t
                        + promotion_coefficient * promotion_t
                        + event_coefficient * event_t
                        + seasonality (one Fourier pair)
                        + noise

Each channel's spend passes through the same two transforms as the generator —
geometric adstock then Hill saturation — with the decay, shape and
half-saturation point all estimated rather than fixed.

A fitted coefficient here is a property of this simulation. It does not prove
causal media effectiveness even on synthetic data, and certainly not on real
marketing.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.bayes_diagnostics import posterior_predictive_summary, summarise_trace
from src.mmm.synthetic import CHANNELS
from src.mmm.transforms import geometric_adstock, hill_saturation

LOGGER = logging.getLogger(__name__)


class MMMUnavailableError(RuntimeError):
    """Raised when PyMC is not installed in this environment."""


def _require_pymc():
    try:
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError as exc:  # pragma: no cover - only without the forecast extra
        raise MMMUnavailableError(
            "PyMC is not installed. Install the forecasting extra: pip install -e .[forecast]"
        ) from exc
    return pm, pt


@dataclass
class MMMSamplerConfig:
    """Sampler settings, recorded with every fit."""

    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    # The media transforms create a mildly awkward posterior geometry, so this
    # runs tighter than the usual 0.9 to keep divergences down.
    target_accept: float = 0.95
    seed: int = 42
    progressbar: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MMMFit:
    """A fitted MMM: posterior summaries plus what is needed to simulate."""

    channels: list
    posterior: dict
    diagnostics: dict
    contributions: pd.DataFrame
    response_curves: pd.DataFrame
    posterior_predictive: dict = field(default_factory=dict)
    sampler: dict = field(default_factory=dict)
    training_ranges: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    data_type: str = "synthetic"


#: Lags kept when expressing adstock as a convolution. The geometric series is
#: infinite, but decay**MAX_ADSTOCK_LAG is negligible for the decays this model
#: entertains, and a fixed-width convolution is both differentiable and fast.
MAX_ADSTOCK_LAG = 24


def _pt_adstock(pt, spend, decay, n_periods: int):
    """Geometric adstock in PyTensor as a truncated convolution.

    Equivalent to the recursive form in ``src.mmm.transforms`` up to the
    truncation: A_t = sum_k decay**k * X_(t-k), normalised by (1 - decay) so
    the result keeps the scale of the original spend.
    """
    lags = min(MAX_ADSTOCK_LAG, n_periods)
    padded = pt.concatenate([pt.zeros(lags), spend])
    shifted = pt.stack(
        [padded[lags - lag : lags - lag + n_periods] for lag in range(lags)], axis=0
    )
    weights = pt.power(decay, pt.arange(lags))
    return pt.tensordot(weights, shifted, axes=[[0], [0]]) * (1.0 - decay)


def _pt_hill(pt, x, alpha, theta):
    """Hill saturation in PyTensor."""
    powered = pt.power(pt.clip(x, 1e-9, np.inf), alpha)
    return powered / (powered + pt.power(theta, alpha))


def fit_mmm(
    data: pd.DataFrame,
    channels: list | None = None,
    target: str = "simulated_sales",
    config: MMMSamplerConfig | None = None,
) -> MMMFit:
    """Fit the Bayesian MMM to a synthetic weekly dataset."""
    pm, pt = _require_pymc()
    config = config or MMMSamplerConfig()
    channels = list(channels or CHANNELS)

    if data.get("data_type", pd.Series(["synthetic"])).iloc[0] != "synthetic":
        raise ValueError(
            "The MMM is a synthetic demonstration and refuses non-synthetic input. "
            "Real KH4 signals are modelled by src.forecasting, not here."
        )

    missing = [column for column in [*channels, "price", "promotion", "event", target] if column not in data.columns]
    if missing:
        raise ValueError(f"Synthetic dataset is missing columns: {missing}")
    if len(data) < 24:
        raise ValueError(f"Need at least 24 weeks to fit an MMM, got {len(data)}")

    n = len(data)
    weeks = np.arange(n)
    spend = data[channels].to_numpy(dtype=float)
    if (spend < 0).any():
        raise ValueError("Spend cannot be negative")

    # Scale each channel by its own mean so priors mean the same thing for a
    # channel spending 20k a week and one spending 2k.
    spend_scale = spend.mean(axis=0)
    spend_scale[spend_scale <= 0] = 1.0
    scaled_spend = spend / spend_scale

    sales = data[target].to_numpy(dtype=float)
    sales_scale = float(np.mean(sales))
    price = data["price"].to_numpy(dtype=float)
    price_centred = price - float(np.mean(price))
    promotion = data["promotion"].to_numpy(dtype=float)
    event = data["event"].to_numpy(dtype=float)
    seasonal_sin = np.sin(2 * np.pi * weeks / 52.0)
    seasonal_cos = np.cos(2 * np.pi * weeks / 52.0)

    with pm.Model():
        baseline = pm.Normal("baseline", mu=sales_scale, sigma=sales_scale * 0.5)
        trend = pm.Normal("trend", mu=0.0, sigma=sales_scale * 0.01)

        # Media effects are constrained non-negative: spending on a channel is
        # not assumed to reduce sales.
        beta = pm.HalfNormal("beta", sigma=sales_scale * 0.5, shape=len(channels))
        decay = pm.Beta("decay", alpha=2.0, beta=2.0, shape=len(channels))
        saturation_alpha = pm.Gamma("saturation_alpha", alpha=3.0, beta=2.0, shape=len(channels))
        saturation_theta = pm.Gamma("saturation_theta", alpha=3.0, beta=3.0, shape=len(channels))

        media_contributions = []
        for index, _channel in enumerate(channels):
            adstocked = _pt_adstock(pt, scaled_spend[:, index], decay[index], n)
            saturated = _pt_hill(pt, adstocked, saturation_alpha[index], saturation_theta[index])
            media_contributions.append(beta[index] * saturated)
        media_total = pt.sum(pt.stack(media_contributions, axis=0), axis=0)
        pm.Deterministic("media_contribution", pt.stack(media_contributions, axis=0))

        price_coefficient = pm.Normal("price_coefficient", mu=0.0, sigma=sales_scale * 0.05)
        promotion_coefficient = pm.Normal(
            "promotion_coefficient", mu=0.0, sigma=sales_scale * 0.3
        )
        event_coefficient = pm.Normal("event_coefficient", mu=0.0, sigma=sales_scale * 0.3)
        season_sin_coefficient = pm.Normal("season_sin", mu=0.0, sigma=sales_scale * 0.2)
        season_cos_coefficient = pm.Normal("season_cos", mu=0.0, sigma=sales_scale * 0.2)

        mu = (
            baseline
            + trend * weeks
            + media_total
            + price_coefficient * price_centred
            + promotion_coefficient * promotion
            + event_coefficient * event
            + season_sin_coefficient * seasonal_sin
            + season_cos_coefficient * seasonal_cos
        )
        sigma = pm.HalfNormal("sigma", sigma=sales_scale * 0.2)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=sales)

        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            random_seed=config.seed,
            progressbar=config.progressbar,
        )

    diagnostics = summarise_trace(
        trace,
        var_names=[
            "baseline", "trend", "beta", "decay", "saturation_alpha",
            "saturation_theta", "price_coefficient", "promotion_coefficient",
            "event_coefficient", "sigma",
        ],
    )

    posterior = {
        name: trace.posterior[name].values.reshape(-1, *trace.posterior[name].values.shape[2:])
        for name in [
            "baseline", "trend", "beta", "decay", "saturation_alpha", "saturation_theta",
            "price_coefficient", "promotion_coefficient", "event_coefficient",
            "season_sin", "season_cos", "sigma",
        ]
    }
    media_draws = trace.posterior["media_contribution"].values
    media_draws = media_draws.reshape(-1, *media_draws.shape[2:])

    contributions = _contribution_table(channels, media_draws, sales)
    curves = _response_curves(channels, posterior, spend, spend_scale)

    fitted = (
        posterior["baseline"][:, None]
        + posterior["trend"][:, None] * weeks[None, :]
        + media_draws.sum(axis=1)
        + posterior["price_coefficient"][:, None] * price_centred[None, :]
        + posterior["promotion_coefficient"][:, None] * promotion[None, :]
        + posterior["event_coefficient"][:, None] * event[None, :]
        + posterior["season_sin"][:, None] * seasonal_sin[None, :]
        + posterior["season_cos"][:, None] * seasonal_cos[None, :]
    )
    ppc = posterior_predictive_summary(sales, fitted)

    return MMMFit(
        channels=channels,
        posterior={**posterior, "spend_scale": spend_scale, "price_mean": float(np.mean(price))},
        diagnostics=diagnostics,
        contributions=contributions,
        response_curves=curves,
        posterior_predictive=ppc,
        sampler=config.to_dict(),
        training_ranges={
            **{
                channel: {
                    "min": float(spend[:, index].min()),
                    "max": float(spend[:, index].max()),
                    "mean": float(spend[:, index].mean()),
                }
                for index, channel in enumerate(channels)
            },
            "price": {"min": float(price.min()), "max": float(price.max())},
            "total_spend": {
                "min": float(spend.sum(axis=1).min()),
                "max": float(spend.sum(axis=1).max()),
                "mean": float(spend.sum(axis=1).mean()),
            },
        },
        notes=[
            "Fitted to synthetic data with known parameters. A coefficient here "
            "describes this simulation, not real media effectiveness.",
        ],
    )


def _contribution_table(channels, media_draws, sales) -> pd.DataFrame:
    """Posterior total contribution per channel, with credible intervals."""
    rows = []
    total_sales = float(np.sum(sales))
    for index, channel in enumerate(channels):
        totals = media_draws[:, index, :].sum(axis=1)
        rows.append(
            {
                "channel": channel,
                "contribution_mean": float(np.mean(totals)),
                "contribution_hdi_5": float(np.quantile(totals, 0.05)),
                "contribution_hdi_95": float(np.quantile(totals, 0.95)),
                "contribution_share_mean": float(np.mean(totals) / total_sales),
                "contribution_share_hdi_5": float(np.quantile(totals, 0.05) / total_sales),
                "contribution_share_hdi_95": float(np.quantile(totals, 0.95) / total_sales),
            }
        )
    return pd.DataFrame(rows).sort_values("contribution_mean", ascending=False).reset_index(
        drop=True
    )


def _response_curves(channels, posterior, spend, spend_scale, n_points: int = 25) -> pd.DataFrame:
    """Posterior mean response across a spend grid, per channel."""
    rows = []
    for index, channel in enumerate(channels):
        grid = np.linspace(0.0, float(spend[:, index].max()) * 1.2, n_points)
        scaled = grid / spend_scale[index]
        for level, scaled_level in zip(grid, scaled):
            # A steady-state spend level: adstock converges to the level itself
            # under the normalised transform.
            saturated = hill_saturation(
                np.array([scaled_level]),
                alpha=float(np.mean(posterior["saturation_alpha"][:, index])),
                theta=float(np.mean(posterior["saturation_theta"][:, index])),
            )[0]
            draws = posterior["beta"][:, index] * saturated
            rows.append(
                {
                    "channel": channel,
                    "spend": float(level),
                    "response_mean": float(np.mean(draws)),
                    "response_hdi_5": float(np.quantile(draws, 0.05)),
                    "response_hdi_95": float(np.quantile(draws, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def predict_sales(fit: MMMFit, data: pd.DataFrame, n_draws: int | None = None) -> np.ndarray:
    """Posterior expected sales for a spend/price/promotion plan.

    Used by the scenario simulator, so a reallocation is evaluated through the
    same adstock and saturation curves the model estimated.
    """
    channels = fit.channels
    posterior = fit.posterior
    spend = data[channels].to_numpy(dtype=float)
    weeks = np.arange(len(data))
    scale = posterior["spend_scale"]

    total_draws = posterior["baseline"].shape[0]
    n_draws = min(n_draws or total_draws, total_draws)
    index = np.linspace(0, total_draws - 1, n_draws).astype(int)

    price_centred = data["price"].to_numpy(dtype=float) - posterior["price_mean"]
    promotion = data["promotion"].to_numpy(dtype=float)
    event = data["event"].to_numpy(dtype=float)
    seasonal_sin = np.sin(2 * np.pi * weeks / 52.0)
    seasonal_cos = np.cos(2 * np.pi * weeks / 52.0)

    predictions = np.zeros((n_draws, len(data)))
    for position, draw in enumerate(index):
        media = np.zeros(len(data))
        for channel_index in range(len(channels)):
            adstocked = geometric_adstock(
                spend[:, channel_index] / scale[channel_index],
                decay=float(posterior["decay"][draw, channel_index]),
                normalise=True,
            )
            saturated = hill_saturation(
                adstocked,
                alpha=float(posterior["saturation_alpha"][draw, channel_index]),
                theta=float(posterior["saturation_theta"][draw, channel_index]),
            )
            media += posterior["beta"][draw, channel_index] * saturated
        predictions[position] = (
            posterior["baseline"][draw]
            + posterior["trend"][draw] * weeks
            + media
            + posterior["price_coefficient"][draw] * price_centred
            + posterior["promotion_coefficient"][draw] * promotion
            + posterior["event_coefficient"][draw] * event
            + posterior["season_sin"][draw] * seasonal_sin
            + posterior["season_cos"][draw] * seasonal_cos
        )
    return predictions


def channel_contributions(fit: MMMFit, data: pd.DataFrame, n_draws: int | None = None) -> dict:
    """Posterior mean contribution per channel for a given plan."""
    channels = fit.channels
    posterior = fit.posterior
    spend = data[channels].to_numpy(dtype=float)
    scale = posterior["spend_scale"]
    total_draws = posterior["baseline"].shape[0]
    n_draws = min(n_draws or total_draws, total_draws)
    index = np.linspace(0, total_draws - 1, n_draws).astype(int)

    result: dict[str, float] = {}
    for channel_index, channel in enumerate(channels):
        totals = []
        for draw in index:
            adstocked = geometric_adstock(
                spend[:, channel_index] / scale[channel_index],
                decay=float(posterior["decay"][draw, channel_index]),
                normalise=True,
            )
            saturated = hill_saturation(
                adstocked,
                alpha=float(posterior["saturation_alpha"][draw, channel_index]),
                theta=float(posterior["saturation_theta"][draw, channel_index]),
            )
            totals.append(float(np.sum(posterior["beta"][draw, channel_index] * saturated)))
        result[channel] = float(np.mean(totals))
    return result


__all__ = [
    "MMMFit",
    "MMMSamplerConfig",
    "MMMUnavailableError",
    "channel_contributions",
    "fit_mmm",
    "predict_sales",
]

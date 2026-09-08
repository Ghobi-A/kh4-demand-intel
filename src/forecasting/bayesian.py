"""Bayesian forecasting of behavioural demand proxies with PyMC.

The likelihood follows the target, not the other way round:

* ``actionable_probability_mass`` is continuous, non-negative, and legitimately
  zero in weeks with no discussion. It gets a **hurdle** model: a Bernoulli
  component for whether the week is non-zero, and a Gamma component for the
  level given that it is. The value is never rounded, never converted to a
  count, and no epsilon is added to force a Gamma to accept a zero.
* Integer counts (``total_comments``, ``actionable_count``) get a Negative
  Binomial, which admits the overdispersion these series show.
* Rates are modelled from their **counts**: ``actionable_count ~ Binomial(n =
  total_comments, p)``. Weeks with no comments carry no information about a
  rate and are dropped from the likelihood rather than read as 0%.

Model complexity follows measured history (see ``src.timestamps.complexity_tier``):
under 30 non-empty periods only an intercept, a linear trend and event effects
are identifiable, so nothing richer is offered.

PyMC is imported lazily so that reading generated reports never requires it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.bayes_diagnostics import posterior_predictive_summary, summarise_trace

LOGGER = logging.getLogger(__name__)

KIND_CONTINUOUS = "continuous_nonnegative"
KIND_COUNT = "count"
KIND_RATE = "rate"

TIER_MINIMAL = "minimal"
TIER_MODERATE = "moderate"
TIER_RICH = "rich"


class BayesianUnavailableError(RuntimeError):
    """Raised when PyMC is not installed in this environment."""


def _require_pymc():
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BayesianUnavailableError(
            "PyMC is not installed. Install the forecasting extra: pip install -e .[forecast]"
        ) from exc
    return pm


@dataclass
class SamplerConfig:
    """Sampler settings, recorded verbatim in run metadata."""

    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    target_accept: float = 0.9
    seed: int = 42
    progressbar: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BayesianForecastResult:
    """Posterior forecast plus everything needed to judge whether to trust it."""

    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    target: str
    likelihood: str
    tier: str
    diagnostics: dict = field(default_factory=dict)
    posterior_predictive: dict = field(default_factory=dict)
    effects: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    draws: np.ndarray | None = None
    model_name: str = "bayesian"
    interval_level: float = 0.8

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "horizon_step": np.arange(1, len(self.mean) + 1),
                "forecast": self.mean,
                "lower": self.lower,
                "upper": self.upper,
                "model": self.model_name,
            }
        )


def choose_likelihood(target_kind: str, values: np.ndarray) -> str:
    """Pick the likelihood family for a target, and say why in the name."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if target_kind == KIND_COUNT:
        return "negative_binomial"
    if target_kind == KIND_RATE:
        return "binomial"
    has_zeros = bool((finite <= 0).any())
    return "hurdle_gamma" if has_zeros else "gamma"


def _trend_and_events(pm, tier, n_periods, exog, time_index, prefix=""):
    """Shared linear predictor: intercept + trend + optional event effects."""
    intercept = pm.Normal(f"{prefix}intercept", mu=0.0, sigma=2.0)
    mu = intercept

    # A linear trend is the only trend shape a short history can identify;
    # richer dynamics are offered only at the higher tiers.
    trend_coefficient = pm.Normal(f"{prefix}trend", mu=0.0, sigma=0.5)
    mu = mu + trend_coefficient * time_index

    effects = {f"{prefix}intercept": intercept, f"{prefix}trend": trend_coefficient}

    if tier in (TIER_MODERATE, TIER_RICH):
        # A single Fourier pair: one annual seasonal shape, weakly regularised.
        season = 52.0
        sin_term = np.sin(2 * np.pi * np.arange(n_periods) / season)
        cos_term = np.cos(2 * np.pi * np.arange(n_periods) / season)
        seasonal_sin = pm.Normal(f"{prefix}seasonal_sin", mu=0.0, sigma=0.3)
        seasonal_cos = pm.Normal(f"{prefix}seasonal_cos", mu=0.0, sigma=0.3)
        mu = mu + seasonal_sin * sin_term + seasonal_cos * cos_term
        effects[f"{prefix}seasonal_sin"] = seasonal_sin
        effects[f"{prefix}seasonal_cos"] = seasonal_cos

    if exog is not None and np.asarray(exog).size:
        exog_array = np.asarray(exog, dtype=float)
        event_coefficients = pm.Normal(
            f"{prefix}event_effect", mu=0.0, sigma=1.0, shape=exog_array.shape[1]
        )
        mu = mu + pm.math.dot(exog_array, event_coefficients)
        effects[f"{prefix}event_effect"] = event_coefficients

    return mu, effects


def fit_bayesian_forecast(
    y,
    horizon: int,
    target: str = "actionable_probability_mass",
    target_kind: str = KIND_CONTINUOUS,
    tier: str = TIER_MINIMAL,
    exog: np.ndarray | None = None,
    exog_future: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    exposure_future: np.ndarray | None = None,
    config: SamplerConfig | None = None,
    interval_level: float = 0.8,
) -> BayesianForecastResult:
    """Fit a Bayesian forecast model and return posterior predictive forecasts."""
    pm = _require_pymc()
    config = config or SamplerConfig()

    values = np.asarray(pd.Series(y).astype(float).to_numpy(), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Target series contains non-finite values")
    if values.size < 8:
        raise ValueError(f"Need at least 8 observations to fit a Bayesian model, got {values.size}")
    if (values < 0).any():
        raise ValueError("Target series contains negative values; check the target definition")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    likelihood = choose_likelihood(target_kind, values)
    notes: list[str] = []
    n = values.size

    # Time is scaled so the trend prior means the same thing on any history length.
    scale = max(n - 1, 1)
    time_index = np.arange(n) / scale
    future_time = (np.arange(n, n + horizon)) / scale

    if exog is not None:
        exog = np.asarray(exog, dtype=float)[:n]
    if exog_future is not None:
        exog_future = np.asarray(exog_future, dtype=float)[:horizon]
        if exog is not None and exog_future.shape[1] != exog.shape[1]:
            raise ValueError("Future exogenous features must match the training feature count")

    if likelihood == "binomial":
        return _fit_binomial(
            pm, values, exposure, exposure_future, horizon, target, tier,
            exog, exog_future, time_index, future_time, config, interval_level, notes,
        )
    if likelihood == "negative_binomial":
        return _fit_negative_binomial(
            pm, values, horizon, target, tier, exog, exog_future,
            time_index, future_time, config, interval_level, notes,
        )
    return _fit_continuous(
        pm, values, horizon, target, tier, likelihood, exog, exog_future,
        time_index, future_time, config, interval_level, notes,
    )


def _sample(pm, model, config):
    with model:
        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            random_seed=config.seed,
            progressbar=config.progressbar,
            compute_convergence_checks=True,
        )
    return trace


def _posterior_flat(trace, name: str) -> np.ndarray:
    return trace.posterior[name].values.reshape(-1, *trace.posterior[name].values.shape[2:])


def _interval(draws: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    tail = (1.0 - level) / 2.0
    return np.quantile(draws, tail, axis=0), np.quantile(draws, 1.0 - tail, axis=0)


def _fit_continuous(
    pm, values, horizon, target, tier, likelihood, exog, exog_future,
    time_index, future_time, config, interval_level, notes,
):
    """Gamma, or a hurdle Gamma when the series contains genuine zeros."""
    positive_mask = values > 0
    n_zero = int((~positive_mask).sum())

    if likelihood == "hurdle_gamma":
        notes.append(
            f"{n_zero} of {values.size} periods are zero, so a hurdle model is used: "
            "a Bernoulli component for whether a period is non-zero and a Gamma "
            "component for its level. The target is never rounded and no epsilon "
            "is added to force a Gamma to accept a zero."
        )
    else:
        notes.append(
            "No zero periods are present, so a plain Gamma likelihood is used for "
            "this continuous non-negative target."
        )

    with pm.Model() as model:
        if likelihood == "hurdle_gamma":
            logit_mu, _ = _trend_and_events(
                pm, tier, values.size, exog, time_index, prefix="zero_"
            )
            probability = pm.Deterministic("p_nonzero", pm.math.sigmoid(logit_mu))
            pm.Bernoulli("is_nonzero", p=probability, observed=positive_mask.astype(int))

        log_mu, effects = _trend_and_events(pm, tier, values.size, exog, time_index)
        # Centre the level prior on the observed positive scale so the intercept
        # prior is weakly informative rather than accidentally strong.
        positive_values = values[positive_mask]
        offset = float(np.log(np.mean(positive_values))) if positive_values.size else 0.0
        mu = pm.Deterministic("mu", pm.math.exp(log_mu + offset))
        shape = pm.HalfNormal("gamma_shape", sigma=5.0)

        if likelihood == "hurdle_gamma":
            pm.Gamma(
                "positive_obs",
                alpha=shape,
                beta=shape / mu[positive_mask],
                observed=positive_values,
            )
        else:
            pm.Gamma("obs", alpha=shape, beta=shape / mu, observed=values)

        trace = _sample(pm, model, config)

    diagnostics = summarise_trace(trace)

    intercept = _posterior_flat(trace, "intercept")
    trend = _posterior_flat(trace, "trend")
    log_future = intercept[:, None] + trend[:, None] * future_time[None, :]
    log_future = _add_seasonal(trace, log_future, tier, values.size, horizon)
    log_future = _add_events(trace, log_future, exog_future)
    positive_values = values[positive_mask]
    offset = float(np.log(np.mean(positive_values))) if positive_values.size else 0.0
    mu_future = np.exp(log_future + offset)

    shape_draws = _posterior_flat(trace, "gamma_shape")[:, None]
    rng = np.random.default_rng(config.seed)
    level_draws = rng.gamma(shape=shape_draws, scale=mu_future / shape_draws)

    if likelihood == "hurdle_gamma":
        zero_intercept = _posterior_flat(trace, "zero_intercept")
        zero_trend = _posterior_flat(trace, "zero_trend")
        logit_future = zero_intercept[:, None] + zero_trend[:, None] * future_time[None, :]
        logit_future = _add_seasonal(trace, logit_future, tier, values.size, horizon, prefix="zero_")
        logit_future = _add_events(trace, logit_future, exog_future, name="zero_event_effect")
        probability_future = 1.0 / (1.0 + np.exp(-logit_future))
        occurs = rng.random(probability_future.shape) < probability_future
        forecast_draws = level_draws * occurs
    else:
        forecast_draws = level_draws

    in_sample = np.exp(
        _posterior_flat(trace, "intercept")[:, None]
        + _posterior_flat(trace, "trend")[:, None] * time_index[None, :]
        + offset
    )
    ppc = posterior_predictive_summary(values, in_sample)

    lower, upper = _interval(forecast_draws, interval_level)
    return BayesianForecastResult(
        mean=forecast_draws.mean(axis=0),
        lower=lower,
        upper=upper,
        target=target,
        likelihood=likelihood,
        tier=tier,
        diagnostics=diagnostics,
        posterior_predictive=ppc,
        effects=_effect_summary(trace),
        notes=notes,
        draws=forecast_draws,
    )


def _fit_negative_binomial(
    pm, values, horizon, target, tier, exog, exog_future,
    time_index, future_time, config, interval_level, notes,
):
    """Negative Binomial for integer count targets with overdispersion."""
    counts = np.rint(values).astype(int)
    if not np.allclose(counts, values):
        raise ValueError(
            f"Target {target!r} was routed to a count likelihood but is not integer-valued. "
            "Continuous targets must use the Gamma or hurdle-Gamma family."
        )
    dispersion_ratio = float(np.var(counts) / np.mean(counts)) if np.mean(counts) > 0 else 1.0
    notes.append(
        f"Integer count target; variance/mean = {dispersion_ratio:.2f}, so a Negative "
        "Binomial is used to admit overdispersion a Poisson would not."
    )
    offset = float(np.log(max(np.mean(counts), 1e-6)))

    with pm.Model() as model:
        log_mu, _effects = _trend_and_events(pm, tier, counts.size, exog, time_index)
        mu = pm.Deterministic("mu", pm.math.exp(log_mu + offset))
        alpha = pm.HalfNormal("nb_alpha", sigma=5.0)
        pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=counts)
        trace = _sample(pm, model, config)

    diagnostics = summarise_trace(trace)
    intercept = _posterior_flat(trace, "intercept")
    trend = _posterior_flat(trace, "trend")
    log_future = intercept[:, None] + trend[:, None] * future_time[None, :]
    log_future = _add_seasonal(trace, log_future, tier, counts.size, horizon)
    log_future = _add_events(trace, log_future, exog_future)
    mu_future = np.exp(log_future + offset)

    alpha_draws = _posterior_flat(trace, "nb_alpha")[:, None]
    rng = np.random.default_rng(config.seed)
    probability = alpha_draws / (alpha_draws + mu_future)
    forecast_draws = rng.negative_binomial(alpha_draws * np.ones_like(mu_future), probability)

    in_sample_mu = np.exp(
        intercept[:, None] + trend[:, None] * time_index[None, :] + offset
    )
    ppc = posterior_predictive_summary(counts, in_sample_mu)

    lower, upper = _interval(forecast_draws, interval_level)
    return BayesianForecastResult(
        mean=forecast_draws.mean(axis=0),
        lower=lower,
        upper=upper,
        target=target,
        likelihood="negative_binomial",
        tier=tier,
        diagnostics=diagnostics,
        posterior_predictive=ppc,
        effects=_effect_summary(trace),
        notes=notes,
        draws=forecast_draws,
    )


def _fit_binomial(
    pm, values, exposure, exposure_future, horizon, target, tier,
    exog, exog_future, time_index, future_time, config, interval_level, notes,
):
    """Binomial on the underlying counts, never on a precomputed float rate."""
    if exposure is None:
        raise ValueError(
            "A rate target needs its denominator: pass exposure (e.g. total_comments). "
            "Fitting a Binomial to a precomputed float rate discards the sample size."
        )
    successes = np.rint(np.asarray(values, dtype=float)).astype(int)
    trials = np.rint(np.asarray(exposure, dtype=float)).astype(int)
    if successes.size != trials.size:
        raise ValueError("Rate numerator and denominator must align")
    if (successes > trials).any():
        raise ValueError("Rate numerator exceeds its denominator")

    # Periods with no trials observed nothing about the rate.
    informative = trials > 0
    dropped = int((~informative).sum())
    if dropped:
        notes.append(
            f"{dropped} periods have a zero denominator and carry no information "
            "about the rate, so they are excluded from the likelihood rather than "
            "treated as an observed 0%."
        )

    with pm.Model() as model:
        logit_mu, _ = _trend_and_events(pm, tier, successes.size, exog, time_index)
        probability = pm.Deterministic("p", pm.math.sigmoid(logit_mu))
        pm.Binomial(
            "obs",
            n=trials[informative],
            p=probability[informative],
            observed=successes[informative],
        )
        trace = _sample(pm, model, config)

    diagnostics = summarise_trace(trace)
    intercept = _posterior_flat(trace, "intercept")
    trend = _posterior_flat(trace, "trend")
    logit_future = intercept[:, None] + trend[:, None] * future_time[None, :]
    logit_future = _add_seasonal(trace, logit_future, tier, successes.size, horizon)
    logit_future = _add_events(trace, logit_future, exog_future)
    probability_future = 1.0 / (1.0 + np.exp(-logit_future))

    if exposure_future is None:
        notes.append(
            "No future denominator supplied; the forecast is reported as a probability."
        )
        forecast_draws = probability_future
    else:
        rng = np.random.default_rng(config.seed)
        trials_future = np.rint(np.asarray(exposure_future, dtype=float)).astype(int)[:horizon]
        forecast_draws = rng.binomial(trials_future, probability_future)

    lower, upper = _interval(forecast_draws, interval_level)
    return BayesianForecastResult(
        mean=forecast_draws.mean(axis=0),
        lower=lower,
        upper=upper,
        target=target,
        likelihood="binomial",
        tier=tier,
        diagnostics=diagnostics,
        posterior_predictive={},
        effects=_effect_summary(trace),
        notes=notes,
        draws=forecast_draws,
    )


def _add_seasonal(trace, linear, tier, n_periods, horizon, prefix=""):
    """Add the Fourier seasonal contribution for future periods, if fitted."""
    if tier not in (TIER_MODERATE, TIER_RICH):
        return linear
    if f"{prefix}seasonal_sin" not in trace.posterior:
        return linear
    season = 52.0
    future = np.arange(n_periods, n_periods + horizon)
    sin_term = np.sin(2 * np.pi * future / season)
    cos_term = np.cos(2 * np.pi * future / season)
    sin_draws = _posterior_flat(trace, f"{prefix}seasonal_sin")[:, None]
    cos_draws = _posterior_flat(trace, f"{prefix}seasonal_cos")[:, None]
    return linear + sin_draws * sin_term[None, :] + cos_draws * cos_term[None, :]


def _add_events(trace, linear, exog_future, name: str = "event_effect"):
    """Add the modelled event response for future periods, if fitted."""
    if exog_future is None or name not in trace.posterior:
        return linear
    coefficients = _posterior_flat(trace, name)
    contribution = coefficients @ np.asarray(exog_future, dtype=float).T
    return linear + contribution


def _effect_summary(trace) -> dict:
    """Posterior mean and 90% credible interval for each scalar effect."""
    summary: dict = {}
    for name in trace.posterior.data_vars:
        if name in {"mu", "p", "p_nonzero"}:
            continue
        draws = _posterior_flat(trace, str(name))
        flat = draws.reshape(draws.shape[0], -1)
        for column in range(flat.shape[1]):
            key = str(name) if flat.shape[1] == 1 else f"{name}[{column}]"
            values = flat[:, column]
            summary[key] = {
                "mean": float(np.mean(values)),
                "hdi_5": float(np.quantile(values, 0.05)),
                "hdi_95": float(np.quantile(values, 0.95)),
            }
    return summary


__all__ = [
    "BayesianForecastResult",
    "BayesianUnavailableError",
    "KIND_CONTINUOUS",
    "KIND_COUNT",
    "KIND_RATE",
    "SamplerConfig",
    "choose_likelihood",
    "fit_bayesian_forecast",
]

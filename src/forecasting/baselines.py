"""Simple, credible forecasting baselines.

Simple models stay eligible to win. On a short, sparse weekly series a naive
forecast is frequently hard to beat, and a benchmark that cannot lose is not
a benchmark.

Every forecaster shares one interface:

    model.fit(y, exog=None)
    model.predict(horizon, exog=None) -> ForecastResult(mean, lower, upper)

Intervals for the non-probabilistic models come from the empirical quantiles
of their in-sample residuals, so they widen honestly on a noisy series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_INTERVAL = 0.8


@dataclass
class ForecastResult:
    """Point forecast plus an interval, one entry per horizon step."""

    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    model_name: str = "model"
    interval_level: float = DEFAULT_INTERVAL

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


class BaseForecaster:
    """Common fit/predict contract and residual-based intervals."""

    name = "base"
    #: Rank used to break ties between models with equivalent accuracy.
    complexity = 0

    def __init__(self, interval_level: float = DEFAULT_INTERVAL, non_negative: bool = True):
        self.interval_level = interval_level
        self.non_negative = non_negative
        self._residuals: np.ndarray = np.array([])
        self._history: np.ndarray = np.array([])

    def fit(self, y, exog=None) -> "BaseForecaster":
        raise NotImplementedError

    def _point_forecast(self, horizon: int, exog=None) -> np.ndarray:
        raise NotImplementedError

    def predict(self, horizon: int, exog=None) -> ForecastResult:
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        mean = np.asarray(self._point_forecast(horizon, exog), dtype=float)
        if self.non_negative:
            mean = np.clip(mean, 0.0, None)
        lower, upper = self._interval(mean)
        return ForecastResult(
            mean=mean,
            lower=lower,
            upper=upper,
            model_name=self.name,
            interval_level=self.interval_level,
        )

    def _interval(self, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Empirical residual quantiles, widened with the horizon."""
        if self._residuals.size < 2:
            spread = float(np.std(self._history)) if self._history.size > 1 else 0.0
            low_offset = np.full(mean.shape, -spread)
            high_offset = np.full(mean.shape, spread)
        else:
            tail = (1.0 - self.interval_level) / 2.0
            low_offset = np.full(mean.shape, float(np.quantile(self._residuals, tail)))
            high_offset = np.full(mean.shape, float(np.quantile(self._residuals, 1.0 - tail)))
        # Uncertainty grows with the horizon: a 4-week-ahead forecast is not as
        # reliable as a 1-week-ahead one.
        growth = np.sqrt(np.arange(1, len(mean) + 1))
        lower = mean + low_offset * growth
        upper = mean + high_offset * growth
        # Residual quantiles can sit entirely on one side of zero on a short or
        # trending history. An interval that excludes its own point forecast is
        # meaningless, so the bounds are clamped around it.
        lower = np.minimum(lower, mean)
        upper = np.maximum(upper, mean)
        if self.non_negative:
            lower = np.clip(lower, 0.0, None)
        return lower, upper

    @staticmethod
    def _as_array(y) -> np.ndarray:
        values = np.asarray(pd.Series(y).astype(float).values, dtype=float)
        if values.size == 0:
            raise ValueError("Cannot fit a forecaster on an empty series")
        if not np.isfinite(values).all():
            raise ValueError("Series contains non-finite values")
        return values


class NaiveForecaster(BaseForecaster):
    """Repeat the last observed value."""

    name = "naive"
    complexity = 1

    def fit(self, y, exog=None) -> "NaiveForecaster":
        values = self._as_array(y)
        self._history = values
        self._last = float(values[-1])
        self._residuals = values[1:] - values[:-1] if values.size > 1 else np.array([])
        return self

    def _point_forecast(self, horizon: int, exog=None) -> np.ndarray:
        return np.full(horizon, self._last)


class MeanForecaster(BaseForecaster):
    """Repeat the mean of a trailing window; robust on spiky sparse series."""

    name = "mean"
    complexity = 1

    def __init__(self, window: int | None = 8, **kwargs):
        super().__init__(**kwargs)
        self.window = window

    def fit(self, y, exog=None) -> "MeanForecaster":
        values = self._as_array(y)
        self._history = values
        window = values if self.window is None else values[-self.window :]
        self._level = float(np.mean(window))
        self._residuals = values - self._level
        return self

    def _point_forecast(self, horizon: int, exog=None) -> np.ndarray:
        return np.full(horizon, self._level)


class SeasonalNaiveForecaster(BaseForecaster):
    """Repeat the value from one season ago; needs at least two full seasons."""

    name = "seasonal_naive"
    complexity = 2

    def __init__(self, period: int = 52, **kwargs):
        super().__init__(**kwargs)
        if period < 2:
            raise ValueError("Seasonal period must be at least 2")
        self.period = period

    @classmethod
    def is_eligible(cls, n_observations: int, period: int = 52) -> bool:
        """Seasonal naive needs two full cycles to say anything at all."""
        return n_observations >= 2 * period

    def fit(self, y, exog=None) -> "SeasonalNaiveForecaster":
        values = self._as_array(y)
        if values.size < self.period:
            raise ValueError(
                f"Seasonal naive needs at least {self.period} observations, got {values.size}"
            )
        self._history = values
        self._season = values[-self.period :]
        if values.size > self.period:
            self._residuals = values[self.period :] - values[: -self.period]
        else:
            self._residuals = np.array([])
        return self

    def _point_forecast(self, horizon: int, exog=None) -> np.ndarray:
        repeats = int(np.ceil(horizon / self.period))
        return np.tile(self._season, repeats)[:horizon]


class ExponentialSmoothingForecaster(BaseForecaster):
    """Holt-style exponential smoothing via statsmodels, with a safe fallback."""

    name = "exponential_smoothing"
    complexity = 3

    def __init__(self, trend: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.trend = trend
        self._model = None

    @classmethod
    def is_eligible(cls, n_observations: int) -> bool:
        return n_observations >= 8

    def fit(self, y, exog=None) -> "ExponentialSmoothingForecaster":
        values = self._as_array(y)
        self._history = values
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            self._model = ExponentialSmoothing(
                values, trend=self.trend, seasonal=None, initialization_method="estimated"
            ).fit()
            self._residuals = values - np.asarray(self._model.fittedvalues, dtype=float)
        except Exception as exc:  # noqa: BLE001 - degenerate series must not abort a backtest
            LOGGER.warning("Exponential smoothing fit failed (%s); using last value", exc)
            self._model = None
            self._residuals = values[1:] - values[:-1] if values.size > 1 else np.array([])
        return self

    def _point_forecast(self, horizon: int, exog=None) -> np.ndarray:
        if self._model is None:
            return np.full(horizon, float(self._history[-1]))
        return np.asarray(self._model.forecast(horizon), dtype=float)


class RidgeEventForecaster(BaseForecaster):
    """Ridge regression on lagged values plus event exogenous features.

    This is the "regularised regression with exogenous event features" option:
    it can express an event response that the pure baselines cannot, while
    staying stable on a short series where SARIMAX would not identify.
    """

    name = "ridge_event"
    complexity = 4

    def __init__(self, lags: int = 2, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lags = lags
        self.alpha = alpha
        self._model = None

    @classmethod
    def is_eligible(cls, n_observations: int) -> bool:
        return n_observations >= 16

    def _design(self, values: np.ndarray, exog: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        rows, targets = [], []
        for position in range(self.lags, values.size):
            features = list(values[position - self.lags : position][::-1])
            if exog is not None:
                features.extend(np.asarray(exog[position], dtype=float).ravel())
            rows.append(features)
            targets.append(values[position])
        return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)

    def fit(self, y, exog=None) -> "RidgeEventForecaster":
        from sklearn.linear_model import Ridge

        values = self._as_array(y)
        self._history = values
        exog_array = None if exog is None else np.asarray(exog, dtype=float)
        design, targets = self._design(values, exog_array)
        if design.shape[0] < 3:
            raise ValueError("Not enough observations to fit the ridge forecaster")
        self._model = Ridge(alpha=self.alpha).fit(design, targets)
        self._residuals = targets - self._model.predict(design)
        return self

    def _point_forecast(self, horizon: int, exog=None) -> np.ndarray:
        history = list(self._history)
        exog_future = None if exog is None else np.asarray(exog, dtype=float)
        predictions = []
        for step in range(horizon):
            features = list(np.asarray(history[-self.lags :], dtype=float)[::-1])
            if exog_future is not None:
                features.extend(np.asarray(exog_future[step], dtype=float).ravel())
            value = float(self._model.predict(np.asarray([features], dtype=float))[0])
            predictions.append(value)
            history.append(value)
        return np.asarray(predictions, dtype=float)


#: Registry consulted by the experiment runner; ordered simplest first.
BASELINE_REGISTRY = {
    NaiveForecaster.name: NaiveForecaster,
    MeanForecaster.name: MeanForecaster,
    SeasonalNaiveForecaster.name: SeasonalNaiveForecaster,
    ExponentialSmoothingForecaster.name: ExponentialSmoothingForecaster,
    RidgeEventForecaster.name: RidgeEventForecaster,
}


def eligible_baselines(
    n_observations: int, seasonal_period: int = 52, min_train: int | None = None
) -> list[str]:
    """Names of baselines the observed history can actually support.

    ``min_train`` is the smallest training prefix a backtest will use. A
    seasonal model that cannot be fitted at the earliest origin is not offered
    at all, rather than being entered and failing on most origins.
    """
    names = [NaiveForecaster.name, MeanForecaster.name]
    smallest_fit = n_observations if min_train is None else min(min_train, n_observations)
    if SeasonalNaiveForecaster.is_eligible(n_observations, seasonal_period) and (
        smallest_fit >= seasonal_period
    ):
        names.append(SeasonalNaiveForecaster.name)
    if ExponentialSmoothingForecaster.is_eligible(smallest_fit):
        names.append(ExponentialSmoothingForecaster.name)
    if RidgeEventForecaster.is_eligible(smallest_fit):
        names.append(RidgeEventForecaster.name)
    return names


__all__ = [
    "BASELINE_REGISTRY",
    "BaseForecaster",
    "ExponentialSmoothingForecaster",
    "ForecastResult",
    "MeanForecaster",
    "NaiveForecaster",
    "RidgeEventForecaster",
    "SeasonalNaiveForecaster",
    "eligible_baselines",
]

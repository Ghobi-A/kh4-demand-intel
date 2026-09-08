"""Forecast accuracy, bias and interval metrics.

Ordinary MAPE is deliberately absent: the weekly series contains zero weeks,
where a percentage error is undefined or explodes. WAPE (weighted absolute
percentage error) carries the same "error relative to scale" reading while
staying finite in the presence of zeros.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def _aligned(actual, forecast) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)
    if actual_array.shape != forecast_array.shape:
        raise ValueError(
            f"actual and forecast must align: {actual_array.shape} vs {forecast_array.shape}"
        )
    finite = np.isfinite(actual_array) & np.isfinite(forecast_array)
    return actual_array[finite], forecast_array[finite]


def mae(actual, forecast) -> float:
    """Mean absolute error."""
    a, f = _aligned(actual, forecast)
    return float(np.mean(np.abs(a - f))) if a.size else float("nan")


def rmse(actual, forecast) -> float:
    """Root mean squared error."""
    a, f = _aligned(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2))) if a.size else float("nan")


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: total error over total actual.

    Finite whenever the actuals are not all zero, which is why it replaces
    MAPE on a series with empty weeks.
    """
    a, f = _aligned(actual, forecast)
    if not a.size:
        return float("nan")
    denominator = np.sum(np.abs(a))
    if denominator <= EPSILON:
        return float("nan")
    return float(np.sum(np.abs(a - f)) / denominator)


def bias(actual, forecast) -> float:
    """Signed mean error: positive means the forecast runs above actuals."""
    a, f = _aligned(actual, forecast)
    return float(np.mean(f - a)) if a.size else float("nan")


def relative_bias(actual, forecast) -> float:
    """Signed error as a share of the mean actual level."""
    a, f = _aligned(actual, forecast)
    if not a.size:
        return float("nan")
    scale = np.mean(np.abs(a))
    if scale <= EPSILON:
        return float("nan")
    return float(np.mean(f - a) / scale)


def mase(actual, forecast, insample, seasonality: int = 1) -> float:
    """Mean absolute scaled error, or NaN when the naive scale is degenerate.

    MASE is only meaningful when the in-sample naive forecast has non-zero
    mean absolute error; on a sparse series that is not guaranteed, so the
    caller gets NaN rather than a divide-by-zero artefact.
    """
    a, f = _aligned(actual, forecast)
    insample_array = np.asarray(insample, dtype=float)
    insample_array = insample_array[np.isfinite(insample_array)]
    if a.size == 0 or insample_array.size <= seasonality:
        return float("nan")
    naive_errors = np.abs(insample_array[seasonality:] - insample_array[:-seasonality])
    scale = np.mean(naive_errors)
    if not np.isfinite(scale) or scale <= EPSILON:
        return float("nan")
    return float(np.mean(np.abs(a - f)) / scale)


def interval_coverage(actual, lower, upper) -> float:
    """Share of actuals falling inside the forecast interval."""
    a = np.asarray(actual, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    finite = np.isfinite(a) & np.isfinite(low) & np.isfinite(high)
    if not finite.any():
        return float("nan")
    inside = (a[finite] >= low[finite]) & (a[finite] <= high[finite])
    return float(np.mean(inside))


def mean_interval_width(lower, upper) -> float:
    """Average width of the forecast interval."""
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    finite = np.isfinite(low) & np.isfinite(high)
    if not finite.any():
        return float("nan")
    return float(np.mean(high[finite] - low[finite]))


def forecast_metrics(
    actual,
    forecast,
    lower=None,
    upper=None,
    insample=None,
    seasonality: int = 1,
) -> dict:
    """Full metric set for one set of forecasts."""
    result = {
        "n": int(np.size(np.asarray(actual))),
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "wape": wape(actual, forecast),
        "bias": bias(actual, forecast),
        "relative_bias": relative_bias(actual, forecast),
    }
    if insample is not None:
        result["mase"] = mase(actual, forecast, insample, seasonality=seasonality)
    if lower is not None and upper is not None:
        result["interval_coverage"] = interval_coverage(actual, lower, upper)
        result["mean_interval_width"] = mean_interval_width(lower, upper)
    return result


__all__ = [
    "bias",
    "forecast_metrics",
    "interval_coverage",
    "mae",
    "mase",
    "mean_interval_width",
    "relative_bias",
    "rmse",
    "wape",
]

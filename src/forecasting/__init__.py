"""Forecasting of observable behavioural demand proxies.

Everything here operates on the weekly table produced by ``src.temporal``.
Targets are observable quantities derived from public discussion, never
sales: see ``src/temporal.py`` for the demand-proxy definition.

PyMC is imported lazily inside ``src.forecasting.bayesian`` so that reading
generated reports (the dashboard's job) never requires a Bayesian stack.
"""

from src.forecasting.metrics import (
    bias,
    forecast_metrics,
    interval_coverage,
    mae,
    mase,
    mean_interval_width,
    rmse,
    wape,
)
from src.forecasting.validation import RollingOriginSplit, backtest, rolling_origin_splits

__all__ = [
    "RollingOriginSplit",
    "backtest",
    "bias",
    "forecast_metrics",
    "interval_coverage",
    "mae",
    "mase",
    "mean_interval_width",
    "rmse",
    "rolling_origin_splits",
    "wape",
]

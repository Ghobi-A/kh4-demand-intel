"""Rolling-origin (walk-forward) validation for time-series forecasts.

Random splits are never used: a random split lets a model learn from weeks
that come after the week it predicts, which is not a situation any real
forecast faces. Every split here trains on a contiguous prefix and predicts
the periods immediately after it.

    train weeks 1-12 -> predict 13-14
    train weeks 1-14 -> predict 15-16
    train weeks 1-16 -> predict 17-18
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RollingOriginSplit:
    """One forecast origin: the training prefix and the periods to predict."""

    origin: int
    train_index: np.ndarray
    test_index: np.ndarray

    def __post_init__(self) -> None:
        if self.train_index.size and self.test_index.size:
            if self.train_index.max() >= self.test_index.min():
                raise ValueError("Training data must end before the forecast period begins")


def rolling_origin_splits(
    n_observations: int,
    min_train: int = 12,
    horizon: int = 2,
    step: int = 2,
    max_origins: int | None = None,
) -> list[RollingOriginSplit]:
    """Generate walk-forward splits over ``n_observations`` periods."""
    if min_train < 1:
        raise ValueError("min_train must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if step < 1:
        raise ValueError("step must be at least 1")

    splits: list[RollingOriginSplit] = []
    origin = min_train
    while origin + horizon <= n_observations:
        splits.append(
            RollingOriginSplit(
                origin=origin,
                train_index=np.arange(0, origin),
                test_index=np.arange(origin, origin + horizon),
            )
        )
        origin += step
    if max_origins is not None:
        splits = splits[-max_origins:]
    return splits


def backtest(
    model_factory,
    y: pd.Series,
    splits: list[RollingOriginSplit],
    exog_builder=None,
    model_name: str = "model",
) -> pd.DataFrame:
    """Run a forecaster across every origin and return one row per prediction.

    ``exog_builder(train_end_index)`` must construct features using only data
    available at that origin; it is called once per origin so a leaky builder
    cannot quietly reuse features fitted on the whole series.
    """
    values = pd.Series(y).reset_index(drop=True).astype(float)
    rows: list[dict] = []

    for split in splits:
        train_y = values.iloc[split.train_index]
        exog_train = exog_future = None
        if exog_builder is not None:
            exog_train, exog_future = exog_builder(split.origin, len(split.test_index))

        model = model_factory()
        try:
            model.fit(train_y, exog=exog_train)
            result = model.predict(len(split.test_index), exog=exog_future)
        except Exception as exc:  # noqa: BLE001 - one origin failing must not lose the rest
            LOGGER.warning("%s failed at origin %s: %s", model_name, split.origin, exc)
            continue

        for step_index, position in enumerate(split.test_index):
            rows.append(
                {
                    "model": model_name,
                    "origin": int(split.origin),
                    "horizon_step": step_index + 1,
                    "period_index": int(position),
                    "actual": float(values.iloc[position]),
                    "forecast": float(result.mean[step_index]),
                    "lower": float(result.lower[step_index]),
                    "upper": float(result.upper[step_index]),
                }
            )

    return pd.DataFrame(rows)


__all__ = ["RollingOriginSplit", "backtest", "rolling_origin_splits"]

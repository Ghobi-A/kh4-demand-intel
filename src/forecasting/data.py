"""Loading the weekly table and building origin-honest exogenous features.

The one rule enforced here is that a feature offered to a model at forecast
origin *t* must have been knowable at *t*. Two things follow:

* Lagged behavioural terms are shifted by at least the forecast horizon, so a
  model never conditions on a week it is supposed to be predicting.
* Event indicators are gated on ``announced_date``: a public event counts as a
  known future input only once it had actually been announced at the origin.
  A trailer that had not been revealed yet cannot inform that origin.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.events import DEFAULT_EVENTS_PATH, event_features_for_periods, load_events
from src.timestamps import parse_timestamps

LOGGER = logging.getLogger(__name__)

#: Targets the runner benchmarks, with the likelihood family each implies.
TARGET_KINDS = {
    "actionable_probability_mass": "continuous_nonnegative",
    "actionable_count": "count",
    "total_comments": "count",
    "high_intent_count": "count",
    "actionable_rate": "rate",
}

DEFAULT_TARGETS = ("actionable_probability_mass", "total_comments")

EVENT_FEATURE_COLUMNS = ["event_active", "event_decay", "major_announcement_flag"]


@dataclass
class WeeklyDataset:
    """The weekly table plus the metadata describing how it was produced."""

    frame: pd.DataFrame
    meta: dict

    @property
    def evaluation_status(self) -> str:
        return str(self.meta.get("evaluation_status", "unknown"))

    @property
    def is_sample_only(self) -> bool:
        return bool(self.meta.get("sample_only", False))

    def available_targets(self, requested) -> list[str]:
        """Requested targets that this table actually carries."""
        return [name for name in requested if name in self.frame.columns]


def load_weekly(path: Path) -> WeeklyDataset:
    """Load a weekly demand-signal CSV and its sidecar metadata."""
    path = Path(path)
    frame = pd.read_csv(path)
    if "period_start" not in frame.columns:
        raise ValueError(f"{path} has no 'period_start' column; rebuild it with src.temporal")
    parsed, _ = parse_timestamps(frame["period_start"])
    frame["period_start"] = parsed
    frame = frame.sort_values("period_start").reset_index(drop=True)

    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return WeeklyDataset(frame=frame, meta=meta)


def select_target(dataset: WeeklyDataset, target: str) -> pd.Series:
    """Return one target series, refusing targets the table does not carry."""
    if target not in dataset.frame.columns:
        raise ValueError(
            f"Target {target!r} is not in the weekly table. Available: "
            f"{sorted(set(TARGET_KINDS) & set(dataset.frame.columns))}"
        )
    series = pd.to_numeric(dataset.frame[target], errors="coerce")
    return series


def target_kind(target: str) -> str:
    """Likelihood family implied by a target, used to pick a Bayesian model."""
    return TARGET_KINDS.get(target, "continuous_nonnegative")


def make_event_exog(
    periods: pd.Series,
    events: pd.DataFrame | None,
    origin_period: pd.Timestamp | None,
    columns: list[str] | None = None,
) -> np.ndarray | None:
    """Event features for every period, as known at ``origin_period``.

    ``origin_period`` is the last period the model may see. Events announced
    after it are invisible, so backtest features never carry knowledge that
    did not exist when the forecast would have been made.
    """
    if events is None or events.empty:
        return None
    columns = columns or EVENT_FEATURE_COLUMNS
    features = event_features_for_periods(periods, events, origin=origin_period)
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise ValueError(f"Event features missing columns: {missing}")
    return features[columns].astype(float).to_numpy()


class ExogBuilder:
    """Builds train/future exogenous blocks for one backtest origin.

    Constructed once per dataset and called once per origin, so the features a
    model sees are always rebuilt from the origin rather than shared from a
    full-series fit.
    """

    def __init__(
        self,
        periods: pd.Series,
        events: pd.DataFrame | None,
        columns: list[str] | None = None,
    ):
        self.periods = pd.Series(periods).reset_index(drop=True)
        self.events = events
        self.columns = columns or EVENT_FEATURE_COLUMNS

    def _extended_periods(self, needed: int) -> pd.Series:
        """Period index long enough to cover the forecast, extended weekly.

        Forecast periods lie beyond the observed table, so their event features
        have to be built for periods that do not exist in the data yet.
        """
        if needed <= len(self.periods):
            return self.periods
        last = self.periods.iloc[-1]
        extra = pd.Series(
            pd.date_range(
                start=last + pd.Timedelta(weeks=1),
                periods=needed - len(self.periods),
                freq="7D",
                tz="UTC",
            )
        )
        return pd.concat([self.periods, extra], ignore_index=True)

    def __call__(self, origin: int, horizon: int):
        """Return (exog_for_training, exog_for_forecast) at this origin."""
        if self.events is None or self.events.empty:
            return None, None
        periods = self._extended_periods(origin + horizon)
        origin_period = periods.iloc[min(origin, len(periods)) - 1]
        full = make_event_exog(periods, self.events, origin_period, self.columns)
        if full is None:
            return None, None
        train_block = full[:origin]
        future_block = full[origin : origin + horizon]
        return train_block, future_block


def build_exog_builder(
    dataset: WeeklyDataset, events_path: Path = DEFAULT_EVENTS_PATH
) -> ExogBuilder | None:
    """Create the origin-aware exogenous builder for a dataset, if events exist."""
    events_path = Path(events_path)
    if not events_path.exists():
        LOGGER.info("No event calendar at %s; forecasting without event features", events_path)
        return None
    events = load_events(events_path)
    return ExogBuilder(dataset.frame["period_start"], events)


def trim_leading_empty(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Drop leading all-zero periods before the series really begins.

    Weeks before the first observation are an artefact of completing the index,
    not evidence of zero demand, and they distort both fits and error metrics.
    """
    values = pd.to_numeric(frame[target], errors="coerce").fillna(0.0)
    non_zero = np.flatnonzero(values.to_numpy() > 0)
    if non_zero.size == 0:
        return frame
    return frame.iloc[non_zero[0] :].reset_index(drop=True)


__all__ = [
    "DEFAULT_TARGETS",
    "EVENT_FEATURE_COLUMNS",
    "ExogBuilder",
    "TARGET_KINDS",
    "WeeklyDataset",
    "build_exog_builder",
    "load_weekly",
    "make_event_exog",
    "select_target",
    "target_kind",
    "trim_leading_empty",
]

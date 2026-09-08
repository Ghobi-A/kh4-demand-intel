"""Forecast monitoring: rolling error, signed bias, and interval calibration.

Accuracy alone hides the failure mode that matters most in planning: a
forecast that is consistently wrong in the *same direction*. A model can post
a respectable error while under-forecasting every single week, which
systematically under-resources whatever depends on it. This module tracks the
sign of the error over successive origins and flags persistent runs.

All thresholds are configurable; none are baked into the analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class BiasThresholds:
    """When a run of same-signed errors counts as persistent bias."""

    #: Consecutive periods erring the same way before a flag is raised.
    consecutive_periods: int = 4
    #: Error must also exceed this share of the mean actual level to matter.
    tolerance_fraction: float = 0.10
    #: Window for the rolling error statistics.
    rolling_window: int = 4
    #: Nominal interval level, used to judge coverage.
    interval_level: float = 0.8

    def to_dict(self) -> dict:
        return asdict(self)


def monitoring_table(
    backtest_results: pd.DataFrame,
    thresholds: BiasThresholds | None = None,
) -> pd.DataFrame:
    """Per-forecast monitoring record with rolling error and interval hits."""
    thresholds = thresholds or BiasThresholds()
    if backtest_results.empty:
        return pd.DataFrame(
            columns=[
                "model", "origin", "horizon_step", "period_index", "actual", "forecast",
                "signed_error", "absolute_error", "rolling_mae", "rolling_wape",
                "rolling_bias", "interval_hit",
            ]
        )

    table = backtest_results.sort_values(["model", "origin", "horizon_step"]).copy()
    table["signed_error"] = table["forecast"] - table["actual"]
    table["absolute_error"] = table["signed_error"].abs()

    if {"lower", "upper"}.issubset(table.columns):
        table["interval_hit"] = (
            (table["actual"] >= table["lower"]) & (table["actual"] <= table["upper"])
        ).astype(int)
        table["interval_width"] = table["upper"] - table["lower"]

    window = max(1, thresholds.rolling_window)
    grouped = table.groupby("model", group_keys=False)
    table["rolling_mae"] = grouped["absolute_error"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    table["rolling_bias"] = grouped["signed_error"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    rolling_abs = grouped["absolute_error"].transform(
        lambda s: s.rolling(window, min_periods=1).sum()
    )
    rolling_actual = grouped["actual"].transform(
        lambda s: s.rolling(window, min_periods=1).sum().abs()
    )
    table["rolling_wape"] = np.where(rolling_actual > 0, rolling_abs / rolling_actual, np.nan)

    return table.reset_index(drop=True)


def flag_persistent_bias(
    monitoring: pd.DataFrame,
    thresholds: BiasThresholds | None = None,
) -> dict:
    """Detect runs of same-signed, materially large forecast errors.

    An error only counts towards a run when it exceeds
    ``tolerance_fraction`` of the mean actual level, so a long run of
    negligible errors does not raise a false alarm.
    """
    thresholds = thresholds or BiasThresholds()
    result: dict = {"thresholds": thresholds.to_dict(), "models": {}}
    if monitoring.empty:
        return result

    for model, frame in monitoring.groupby("model"):
        ordered = frame.sort_values(["origin", "horizon_step"])
        errors = ordered["signed_error"].to_numpy(dtype=float)
        actuals = ordered["actual"].to_numpy(dtype=float)
        scale = float(np.mean(np.abs(actuals))) if actuals.size else 0.0
        tolerance = scale * thresholds.tolerance_fraction

        material = np.abs(errors) > tolerance
        signs = np.sign(errors) * material

        runs: list[dict] = []
        current_sign = 0
        run_length = 0
        for position, sign in enumerate(signs):
            if sign != 0 and sign == current_sign:
                run_length += 1
            else:
                current_sign = sign
                run_length = 1 if sign != 0 else 0
            if run_length >= thresholds.consecutive_periods:
                runs.append(
                    {
                        "direction": "over_forecasting" if current_sign > 0 else "under_forecasting",
                        "length": int(run_length),
                        "ends_at_origin": int(ordered.iloc[position]["origin"]),
                        "mean_signed_error": float(
                            np.mean(errors[position - run_length + 1 : position + 1])
                        ),
                    }
                )

        coverage = (
            float(ordered["interval_hit"].mean()) if "interval_hit" in ordered.columns else None
        )
        coverage_gap = (
            None if coverage is None else float(coverage - thresholds.interval_level)
        )

        # Keep only the longest run per direction so a single long streak is
        # reported once rather than once per extra period.
        longest: dict[str, dict] = {}
        for run in runs:
            best = longest.get(run["direction"])
            if best is None or run["length"] > best["length"]:
                longest[run["direction"]] = run

        result["models"][str(model)] = {
            "n_forecasts": int(len(ordered)),
            "mean_signed_error": float(np.mean(errors)) if errors.size else float("nan"),
            "materiality_threshold": tolerance,
            "persistent_bias": bool(longest),
            "bias_runs": list(longest.values()),
            "interval_coverage": coverage,
            "coverage_gap_vs_nominal": coverage_gap,
            "interval_calibration": _describe_coverage(coverage, thresholds.interval_level),
        }
    return result


def _describe_coverage(coverage: float | None, nominal: float) -> str:
    if coverage is None:
        return "unknown"
    gap = coverage - nominal
    if abs(gap) <= 0.10:
        return "close to nominal"
    return "too narrow (overconfident)" if gap < 0 else "too wide (underconfident)"


__all__ = ["BiasThresholds", "flag_persistent_bias", "monitoring_table"]

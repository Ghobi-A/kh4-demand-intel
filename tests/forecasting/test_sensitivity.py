"""Tests for matched-origin ranking sensitivity."""

from __future__ import annotations

import pandas as pd

from src.forecasting.sensitivity import matched_origin_sensitivity, matched_subset


def _rows(model: str, origin: int, actual: float, forecast: float) -> dict:
    return {
        "model": model,
        "origin": origin,
        "horizon_step": 1,
        "actual": actual,
        "forecast": forecast,
        "lower": forecast,
        "upper": forecast,
    }


def _backtest() -> pd.DataFrame:
    rows = []
    # naive wins only because of one lucky origin (30).
    for origin, actual, naive, bayes in [(10, 5, 4, 5), (20, 5, 4, 5), (30, 20, 20, 5)]:
        rows.append(_rows("naive", origin, actual, naive))
        rows.append(_rows("bayesian", origin, actual, bayes))
    # A baseline origin the Bayesian model was not scored on is excluded.
    rows.append(_rows("naive", 0, 1, 1))
    return pd.DataFrame(rows)


def test_matched_subset_keeps_only_reference_origins() -> None:
    subset = matched_subset(_backtest())
    assert set(subset["origin"]) == {10, 20, 30}


def test_single_origin_win_is_flagged_fragile() -> None:
    result = matched_origin_sensitivity(_backtest())
    assert result["leader"] == "naive"
    assert result["dominant_origin"] == 30
    assert result["leave_one_origin_out_leaders"]["30"] == "bayesian"
    assert result["leader_stable_under_leave_one_out"] is False
    assert result["fragile"] is True
    assert "fragile" in result["verdict"]


def test_missing_reference_model_is_reported_not_raised() -> None:
    backtest = _backtest()
    result = matched_origin_sensitivity(backtest[backtest["model"] != "bayesian"])
    assert result["available"] is False

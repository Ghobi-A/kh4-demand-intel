"""Tests for model selection and the forecasting experiment runner."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.forecasting.experiment import ForecastConfig, run_experiment, select_best_model


def _comparison(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _flags(models) -> dict:
    return {"thresholds": {"interval_level": 0.8}, "models": models}


def test_lowest_wape_wins_when_it_is_also_sound() -> None:
    comparison = _comparison(
        [
            {"model": "naive", "wape": 0.30, "mae": 3.0, "complexity": 1},
            {"model": "ridge_event", "wape": 0.20, "mae": 2.0, "complexity": 4},
        ]
    )
    flags = _flags(
        {
            "naive": {"persistent_bias": False, "interval_coverage": 0.8},
            "ridge_event": {"persistent_bias": False, "interval_coverage": 0.8},
        }
    )

    selected, reasoning = select_best_model(comparison, flags)

    assert selected == "ridge_event"
    assert any("passes the bias" in line for line in reasoning)


def test_a_wape_winner_with_persistent_bias_is_demoted() -> None:
    comparison = _comparison(
        [
            {"model": "naive", "wape": 0.30, "mae": 3.0, "complexity": 1},
            {"model": "ridge_event", "wape": 0.20, "mae": 2.0, "complexity": 4},
        ]
    )
    flags = _flags(
        {
            "naive": {"persistent_bias": False, "interval_coverage": 0.8},
            "ridge_event": {"persistent_bias": True, "interval_coverage": 0.8},
        }
    )

    selected, reasoning = select_best_model(comparison, flags)

    assert selected == "naive"
    assert any("demoted" in line for line in reasoning)


def test_badly_calibrated_intervals_also_demote_a_model() -> None:
    comparison = _comparison(
        [
            {"model": "mean", "wape": 0.40, "mae": 4.0, "complexity": 1},
            {"model": "exponential_smoothing", "wape": 0.25, "mae": 2.5, "complexity": 3},
        ]
    )
    flags = _flags(
        {
            "mean": {"persistent_bias": False, "interval_coverage": 0.78},
            "exponential_smoothing": {"persistent_bias": False, "interval_coverage": 0.20},
        }
    )

    selected, _ = select_best_model(comparison, flags)

    assert selected == "mean"


def test_ties_go_to_the_simpler_model() -> None:
    comparison = _comparison(
        [
            {"model": "ridge_event", "wape": 0.25, "mae": 2.5, "complexity": 4},
            {"model": "naive", "wape": 0.25, "mae": 2.5, "complexity": 1},
        ]
    )
    flags = _flags(
        {
            "ridge_event": {"persistent_bias": False, "interval_coverage": 0.8},
            "naive": {"persistent_bias": False, "interval_coverage": 0.8},
        }
    )

    selected, _ = select_best_model(comparison, flags)

    assert selected == "naive"


def test_when_no_model_is_sound_the_report_says_so() -> None:
    comparison = _comparison([{"model": "naive", "wape": 0.5, "mae": 5.0, "complexity": 1}])
    flags = _flags({"naive": {"persistent_bias": True, "interval_coverage": 0.8}})

    selected, reasoning = select_best_model(comparison, flags)

    assert selected == "naive"
    assert any("none of them is reliable" in line for line in reasoning)


def _write_weekly(tmp_path, sample_only: bool):
    periods = pd.date_range("2024-01-01", periods=60, freq="7D", tz="UTC")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "period_start": periods.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "total_comments": rng.integers(0, 30, 60),
            "actionable_probability_mass": np.abs(rng.normal(6, 2, 60)),
        }
    )
    path = tmp_path / "weekly.csv"
    frame.to_csv(path, index=False)
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "sample_only": sample_only,
                "volume_representative": not sample_only,
                "evaluation_status": "demo_only" if sample_only else "full_dataset",
                "coverage": {"observed_periods": 60, "non_empty_periods": 55},
                "aggregation": {"period_start_min": str(periods[0]), "period_start_max": str(periods[-1])},
            }
        )
    )
    return path


def test_a_sample_only_dataset_cannot_populate_the_canonical_report_directory(tmp_path) -> None:
    """The stratified sample must never look like a measured result."""
    path = _write_weekly(tmp_path, sample_only=True)

    with pytest.raises(ValueError, match="sample_only"):
        run_experiment(path, report_dir=tmp_path / "reports", demo=False)


def test_experiment_writes_every_artefact_and_stamps_demo_runs(tmp_path) -> None:
    path = _write_weekly(tmp_path, sample_only=True)
    report_dir = tmp_path / "demo_reports"

    outcome = run_experiment(
        path,
        report_dir=report_dir,
        config=ForecastConfig(min_train=20, horizon=2, step=4, include_bayesian=False),
        demo=True,
    )

    for name in [
        "backtest_results.csv",
        "model_comparison.csv",
        "monitoring.csv",
        "bias_flags.json",
        "run_metadata.json",
        "forecast_report.md",
        "observed_series.csv",
    ]:
        assert (report_dir / name).exists(), name

    report = (report_dir / "forecast_report.md").read_text()
    assert "not a substantive result" in report
    assert "not expected purchases" in report
    assert "Limitations" in report

    metadata = json.loads((report_dir / "run_metadata.json").read_text())
    assert metadata["sample_only"] is True
    assert set(metadata["targets"]) == {"actionable_probability_mass", "total_comments"}
    assert outcome["per_target"]["total_comments"]["usable"]


def test_both_targets_are_benchmarked_not_just_the_derived_proxy(tmp_path) -> None:
    path = _write_weekly(tmp_path, sample_only=True)

    outcome = run_experiment(
        path,
        report_dir=tmp_path / "reports",
        config=ForecastConfig(min_train=20, horizon=2, step=4, include_bayesian=False),
        demo=True,
    )

    comparison = pd.read_csv(tmp_path / "reports" / "model_comparison.csv")
    assert set(comparison["target"]) == {"actionable_probability_mass", "total_comments"}
    assert outcome["per_target"]["actionable_probability_mass"]["selected_model"]


def test_a_target_with_too_little_history_is_reported_not_forced(tmp_path) -> None:
    path = _write_weekly(tmp_path, sample_only=True)

    outcome = run_experiment(
        path,
        report_dir=tmp_path / "reports",
        config=ForecastConfig(min_train=500, horizon=2, include_bayesian=False),
        demo=True,
    )

    result = outcome["per_target"]["total_comments"]
    assert result["usable"] is False
    assert "rolling-origin backtest needs" in result["reason"]


def test_bayesian_model_is_scored_on_matched_origins(tmp_path, monkeypatch) -> None:
    """A model that is never backtested has not actually been compared."""
    import numpy as np

    import src.forecasting.experiment as experiment_module

    path = _write_weekly(tmp_path, sample_only=True)

    def _fake_backtest(values, target, tier, config, exog_builder, frame, splits):
        rows = []
        for split in splits[-config.bayesian_backtest_origins :]:
            for step, position in enumerate(split.test_index):
                rows.append(
                    {
                        "model": "bayesian",
                        "origin": int(split.origin),
                        "horizon_step": step + 1,
                        "period_index": int(position),
                        "actual": float(values[position]),
                        "forecast": float(values[position]) + 1.0,
                        "lower": float(values[position]) - 2.0,
                        "upper": float(values[position]) + 3.0,
                    }
                )
        return pd.DataFrame(rows), ["scored on recent origins"]

    monkeypatch.setattr(experiment_module, "_backtest_bayesian", _fake_backtest)

    outcome = run_experiment(
        path,
        report_dir=tmp_path / "reports",
        config=ForecastConfig(
            min_train=20, horizon=2, step=4, bayesian_backtest_origins=2
        ),
        demo=True,
    )

    result = outcome["per_target"]["total_comments"]
    assert "bayesian" in set(result["comparison"]["model"])

    matched = result["matched_comparison"]
    assert not matched.empty
    assert "bayesian" in set(matched["model"])
    # Every model in the matched table is scored on the same number of origins.
    assert matched["origins"].nunique() == 1
    assert (tmp_path / "reports" / "model_comparison_matched_origins.csv").exists()

    report = (tmp_path / "reports" / "forecast_report.md").read_text()
    assert "Matched-origin comparison" in report
    assert np.isfinite(matched["wape"]).all()

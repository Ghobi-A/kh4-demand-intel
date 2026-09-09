"""Contract tests for the thin analytical API."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi", reason="The API needs the api extra")
from fastapi.testclient import TestClient  # noqa: E402

from src import services  # noqa: E402
from src.api import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def forecast_dir(tmp_path, monkeypatch):
    """A minimal generated forecasting run for the API to serve."""
    report_dir = tmp_path / "forecasting"
    report_dir.mkdir()
    (report_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": "abc123",
                "timestamp": "2026-09-01T00:00:00+00:00",
                "git_commit": "deadbee",
                "evaluation_status": "demo_only",
                "sample_only": True,
                "volume_representative": False,
                "observed_period": {"start": "2026-01-05", "end": "2026-06-01"},
                "targets": {
                    "actionable_probability_mass": {
                        "usable": True,
                        "target_kind": "continuous_nonnegative",
                        "likelihood": "hurdle_gamma",
                        "complexity_tier": "minimal",
                        "periods": 40,
                        "non_empty_periods": 25,
                        "origins": [12, 14],
                        "selected_model": "naive",
                        "selection_reasoning": ["Lowest WAPE: naive (0.300)."],
                        "bayesian_diagnostics": {"max_r_hat": 1.004, "converged": True},
                    },
                    "total_comments": {"usable": False, "reason": "too few periods"},
                },
            }
        )
    )
    pd.DataFrame(
        {
            "target": ["actionable_probability_mass"],
            "model": ["naive"],
            "wape": [0.3],
            "mae": [1.2],
            "persistent_bias": [False],
        }
    ).to_csv(report_dir / "model_comparison.csv", index=False)
    pd.DataFrame(
        {
            "target": ["actionable_probability_mass"] * 2,
            "period_start": ["2026-06-08", "2026-06-15"],
            "horizon_step": [1, 2],
            "model": ["bayesian"] * 2,
            "forecast": [4.0, 4.2],
            "lower": [1.0, 1.1],
            "upper": [8.0, 8.5],
        }
    ).to_csv(report_dir / "forecast.csv", index=False)
    pd.DataFrame(
        {"target": ["actionable_probability_mass"] * 3, "actual": [3.0, 5.0, 4.0]}
    ).to_csv(report_dir / "observed_series.csv", index=False)
    (report_dir / "bias_flags.json").write_text(
        json.dumps(
            {
                "actionable_probability_mass": {
                    "models": {
                        "naive": {
                            "persistent_bias": False,
                            "mean_signed_error": -0.1,
                            "interval_coverage": 0.8,
                            "interval_calibration": "close to nominal",
                        }
                    }
                }
            }
        )
    )
    rng = np.random.default_rng(0)
    np.savez_compressed(
        report_dir / "posterior_samples_actionable_probability_mass.npz",
        draws=rng.gamma(4.0, 1.0, (200, 4)),
    )
    monkeypatch.setattr(services, "resolve_forecast_dir", lambda prefer_demo=None: report_dir)
    return report_dir


def test_health_reports_which_artefacts_exist(client, forecast_dir) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["forecast_results_available"] is True
    assert payload["forecast_is_sample_only"] is True


def test_forecast_summary_returns_the_run_contract(client, forecast_dir) -> None:
    response = client.get("/forecast/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "abc123"
    assert payload["evaluation_status"] == "demo_only"
    assert payload["sample_only"] is True
    assert "not sales" in payload["interpretation"]

    targets = {entry["target"]: entry for entry in payload["targets"]}
    assert targets["actionable_probability_mass"]["selected_model"] == "naive"
    assert targets["actionable_probability_mass"]["likelihood"] == "hurdle_gamma"
    assert targets["total_comments"]["usable"] is False


def test_forecast_summary_is_404_when_nothing_has_been_generated(
    client, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(services, "resolve_forecast_dir", lambda prefer_demo=None: tmp_path)

    response = client.get("/forecast/summary")

    assert response.status_code == 404
    assert "run_forecasting" in response.json()["detail"]


def test_behavioural_scenario_returns_difference_with_uncertainty(client, forecast_dir) -> None:
    response = client.post(
        "/forecast/scenario", json={"level_shift": 1.5, "label": "more_discussion"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["target"] == "actionable_probability_mass"
    assert payload["total_scenario"] > payload["total_baseline"]
    assert payload["total_difference_lower"] <= payload["total_difference_mean"]
    assert payload["total_difference_mean"] <= payload["total_difference_upper"]
    assert "Not a sales" in payload["interpretation"]
    assert payload["sample_only"] is True


def test_scenario_reports_missing_posterior_rather_than_fitting_a_model(
    client, tmp_path, monkeypatch
) -> None:
    """A web request must never silently start Bayesian sampling."""
    (tmp_path / "run_metadata.json").write_text(json.dumps({"targets": {}}))
    monkeypatch.setattr(services, "resolve_forecast_dir", lambda prefer_demo=None: tmp_path)

    response = client.post("/forecast/scenario", json={"level_shift": 1.2})

    assert response.status_code == 409
    assert "--save-posterior" in response.json()["detail"]


def test_scenario_input_is_validated(client, forecast_dir) -> None:
    for payload in [
        {"level_shift": -1.0},
        {"level_shift": 0.0},
        {"event_pulse": 100.0},
        {"pulse_periods": -3},
    ]:
        assert client.post("/forecast/scenario", json=payload).status_code == 422


def test_mmm_scenario_is_labelled_synthetic(client, tmp_path, monkeypatch) -> None:
    captured = {}

    def _fake_scenario(**kwargs):
        captured.update(kwargs)
        return {
            "notice": "SYNTHETIC MARKETING SCIENCE DEMONSTRATION — not Square Enix data.",
            "data_type": "synthetic",
            "scenario": {"label": kwargs.get("label")},
            "baseline_expected_sales": 100.0,
            "scenario_expected_sales": 110.0,
            "difference_mean": 10.0,
            "difference_lower": 2.0,
            "difference_upper": 18.0,
            "difference_percent": 10.0,
            "probability_of_increase": 0.9,
            "channel_contribution_change": {"video_spend": 5.0},
            "warnings": [],
            "interpretation": "Change in expected simulated sales.",
        }

    monkeypatch.setattr(services, "mmm_scenario", _fake_scenario)

    response = client.post(
        "/mmm/scenario",
        json={"budget_changes": {"social_spend": 0.8, "video_spend": 1.2}, "label": "shift"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data_type"] == "synthetic"
    assert "not Square Enix data" in payload["notice"]
    assert captured["budget_changes"] == {"social_spend": 0.8, "video_spend": 1.2}


def test_mmm_scenario_reports_missing_posterior(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        services,
        "mmm_scenario",
        lambda **kwargs: (_ for _ in ()).throw(services.ArtefactMissingError("no samples; run_mmm")),
    )

    response = client.post("/mmm/scenario", json={"budget_changes": {}})

    assert response.status_code == 409
    assert "run_mmm" in response.json()["detail"]


def test_the_api_exposes_exactly_the_four_documented_endpoints() -> None:
    """The API is deliberately thin; new endpoints belong in a considered change."""
    paths = {route.path for route in app.routes if getattr(route, "methods", None)}

    assert {"/health", "/forecast/summary", "/forecast/scenario", "/mmm/scenario"} <= paths
    business = {path for path in paths if not path.startswith("/openapi") and path not in {"/docs", "/redoc", "/docs/oauth2-redirect"}}
    assert business == {"/health", "/forecast/summary", "/forecast/scenario", "/mmm/scenario"}


def test_handlers_contain_no_business_logic() -> None:
    """Endpoint handlers must delegate to the shared service layer."""
    import inspect

    import src.api as api_module

    source = inspect.getsource(api_module)
    for forbidden in ["pm.sample", "fit_mmm(", "fit_bayesian_forecast(", "pd.read_csv"]:
        assert forbidden not in source

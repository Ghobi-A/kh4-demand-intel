"""Thin analytical API over the generated forecasting and MMM results.

Deliberately small: four endpoints, Pydantic models for the contract, and no
business logic. Every handler delegates to ``src.services``, which is the same
layer the Streamlit dashboard uses, so the two cannot drift apart.

The API never fits a model. It serves results produced by the experiment
scripts and reports clearly when a result has not been generated yet.

    pip install -e .[api]
    uvicorn src.api:app --reload
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import services

app = FastAPI(
    title="KH4 Behavioural Demand Intelligence API",
    version="0.3.0",
    description=(
        "Serves behavioural demand forecasts derived from public Kingdom Hearts IV "
        "community discussion, plus a clearly separated synthetic marketing mix "
        "modelling demonstration. Forecast targets are observable discussion "
        "proxies, not sales. The marketing endpoints are synthetic and are not "
        "Square Enix data."
    ),
)


class HealthResponse(BaseModel):
    """Which generated artefacts this deployment can serve."""

    status: str
    forecast_results_available: bool
    forecast_report_dir: str
    forecast_is_sample_only: bool
    mmm_results_available: bool
    mmm_posterior_available: bool


class ForecastSummaryResponse(BaseModel):
    """Headline forecasting results for every benchmarked target."""

    report_dir: str
    run_id: str | None = None
    generated_at: str | None = None
    git_commit: str | None = None
    evaluation_status: str | None = None
    sample_only: bool = False
    volume_representative: bool | None = None
    observed_period: dict[str, Any] | None = None
    targets: list[dict[str, Any]]
    interpretation: str


class BehaviouralScenarioRequest(BaseModel):
    """A hypothetical change to future behavioural demand."""

    target: str = Field(
        default="actionable_probability_mass",
        description="Which forecast target to run the scenario against.",
    )
    level_shift: float = Field(
        default=1.0, gt=0, le=10, description="Multiplier on the whole forecast."
    )
    event_pulse: float = Field(
        default=1.0, gt=0, le=10, description="Extra multiplier during the pulse periods."
    )
    pulse_periods: int = Field(default=1, ge=0, le=52)
    trajectory: float = Field(
        default=1.0, gt=0, le=5, description="Per-period compounding growth multiplier."
    )
    label: str = Field(default="scenario", max_length=80)


class BehaviouralScenarioResponse(BaseModel):
    """Baseline and scenario forecasts with their difference and uncertainty."""

    target: str
    scenario: dict[str, Any]
    horizon: int
    baseline_mean: list[float]
    scenario_mean: list[float]
    difference_mean: list[float]
    difference_lower: list[float]
    difference_upper: list[float]
    total_baseline: float
    total_scenario: float
    total_difference_mean: float
    total_difference_lower: float
    total_difference_upper: float
    evaluation_status: str | None = None
    sample_only: bool = False
    warnings: list[str] = Field(default_factory=list)
    interpretation: str


class MMMScenarioRequest(BaseModel):
    """A hypothetical marketing plan for the SYNTHETIC market."""

    budget_changes: dict[str, float] = Field(
        default_factory=dict,
        description="Per-channel spend multipliers, e.g. {'social_spend': 0.8, 'video_spend': 1.2}.",
    )
    price_change: float = Field(default=0.0, ge=-50, le=50)
    promotion: bool | None = None
    event: bool | None = None
    label: str = Field(default="scenario", max_length=80)


class MMMScenarioResponse(BaseModel):
    """Synthetic scenario outcome. Not a forecast of real performance."""

    notice: str
    data_type: str
    scenario: dict[str, Any]
    baseline_expected_sales: float
    scenario_expected_sales: float
    difference_mean: float
    difference_lower: float
    difference_upper: float
    difference_percent: float
    probability_of_increase: float
    channel_contribution_change: dict[str, float]
    warnings: list[str] = Field(default_factory=list)
    interpretation: str


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def get_health() -> dict:
    """Report which generated results are available."""
    return services.health()


@app.get("/forecast/summary", response_model=ForecastSummaryResponse, tags=["forecast"])
def get_forecast_summary() -> dict:
    """Headline forecasting results, including bias flags and diagnostics."""
    try:
        return services.forecast_summary()
    except services.ArtefactMissingError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/forecast/scenario", response_model=BehaviouralScenarioResponse, tags=["forecast"])
def post_forecast_scenario(request: BehaviouralScenarioRequest) -> dict:
    """Run a behavioural-demand scenario against the stored posterior."""
    try:
        return services.behavioural_scenario(
            level_shift=request.level_shift,
            event_pulse=request.event_pulse,
            pulse_periods=request.pulse_periods,
            trajectory=request.trajectory,
            target=request.target,
            label=request.label,
        )
    except services.ArtefactMissingError as exc:
        # 409: the request is valid, but the artefact it needs has not been generated.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/mmm/scenario", response_model=MMMScenarioResponse, tags=["synthetic-mmm"])
def post_mmm_scenario(request: MMMScenarioRequest) -> dict:
    """Run a budget-reallocation scenario on the SYNTHETIC marketing model."""
    try:
        return services.mmm_scenario(
            budget_changes=request.budget_changes,
            price_change=request.price_change,
            promotion=request.promotion,
            event=request.event,
            label=request.label,
        )
    except services.ArtefactMissingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


__all__ = ["app"]

"""Service layer shared by the Streamlit app and the FastAPI wrapper.

All business logic lives in the forecasting and MMM packages; this module
only loads what those packages generated and shapes it for a caller. Keeping
it here means the dashboard and the API answer identically, and neither
duplicates model code.

Nothing here fits a model. Bayesian fitting happens in the experiment scripts,
so a lightweight deployment can serve results without PyMC installed. Where a
scenario genuinely needs posterior samples, this module reports clearly that
they must be regenerated rather than silently fitting a model at request time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FORECAST_REPORT_DIR = Path("reports/forecasting")
DEMO_FORECAST_REPORT_DIR = Path("reports/demo_forecasting")
MMM_REPORT_DIR = Path("reports/mmm")

REGENERATE_FORECAST_HINT = (
    "Run `python scripts/run_forecasting.py --save-posterior` (add --demo for the "
    "sample dataset) to generate the posterior samples this scenario needs."
)
REGENERATE_MMM_HINT = (
    "Run `python scripts/run_mmm.py --save-posterior` to generate the posterior "
    "samples this scenario needs."
)


class ArtefactMissingError(RuntimeError):
    """Raised when a requested result has not been generated yet."""


@dataclass
class ForecastArtefacts:
    """Everything a caller needs about a completed forecasting run."""

    report_dir: Path
    metadata: dict
    comparison: pd.DataFrame
    forecast: pd.DataFrame
    monitoring: pd.DataFrame
    observed: pd.DataFrame
    bias_flags: dict
    diagnostics: dict


def resolve_forecast_dir(prefer_demo: bool | None = None) -> Path:
    """Pick the canonical report directory, falling back to the demo one.

    The canonical directory is only populated by a run over the full dataset,
    so a repository that has only ever run the smoke path serves the clearly
    labelled demo results rather than nothing.
    """
    if prefer_demo is True:
        return DEMO_FORECAST_REPORT_DIR
    if prefer_demo is False:
        return FORECAST_REPORT_DIR
    if (FORECAST_REPORT_DIR / "run_metadata.json").exists():
        return FORECAST_REPORT_DIR
    return DEMO_FORECAST_REPORT_DIR


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_forecast_artefacts(report_dir: Path | None = None) -> ForecastArtefacts:
    """Load a forecasting run's generated artefacts."""
    report_dir = Path(report_dir or resolve_forecast_dir())
    metadata_path = report_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise ArtefactMissingError(
            f"No forecasting results in {report_dir}. Build the weekly dataset with "
            "`python scripts/build_temporal_dataset.py` and run "
            "`python scripts/run_forecasting.py`."
        )
    return ForecastArtefacts(
        report_dir=report_dir,
        metadata=_read_json(metadata_path),
        comparison=_read_csv(report_dir / "model_comparison.csv"),
        forecast=_read_csv(report_dir / "forecast.csv"),
        monitoring=_read_csv(report_dir / "monitoring.csv"),
        observed=_read_csv(report_dir / "observed_series.csv"),
        bias_flags=_read_json(report_dir / "bias_flags.json"),
        diagnostics=_read_json(report_dir / "bayesian_diagnostics.json"),
    )


def forecast_summary(report_dir: Path | None = None) -> dict:
    """Headline forecasting results: what was run, what won, and how reliable."""
    artefacts = load_forecast_artefacts(report_dir)
    metadata = artefacts.metadata

    targets: list[dict] = []
    for target, details in metadata.get("targets", {}).items():
        if not details.get("usable"):
            targets.append({"target": target, "usable": False, "reason": details.get("reason")})
            continue

        comparison = artefacts.comparison
        rows = (
            comparison[comparison["target"] == target].to_dict(orient="records")
            if not comparison.empty and "target" in comparison.columns
            else []
        )
        forecast_rows = (
            artefacts.forecast[artefacts.forecast["target"] == target].to_dict(orient="records")
            if not artefacts.forecast.empty and "target" in artefacts.forecast.columns
            else []
        )
        bias = artefacts.bias_flags.get(target, {}).get("models", {})
        targets.append(
            {
                "target": target,
                "usable": True,
                "target_kind": details.get("target_kind"),
                "likelihood": details.get("likelihood"),
                "complexity_tier": details.get("complexity_tier"),
                "periods": details.get("periods"),
                "non_empty_periods": details.get("non_empty_periods"),
                "origins": len(details.get("origins") or []),
                "selected_model": details.get("selected_model"),
                "selection_reasoning": details.get("selection_reasoning"),
                "model_comparison": rows,
                "forecast": forecast_rows,
                "bias_flags": {
                    model: {
                        "persistent_bias": flags.get("persistent_bias"),
                        "mean_signed_error": flags.get("mean_signed_error"),
                        "interval_coverage": flags.get("interval_coverage"),
                        "interval_calibration": flags.get("interval_calibration"),
                    }
                    for model, flags in bias.items()
                },
                "bayesian_diagnostics": details.get("bayesian_diagnostics"),
            }
        )

    return {
        "report_dir": str(artefacts.report_dir),
        "run_id": metadata.get("run_id"),
        "generated_at": metadata.get("timestamp"),
        "git_commit": metadata.get("git_commit"),
        "evaluation_status": metadata.get("evaluation_status"),
        "sample_only": metadata.get("sample_only", False),
        "volume_representative": metadata.get("volume_representative"),
        "observed_period": metadata.get("observed_period"),
        "targets": targets,
        "interpretation": (
            "Targets are observable behavioural demand proxies derived from public "
            "community discussion. They are not sales, revenue, or units."
        ),
    }


def _load_posterior(path: Path, key: str = "draws") -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path) as payload:
        return payload[key] if key in payload else None


def behavioural_scenario(
    level_shift: float = 1.0,
    event_pulse: float = 1.0,
    pulse_periods: int = 1,
    trajectory: float = 1.0,
    target: str = "actionable_probability_mass",
    label: str = "scenario",
    report_dir: Path | None = None,
) -> dict:
    """Run a behavioural-demand scenario against stored posterior samples."""
    from src.forecasting.scenarios import BehaviouralScenario, run_behavioural_scenario

    report_dir = Path(report_dir or resolve_forecast_dir())
    draws = _load_posterior(report_dir / f"posterior_samples_{target}.npz")
    if draws is None:
        raise ArtefactMissingError(
            f"No posterior samples for {target!r} in {report_dir}. {REGENERATE_FORECAST_HINT}"
        )

    observed = None
    observed_frame = _read_csv(report_dir / "observed_series.csv")
    if not observed_frame.empty and "target" in observed_frame.columns:
        subset = observed_frame[observed_frame["target"] == target]
        if not subset.empty:
            observed = subset["actual"].to_numpy(dtype=float)

    scenario = BehaviouralScenario(
        level_shift=level_shift,
        event_pulse=event_pulse,
        pulse_periods=pulse_periods,
        trajectory=trajectory,
        label=label,
    )
    result = run_behavioural_scenario(draws, scenario, observed=observed)
    result["target"] = target
    result["report_dir"] = str(report_dir)
    metadata = _read_json(report_dir / "run_metadata.json")
    result["evaluation_status"] = metadata.get("evaluation_status")
    result["sample_only"] = metadata.get("sample_only", False)
    return result


def load_mmm_artefacts(report_dir: Path | None = None) -> dict:
    """Load the synthetic MMM's generated artefacts."""
    report_dir = Path(report_dir or MMM_REPORT_DIR)
    metadata_path = report_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise ArtefactMissingError(
            f"No MMM results in {report_dir}. Run `python scripts/run_mmm.py`."
        )
    return {
        "report_dir": report_dir,
        "metadata": _read_json(metadata_path),
        "contributions": _read_csv(report_dir / "contributions.csv"),
        "response_curves": _read_csv(report_dir / "response_curves.csv"),
        "recovery": _read_csv(report_dir / "recovery.csv"),
        "diagnostics": _read_json(report_dir / "diagnostics.json"),
        "scenario_example": _read_json(report_dir / "scenario_example.json"),
    }


def mmm_summary(report_dir: Path | None = None) -> dict:
    """Headline synthetic-MMM results, always labelled as synthetic."""
    from src.mmm import SYNTHETIC_NOTICE

    artefacts = load_mmm_artefacts(report_dir)
    metadata = artefacts["metadata"]
    return {
        "notice": SYNTHETIC_NOTICE,
        "data_type": "synthetic",
        "is_square_enix_data": False,
        "report_dir": str(artefacts["report_dir"]),
        "run_id": metadata.get("run_id"),
        "generated_at": metadata.get("timestamp"),
        "generator_seed": metadata.get("generator_seed"),
        "contributions": artefacts["contributions"].to_dict(orient="records"),
        "recovery": artefacts["recovery"].to_dict(orient="records"),
        "recovery_checks_passed": metadata.get("recovery_checks_passed"),
        "recovery_checks_total": metadata.get("recovery_checks_total"),
        "diagnostics": metadata.get("diagnostics_summary"),
        "identifiability_notes": metadata.get("identifiability_notes"),
    }


def mmm_scenario(
    budget_changes: dict | None = None,
    price_change: float = 0.0,
    promotion: bool | None = None,
    event: bool | None = None,
    label: str = "scenario",
    report_dir: Path | None = None,
    n_draws: int = 100,
) -> dict:
    """Run a synthetic marketing scenario, refitting the MMM only if needed.

    Scenarios need the fitted posterior. Rather than silently sampling a model
    inside a web request, this reports that the samples must be regenerated.
    """
    from src.mmm.model import MMMFit
    from src.mmm.scenarios import MarketingScenario, simulate_scenario
    from src.mmm.synthetic import CHANNELS, load_synthetic_dataset

    report_dir = Path(report_dir or MMM_REPORT_DIR)
    posterior_path = report_dir / "posterior_samples.npz"
    if not posterior_path.exists():
        raise ArtefactMissingError(
            f"No MMM posterior samples in {report_dir}. {REGENERATE_MMM_HINT}"
        )

    with np.load(posterior_path) as payload:
        posterior = {key: payload[key] for key in payload.files}

    data, _truth = load_synthetic_dataset()
    artefacts = load_mmm_artefacts(report_dir)
    channels = list(CHANNELS)
    spend = data[channels].to_numpy(dtype=float)

    fit = MMMFit(
        channels=channels,
        posterior=posterior,
        diagnostics=artefacts["metadata"].get("diagnostics_summary", {}),
        contributions=artefacts["contributions"],
        response_curves=artefacts["response_curves"],
        sampler=artefacts["metadata"].get("sampler", {}),
        training_ranges={
            **{
                channel: {
                    "min": float(spend[:, index].min()),
                    "max": float(spend[:, index].max()),
                    "mean": float(spend[:, index].mean()),
                }
                for index, channel in enumerate(channels)
            },
            "price": {"min": float(data["price"].min()), "max": float(data["price"].max())},
        },
    )

    scenario = MarketingScenario(
        budget_changes=budget_changes or {},
        price_change=price_change,
        promotion=promotion,
        event=event,
        label=label,
    )
    return simulate_scenario(fit, data, scenario, n_draws=n_draws)


def health() -> dict:
    """Which generated artefacts are present, for the health endpoint."""
    forecast_dir = resolve_forecast_dir()
    return {
        "status": "ok",
        "forecast_results_available": (forecast_dir / "run_metadata.json").exists(),
        "forecast_report_dir": str(forecast_dir),
        "forecast_is_sample_only": bool(
            _read_json(forecast_dir / "run_metadata.json").get("sample_only", False)
        ),
        "mmm_results_available": (MMM_REPORT_DIR / "run_metadata.json").exists(),
        "mmm_posterior_available": (MMM_REPORT_DIR / "posterior_samples.npz").exists(),
    }


__all__ = [
    "ArtefactMissingError",
    "ForecastArtefacts",
    "behavioural_scenario",
    "forecast_summary",
    "health",
    "load_forecast_artefacts",
    "load_mmm_artefacts",
    "mmm_scenario",
    "mmm_summary",
    "resolve_forecast_dir",
]

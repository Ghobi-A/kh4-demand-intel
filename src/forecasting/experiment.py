"""The forecasting experiment: backtest, select, monitor, report.

Model selection is deliberately multi-criterion. WAPE is primary, but a model
that wins on WAPE while over- or under-forecasting persistently, or whose
intervals are badly calibrated, is not declared best: those failures cost more
in planning than a slightly higher error. Ties go to the simpler model.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.baselines import BASELINE_REGISTRY, eligible_baselines
from src.forecasting.data import (
    DEFAULT_TARGETS,
    ExogBuilder,
    WeeklyDataset,
    build_exog_builder,
    load_weekly,
    target_kind,
    trim_leading_empty,
)
from src.forecasting.metrics import forecast_metrics
from src.forecasting.monitoring import BiasThresholds, flag_persistent_bias, monitoring_table
from src.forecasting.validation import backtest, rolling_origin_splits
from src.timestamps import complexity_tier

LOGGER = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = Path("reports/forecasting")
DEMO_REPORT_DIR = Path("reports/demo_forecasting")

DEMO_BANNER = (
    "> **Demo smoke run — not a substantive result.** This report was generated "
    "from the class-stratified 304-row portfolio extract, which is not "
    "volume-representative. It exists to prove the pipeline runs end to end. "
    "Measured forecasting results require a run over the full canonical dataset.\n"
)


@dataclass
class ForecastConfig:
    """Everything that decides what a forecasting run does."""

    targets: tuple = DEFAULT_TARGETS
    min_train: int = 12
    horizon: int = 2
    step: int = 2
    forecast_horizon: int = 4
    interval_level: float = 0.8
    seed: int = 42
    max_origins: int | None = None
    bias: BiasThresholds = field(default_factory=BiasThresholds)
    include_bayesian: bool = True
    #: Origins the Bayesian model is backtested over. Refitting the posterior at
    #: every origin is expensive, so it is scored on the most recent origins and
    #: compared against the baselines on exactly those origins.
    bayesian_backtest_origins: int = 3
    bayesian_draws: int = 1000
    bayesian_tune: int = 1000
    bayesian_chains: int = 2
    bayesian_target_accept: float = 0.9

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["targets"] = list(self.targets)
        return payload


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def select_best_model(comparison: pd.DataFrame, bias_flags: dict) -> tuple[str, list[str]]:
    """Choose a model on accuracy, bias and interval calibration together.

    WAPE ranks the candidates; mean absolute error and then simplicity break
    ties. A model is demoted only when its own evaluation *demonstrates* a
    problem: persistent directional bias, or interval coverage far from
    nominal, judged over enough forecasts to mean anything.

    A model evaluated over too few forecasts is never promoted over a more
    accurate one. Being unproven is not a merit, and selecting on it would
    reward whichever model was evaluated least — which on this project would
    systematically favour the expensive Bayesian model, since it is backtested
    over fewer origins than the baselines.

    Returns the chosen model and the reasoning, so a reader can see why the
    lowest-WAPE model was or was not selected.
    """
    if comparison.empty:
        return "", ["No models produced forecasts."]

    reasoning: list[str] = []
    ranked = comparison.sort_values(["wape", "mae", "complexity"], ascending=True).reset_index(
        drop=True
    )
    leader = ranked.iloc[0]
    reasoning.append(f"Lowest WAPE: {leader['model']} ({leader['wape']:.3f}).")

    def _flags(name: str) -> dict:
        return bias_flags.get("models", {}).get(str(name), {})

    def _has_evidence(name: str) -> bool:
        return bool(_flags(name).get("sufficient_evidence", True))

    def _is_disqualified(name: str) -> bool:
        flags = _flags(name)
        if not _has_evidence(name):
            return False
        if flags.get("persistent_bias"):
            return True
        coverage = flags.get("interval_coverage")
        nominal = bias_flags.get("thresholds", {}).get("interval_level", 0.8)
        return coverage is not None and abs(coverage - nominal) > 0.25

    unproven = [str(row["model"]) for _, row in ranked.iterrows() if not _has_evidence(row["model"])]
    if unproven:
        reasoning.append(
            "Evaluated over too few forecasts to judge, so not eligible for "
            f"selection over a more accurate model: {', '.join(unproven)}."
        )

    eligible = ranked[[_has_evidence(row["model"]) for _, row in ranked.iterrows()]]
    if eligible.empty:
        reasoning.append(
            "No model has enough forecasts to judge its reliability; the "
            "lowest-WAPE model is reported, but nothing here is established."
        )
        return str(leader["model"]), reasoning

    for _, row in eligible.iterrows():
        if not _is_disqualified(row["model"]):
            if row["model"] != leader["model"]:
                reasoning.append(
                    f"{leader['model']} was demoted: its evaluation shows persistent "
                    "directional bias or badly calibrated intervals, which cost more "
                    f"in planning than its WAPE advantage. Selected {row['model']}."
                )
            else:
                reasoning.append(
                    f"{row['model']} also passes the bias and interval-calibration checks."
                )
            return str(row["model"]), reasoning

    best_evaluated = eligible.iloc[0]
    reasoning.append(
        "Every model with enough forecasts to judge shows persistent bias or "
        "poor interval calibration. None of them is reliable on this history; "
        f"{best_evaluated['model']} is reported as the lowest-error option, not "
        "as a recommendation."
    )
    return str(best_evaluated["model"]), reasoning


def _run_bayesian(
    values: np.ndarray,
    target: str,
    tier: str,
    config: ForecastConfig,
    exog_builder: ExogBuilder | None,
    frame: pd.DataFrame,
) -> dict:
    """Fit the Bayesian model on the full history, returning results or a reason."""
    from src.forecasting.bayesian import (
        BayesianUnavailableError,
        SamplerConfig,
        fit_bayesian_forecast,
    )

    exog = exog_future = None
    if exog_builder is not None:
        exog, exog_future = exog_builder(len(values), config.forecast_horizon)

    sampler = SamplerConfig(
        draws=config.bayesian_draws,
        tune=config.bayesian_tune,
        chains=config.bayesian_chains,
        target_accept=config.bayesian_target_accept,
        seed=config.seed,
    )
    kind = target_kind(target)
    exposure = exposure_future = None
    if kind == "rate":
        exposure = frame["total_comments"].to_numpy(dtype=float)

    try:
        result = fit_bayesian_forecast(
            values,
            horizon=config.forecast_horizon,
            target=target,
            target_kind=kind,
            tier=tier,
            exog=exog,
            exog_future=exog_future,
            exposure=exposure,
            exposure_future=exposure_future,
            config=sampler,
            interval_level=config.interval_level,
        )
    except BayesianUnavailableError as exc:
        LOGGER.warning("Bayesian model skipped: %s", exc)
        return {"available": False, "reason": str(exc)}
    except ValueError as exc:
        LOGGER.warning("Bayesian model could not be fitted: %s", exc)
        return {"available": False, "reason": str(exc)}

    return {
        "available": True,
        "likelihood": result.likelihood,
        "tier": result.tier,
        "forecast_mean": result.mean.tolist(),
        "forecast_lower": result.lower.tolist(),
        "forecast_upper": result.upper.tolist(),
        "diagnostics": result.diagnostics,
        "posterior_predictive": result.posterior_predictive,
        "effects": result.effects,
        "notes": result.notes,
        "sampler": sampler.to_dict(),
        "_draws": result.draws,
    }


def _backtest_bayesian(
    values: np.ndarray,
    target: str,
    tier: str,
    config: ForecastConfig,
    exog_builder: ExogBuilder | None,
    frame: pd.DataFrame,
    splits: list,
) -> tuple[pd.DataFrame, list]:
    """Score the Bayesian model on the most recent origins.

    Refitting a posterior at every origin is too slow to be useful here, so the
    model is scored on the last few origins and the comparison table reports it
    alongside baselines evaluated on exactly those same origins. Scoring it on
    fewer origins is stated rather than hidden; not scoring it at all would mean
    it was never actually compared.
    """
    from src.forecasting.bayesian import (
        BayesianUnavailableError,
        SamplerConfig,
        fit_bayesian_forecast,
    )

    notes: list[str] = []
    chosen = splits[-config.bayesian_backtest_origins :] if config.bayesian_backtest_origins else []
    if not chosen:
        return pd.DataFrame(), ["Bayesian backtesting disabled."]

    kind = target_kind(target)
    sampler = SamplerConfig(
        draws=config.bayesian_draws,
        tune=config.bayesian_tune,
        chains=config.bayesian_chains,
        target_accept=config.bayesian_target_accept,
        seed=config.seed,
    )
    rows: list[dict] = []
    for split in chosen:
        train_values = values[split.train_index]
        horizon = len(split.test_index)
        exog = exog_future = None
        if exog_builder is not None:
            exog, exog_future = exog_builder(int(split.origin), horizon)
        exposure = exposure_future = None
        if kind == "rate":
            exposure = frame["total_comments"].to_numpy(dtype=float)[split.train_index]
            exposure_future = frame["total_comments"].to_numpy(dtype=float)[split.test_index]
        try:
            result = fit_bayesian_forecast(
                train_values,
                horizon=horizon,
                target=target,
                target_kind=kind,
                tier=tier,
                exog=exog,
                exog_future=exog_future,
                exposure=exposure,
                exposure_future=exposure_future,
                config=sampler,
                interval_level=config.interval_level,
            )
        except (BayesianUnavailableError, ValueError) as exc:
            LOGGER.warning("Bayesian backtest failed at origin %s: %s", split.origin, exc)
            notes.append(f"Origin {split.origin}: {exc}")
            continue
        for step, position in enumerate(split.test_index):
            rows.append(
                {
                    "model": "bayesian",
                    "origin": int(split.origin),
                    "horizon_step": step + 1,
                    "period_index": int(position),
                    "actual": float(values[position]),
                    "forecast": float(result.mean[step]),
                    "lower": float(result.lower[step]),
                    "upper": float(result.upper[step]),
                }
            )
    if rows:
        notes.append(
            f"The Bayesian model was backtested on the last {len(chosen)} origins "
            "because refitting the posterior at every origin is expensive. Its "
            "comparison row is therefore based on fewer origins than the baselines."
        )
    return pd.DataFrame(rows), notes


def run_target(
    dataset: WeeklyDataset,
    target: str,
    config: ForecastConfig,
    exog_builder: ExogBuilder | None,
) -> dict:
    """Backtest every eligible model on one target and assemble its results."""
    frame = trim_leading_empty(dataset.frame, target)
    series = pd.to_numeric(frame[target], errors="coerce")

    if series.isna().any():
        # A rate can be undefined in empty weeks; those periods cannot be
        # scored, so the target is restricted to the periods it is defined on.
        LOGGER.info("Dropping %s undefined periods from %s", int(series.isna().sum()), target)
        keep = series.notna()
        frame = frame[keep].reset_index(drop=True)
        series = series[keep].reset_index(drop=True)

    values = series.to_numpy(dtype=float)
    n = values.size
    non_empty = int((values > 0).sum())
    tier = complexity_tier(non_empty)

    splits = rolling_origin_splits(
        n,
        min_train=config.min_train,
        horizon=config.horizon,
        step=config.step,
        max_origins=config.max_origins,
    )
    if not splits:
        return {
            "target": target,
            "usable": False,
            "reason": (
                f"Only {n} periods after trimming; a rolling-origin backtest needs at "
                f"least min_train ({config.min_train}) + horizon ({config.horizon})."
            ),
        }

    period_builder = None
    if exog_builder is not None:
        period_builder = ExogBuilder(frame["period_start"], exog_builder.events, exog_builder.columns)

    results: list[pd.DataFrame] = []
    for name in eligible_baselines(n, min_train=config.min_train):
        factory = BASELINE_REGISTRY[name]
        needs_exog = name == "ridge_event"
        outcome = backtest(
            lambda cls=factory: cls(interval_level=config.interval_level),
            series,
            splits,
            exog_builder=period_builder if needs_exog else None,
            model_name=name,
        )
        if not outcome.empty:
            results.append(outcome)

    if not results:
        return {"target": target, "usable": False, "reason": "No model produced a forecast."}

    backtest_results = pd.concat(results, ignore_index=True)

    bayesian_backtest = pd.DataFrame()
    bayesian_notes: list[str] = []
    if config.include_bayesian:
        bayesian_backtest, bayesian_notes = _backtest_bayesian(
            values, target, tier, config, period_builder, frame, splits
        )
        if not bayesian_backtest.empty:
            backtest_results = pd.concat(
                [backtest_results, bayesian_backtest], ignore_index=True
            )

    monitoring = monitoring_table(backtest_results, config.bias)
    bias_flags = flag_persistent_bias(monitoring, config.bias)

    def _metrics_for(group: pd.DataFrame, name: str, origins: int) -> dict:
        metrics = forecast_metrics(
            group["actual"], group["forecast"], group["lower"], group["upper"],
            insample=values,
        )
        metrics["model"] = name
        metrics["complexity"] = (
            BASELINE_REGISTRY[name]().complexity if name in BASELINE_REGISTRY else 5
        )
        flags = bias_flags["models"].get(str(name), {})
        persistent = flags.get("persistent_bias")
        metrics["persistent_bias"] = "unknown" if persistent is None else bool(persistent)
        metrics["interval_calibration"] = flags.get("interval_calibration", "unknown")
        # Accuracy here is computed on this table's own rows; the bias and
        # calibration verdicts come from each model's full evaluation, which is
        # a larger sample for the baselines than for the Bayesian model. The
        # column is named so the two are not read as the same denominator.
        metrics["bias_evidence_forecasts"] = int(flags.get("n_forecasts", len(group)))
        metrics["origins"] = origins
        return metrics

    rows = [
        _metrics_for(group, str(name), int(group["origin"].nunique()))
        for name, group in backtest_results.groupby("model")
    ]
    comparison = pd.DataFrame(rows)

    # Like-for-like: baselines re-scored on exactly the origins the Bayesian
    # model was scored on, so the two are never compared across different
    # evaluation windows.
    matched_comparison = pd.DataFrame()
    if not bayesian_backtest.empty:
        origins = set(bayesian_backtest["origin"])
        subset = backtest_results[backtest_results["origin"].isin(origins)]
        matched_comparison = pd.DataFrame(
            [
                _metrics_for(group, str(name), len(origins))
                for name, group in subset.groupby("model")
            ]
        )

    selection_source = matched_comparison if not matched_comparison.empty else comparison
    selected, reasoning = select_best_model(selection_source, bias_flags)
    if not matched_comparison.empty:
        reasoning.append(
            "Selection used the origins on which every model, including the "
            "Bayesian one, was scored."
        )
    reasoning.extend(bayesian_notes)

    bayesian = {"available": False, "reason": "Bayesian modelling disabled for this run."}
    if config.include_bayesian:
        bayesian = _run_bayesian(values, target, tier, config, period_builder, frame)

    forecast_rows = []
    if bayesian.get("available"):
        last_period = frame["period_start"].iloc[-1]
        for step in range(config.forecast_horizon):
            forecast_rows.append(
                {
                    "period_start": last_period + pd.Timedelta(weeks=step + 1),
                    "horizon_step": step + 1,
                    "model": "bayesian",
                    "forecast": bayesian["forecast_mean"][step],
                    "lower": bayesian["forecast_lower"][step],
                    "upper": bayesian["forecast_upper"][step],
                }
            )

    return {
        "target": target,
        "usable": True,
        "target_kind": target_kind(target),
        "periods": int(n),
        "non_empty_periods": non_empty,
        "zero_period_share": float(np.mean(values <= 0)),
        "complexity_tier": tier,
        "origins": [int(split.origin) for split in splits],
        "backtest": backtest_results,
        "monitoring": monitoring,
        "comparison": comparison,
        "matched_comparison": matched_comparison,
        "bias_flags": bias_flags,
        "selected_model": selected,
        "selection_reasoning": reasoning,
        "bayesian": bayesian,
        "forecast": pd.DataFrame(forecast_rows),
        "observed": series,
        "periods_index": frame["period_start"],
    }


def run_experiment(
    input_path: Path,
    report_dir: Path | None = None,
    config: ForecastConfig | None = None,
    demo: bool = False,
    save_posterior: bool = False,
) -> dict:
    """Run the full forecasting experiment and write every artefact."""
    config = config or ForecastConfig()
    report_dir = Path(report_dir or (DEMO_REPORT_DIR if demo else DEFAULT_REPORT_DIR))
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_weekly(input_path)
    if demo and not dataset.is_sample_only:
        LOGGER.warning("--demo was passed but %s is not marked sample_only", input_path)
    if not demo and dataset.is_sample_only:
        raise ValueError(
            f"{input_path} is marked sample_only. The stratified sample must not populate "
            f"{DEFAULT_REPORT_DIR}; re-run with --demo to write to {DEMO_REPORT_DIR}."
        )

    exog_builder = build_exog_builder(dataset)
    targets = dataset.available_targets(config.targets)
    if not targets:
        raise ValueError(
            f"None of the requested targets {list(config.targets)} exist in {input_path}"
        )

    per_target: dict[str, dict] = {}
    all_backtests, all_monitoring, all_comparisons, all_forecasts = [], [], [], []
    all_matched: list[pd.DataFrame] = []

    for target in targets:
        LOGGER.info("Running target %s", target)
        outcome = run_target(dataset, target, config, exog_builder)
        per_target[target] = outcome
        if not outcome.get("usable"):
            LOGGER.warning("Target %s unusable: %s", target, outcome.get("reason"))
            continue
        for frame, sink in [
            (outcome["backtest"], all_backtests),
            (outcome["monitoring"], all_monitoring),
            (outcome["comparison"], all_comparisons),
            (outcome["forecast"], all_forecasts),
            (outcome.get("matched_comparison"), all_matched),
        ]:
            if frame is not None and not frame.empty:
                tagged = frame.copy()
                tagged.insert(0, "target", target)
                sink.append(tagged)

        if save_posterior and outcome["bayesian"].get("_draws") is not None:
            path = report_dir / f"posterior_samples_{target}.npz"
            np.savez_compressed(path, draws=outcome["bayesian"]["_draws"])
            LOGGER.info("Wrote posterior samples to %s", path)

    def _write(frames, name):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(report_dir / name, index=False)

    _write(all_backtests, "backtest_results.csv")
    _write(all_monitoring, "monitoring.csv")
    _write(all_comparisons, "model_comparison.csv")
    _write(all_forecasts, "forecast.csv")
    _write(all_matched, "model_comparison_matched_origins.csv")

    observed_frames = [
        pd.DataFrame(
            {
                "target": target,
                "period_start": outcome["periods_index"],
                "actual": outcome["observed"].to_numpy(),
            }
        )
        for target, outcome in per_target.items()
        if outcome.get("usable")
    ]
    _write(observed_frames, "observed_series.csv")

    bias_payload = {
        target: outcome["bias_flags"]
        for target, outcome in per_target.items()
        if outcome.get("usable")
    }
    (report_dir / "bias_flags.json").write_text(json.dumps(bias_payload, indent=2, default=str))

    diagnostics = {
        target: {
            key: value
            for key, value in outcome["bayesian"].items()
            if not key.startswith("_")
        }
        for target, outcome in per_target.items()
        if outcome.get("usable")
    }
    (report_dir / "bayesian_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str)
    )

    coverage = dataset.meta.get("coverage", {})
    (report_dir / "temporal_coverage.json").write_text(json.dumps(coverage, indent=2, default=str))

    metadata = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_path": str(input_path),
        "input_hash": dataset.meta.get("source_dataset_hash"),
        "observed_period": {
            "start": dataset.meta.get("aggregation", {}).get("period_start_min"),
            "end": dataset.meta.get("aggregation", {}).get("period_start_max"),
        },
        "evaluation_status": dataset.evaluation_status,
        "sample_only": dataset.is_sample_only,
        "volume_representative": dataset.meta.get("volume_representative"),
        "config": config.to_dict(),
        "seed": config.seed,
        "targets": {
            target: {
                "usable": outcome.get("usable", False),
                "reason": outcome.get("reason"),
                "target_kind": outcome.get("target_kind"),
                "periods": outcome.get("periods"),
                "non_empty_periods": outcome.get("non_empty_periods"),
                "zero_period_share": outcome.get("zero_period_share"),
                "complexity_tier": outcome.get("complexity_tier"),
                "origins": outcome.get("origins"),
                "selected_model": outcome.get("selected_model"),
                "selection_reasoning": outcome.get("selection_reasoning"),
                "likelihood": outcome.get("bayesian", {}).get("likelihood"),
                "likelihood_notes": outcome.get("bayesian", {}).get("notes"),
                "sampler": outcome.get("bayesian", {}).get("sampler"),
                "bayesian_diagnostics": {
                    key: value
                    for key, value in outcome.get("bayesian", {}).get("diagnostics", {}).items()
                    if key != "parameters"
                },
            }
            for target, outcome in per_target.items()
        },
        "probability_provenance": dataset.meta.get("probabilities", {}),
    }
    (report_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    from src.forecasting.reporting import write_forecast_report

    write_forecast_report(per_target, metadata, dataset, report_dir, demo=demo)

    return {"per_target": per_target, "metadata": metadata, "report_dir": report_dir}


__all__ = [
    "DEMO_BANNER",
    "DEMO_REPORT_DIR",
    "DEFAULT_REPORT_DIR",
    "ForecastConfig",
    "run_experiment",
    "run_target",
    "select_best_model",
]

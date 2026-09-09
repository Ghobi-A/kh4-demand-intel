"""Markdown report generation for forecasting runs.

Every number is rendered from the run's own results. Nothing is typed by hand,
so a report can never drift from the artefacts beside it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LIMITATIONS = [
    "Community discussion is not unit sales. These targets measure observable "
    "public behaviour, and Kingdom Hearts IV is unreleased, so no purchase "
    "conversion exists to validate against.",
    "Sampling is platform-dependent. Comment volume reflects what was collected, "
    "which videos and threads existed, and how platform algorithms surfaced them.",
    "The labelled intent corpus is still small, so the calibrated probabilities "
    "underlying the demand proxy are development-preliminary.",
    "Event coefficients describe association, not causation. An uplift after a "
    "trailer is a modelled event response, not proof the trailer caused it.",
    "Uncertainty grows with the horizon; long-range forecasts on this history "
    "are weak and their intervals should be read as such.",
]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, (bool,)):
        return "yes" if value else "no"
    if pd.isna(value):
        return "—"
    if isinstance(value, (int,)) or (isinstance(value, float) and float(value).is_integer()):
        return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"
    return f"{value:.{digits}f}"


def _comparison_table(comparison: pd.DataFrame) -> list[str]:
    columns = [
        ("model", "Model"),
        ("wape", "WAPE"),
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("bias", "Signed bias"),
        ("interval_coverage", "Coverage"),
        ("mean_interval_width", "Interval width"),
        ("persistent_bias", "Persistent bias"),
        ("interval_calibration", "Interval calibration"),
        ("origins", "Origins"),
        ("bias_evidence_forecasts", "Bias evidence (forecasts)"),
    ]
    present = [(key, label) for key, label in columns if key in comparison.columns]
    lines = [
        "| " + " | ".join(label for _, label in present) + " |",
        "|" + "|".join(["---"] * len(present)) + "|",
    ]
    for _, row in comparison.sort_values("wape").iterrows():
        cells = []
        for key, _ in present:
            value = row[key]
            cells.append(
                ("yes" if value else "no") if key == "persistent_bias" else _fmt(value)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_forecast_report(
    per_target: dict,
    metadata: dict,
    dataset,
    report_dir: Path,
    demo: bool = False,
) -> Path:
    """Render forecast_report.md from the run's generated results."""
    from src.forecasting.experiment import DEMO_BANNER

    coverage = dataset.meta.get("coverage", {})
    lines: list[str] = [
        "# Behavioural demand forecasting report",
        "",
    ]
    if demo or metadata.get("sample_only"):
        lines += [DEMO_BANNER, ""]

    lines += [
        f"- Run: `{metadata['run_id']}` at {metadata['timestamp']}",
        f"- Git commit: `{metadata['git_commit']}`",
        f"- Input: `{metadata['input_path']}` (hash `{metadata.get('input_hash')}`)",
        f"- Evaluation status: `{metadata['evaluation_status']}`",
        f"- Volume representative: {metadata.get('volume_representative')}",
        f"- Seed: {metadata['seed']}",
        "",
        "## Target definitions",
        "",
        "`actionable_probability_mass` is the sum of calibrated P(actionable) over "
        "the comments observed in a period: an **expected actionable-signal volume**, "
        "or behavioural demand proxy. It is not expected purchases, revenue, or units. "
        "`total_comments` and `actionable_count` are raw observable counts.",
        "",
        "## Temporal coverage",
        "",
        f"- Observed periods: {coverage.get('observed_periods', '—')}",
        f"- Non-empty periods: {coverage.get('non_empty_periods', '—')}",
        f"- Longest consecutive non-empty run: {coverage.get('longest_dense_run', '—')}",
        f"- Usable event periods: {coverage.get('usable_event_periods', '—')}",
        "",
        "Model complexity is chosen from this measured coverage, not assumed in advance.",
        "",
        "## Validation design",
        "",
        f"Rolling-origin (walk-forward) backtesting: minimum train "
        f"{metadata['config']['min_train']} periods, horizon "
        f"{metadata['config']['horizon']}, step {metadata['config']['step']}. "
        "Random splits are never used, and exogenous event features are rebuilt at "
        "each origin so an event that had not been announced yet cannot inform "
        "that origin's forecast.",
        "",
    ]

    for target, outcome in per_target.items():
        lines += [f"## Target: `{target}`", ""]
        if not outcome.get("usable"):
            lines += [f"Not modelled: {outcome.get('reason')}", ""]
            continue

        lines += [
            f"- Kind: `{outcome['target_kind']}`",
            f"- Periods modelled: {outcome['periods']} "
            f"({outcome['non_empty_periods']} non-empty, "
            f"{outcome['zero_period_share']:.0%} zero)",
            f"- Complexity tier: `{outcome['complexity_tier']}`",
            f"- Forecast origins: {len(outcome['origins'])}",
            "",
            "### Model comparison",
            "",
        ]
        lines += _comparison_table(outcome["comparison"])

        matched = outcome.get("matched_comparison")
        if matched is not None and not matched.empty:
            lines += [
                "",
                "#### Matched-origin comparison (includes the Bayesian model)",
                "",
                "Refitting the posterior at every origin is expensive, so the "
                "Bayesian model is scored on the most recent origins. This table "
                "re-scores every model on exactly those origins, so no two models "
                "are compared across different evaluation windows. Accuracy is "
                "therefore like-for-like, while the bias and calibration verdicts "
                "use each model's full evaluation — a much larger sample for the "
                "baselines, which is why the last column differs by row.",
                "",
            ]
            lines += _comparison_table(matched)

        lines += [
            "",
            f"**Selected model: `{outcome['selected_model']}`**",
            "",
        ]
        lines += [f"- {reason}" for reason in outcome["selection_reasoning"]]
        lines += ["", "### Forecast bias and reliability", ""]

        for model, flags in outcome["bias_flags"]["models"].items():
            if not flags.get("sufficient_evidence", True):
                lines.append(
                    f"- `{model}`: only {flags.get('n_forecasts')} forecasts, too few to "
                    "call bias or interval calibration either way. Mean signed error "
                    f"{_fmt(flags.get('mean_signed_error'))}, interval coverage "
                    f"{_fmt(flags.get('interval_coverage'))}, both reported as "
                    "observations rather than verdicts."
                )
                continue
            runs = flags.get("bias_runs", [])
            description = (
                ", ".join(
                    f"{run['direction'].replace('_', ' ')} for {run['length']} periods"
                    for run in runs
                )
                or "no persistent directional bias"
            )
            lines.append(
                f"- `{model}`: mean signed error {_fmt(flags.get('mean_signed_error'))}, "
                f"interval coverage {_fmt(flags.get('interval_coverage'))} "
                f"({flags.get('interval_calibration')}) over "
                f"{flags.get('n_forecasts')} forecasts; {description}."
            )

        bayesian = outcome.get("bayesian", {})
        lines += ["", "### Bayesian model", ""]
        if not bayesian.get("available"):
            lines += [f"Not fitted: {bayesian.get('reason')}", ""]
        else:
            diagnostics = bayesian.get("diagnostics", {})
            lines += [
                f"- Likelihood: `{bayesian['likelihood']}` (complexity tier "
                f"`{bayesian['tier']}`)",
                f"- Sampler: {bayesian['sampler']['chains']} chains, "
                f"{bayesian['sampler']['draws']} draws, "
                f"{bayesian['sampler']['tune']} tune, target_accept "
                f"{bayesian['sampler']['target_accept']}",
                f"- Max R-hat: {_fmt(diagnostics.get('max_r_hat'))}, "
                f"min bulk ESS: {_fmt(diagnostics.get('min_ess_bulk'))}, "
                f"divergences: {diagnostics.get('divergences')}",
                f"- Converged: {diagnostics.get('converged')}",
            ]
            for warning in diagnostics.get("warnings", []):
                lines.append(f"  - Warning: {warning}")
            for note in bayesian.get("notes", []):
                lines.append(f"- {note}")
            effects = bayesian.get("effects", {})
            if effects:
                lines += [
                    "",
                    "| Parameter | Posterior mean | 5% | 95% |",
                    "|---|---|---|---|",
                ]
                for name, values in effects.items():
                    lines.append(
                        f"| `{name}` | {_fmt(values['mean'])} | {_fmt(values['hdi_5'])} "
                        f"| {_fmt(values['hdi_95'])} |"
                    )
                lines += [
                    "",
                    "Event terms describe an association between an event window and the "
                    "observed proxy. They are not causal estimates.",
                ]
            forecast = outcome.get("forecast")
            if forecast is not None and not forecast.empty:
                lines += [
                    "",
                    "### Forecast",
                    "",
                    "| Period | Forecast | Lower | Upper |",
                    "|---|---|---|---|",
                ]
                for _, row in forecast.iterrows():
                    lines.append(
                        f"| {str(row['period_start'])[:10]} | {_fmt(row['forecast'])} "
                        f"| {_fmt(row['lower'])} | {_fmt(row['upper'])} |"
                    )
        lines.append("")

    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    lines.append("")

    path = Path(report_dir) / "forecast_report.md"
    path.write_text("\n".join(lines))
    return path

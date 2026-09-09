"""Streamlit dashboard for Kingdom Hearts IV demand intelligence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ML_REPORT_DIR = Path("reports/ml")
ARTIFACT_METADATA_PATH = Path("artifacts/hierarchical/metadata.json")
DEFAULT_SIGNALS_PATH = "data/demo/signals_scored_portfolio.csv"
DEFAULT_VIDEO_SCORES_PATH = "data/demo/video_scores_portfolio.csv"
SIGNALS_PATH = Path(os.getenv("SIGNALS_PATH", DEFAULT_SIGNALS_PATH))
VIDEO_SCORES_PATH = Path(os.getenv("VIDEO_SCORES_PATH", DEFAULT_VIDEO_SCORES_PATH))
REQUIRED_SIGNAL_COLUMNS = {
    "intent_label",
    "sentiment_label",
    "demand_score",
    "activation_score",
    "risk_score",
}
REQUIRED_VIDEO_COLUMNS = {"parent_id", "mean_demand_score"}


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and cache it for repeat dashboard interactions."""
    return pd.read_csv(path)


def positive_share(df: pd.DataFrame) -> float:
    """Return the share of rows labelled positive."""
    if df.empty or "sentiment_label" not in df.columns:
        return 0.0
    return float((df["sentiment_label"] == "positive").mean())


def negative_share(df: pd.DataFrame) -> float:
    """Return the share of rows labelled negative."""
    if df.empty or "sentiment_label" not in df.columns:
        return 0.0
    return float((df["sentiment_label"] == "negative").mean())


def sorted_unique_values(df: pd.DataFrame, column: str) -> list[str]:
    """Return sorted non-null values as strings for sidebar filters."""
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def filter_dataframe(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    """Apply selected multi-select filters to a dataframe."""
    filtered = df.copy()
    for column, selected_values in filters.items():
        if selected_values and column in filtered.columns:
            filtered = filtered[filtered[column].astype(str).isin(selected_values)]
    return filtered


def require_files() -> bool:
    """Show clean guidance when dashboard input files are missing."""
    missing_paths = [path for path in [SIGNALS_PATH, VIDEO_SCORES_PATH] if not path.exists()]
    if not missing_paths:
        return True

    st.warning(
        "Dashboard data files are missing, so charts cannot be rendered yet. "
        "Committed demo data should exist for clean deployment; full local "
        "pipeline outputs can be supplied with environment variables."
    )
    st.markdown(
        "Missing dashboard inputs:\n"
        + "\n".join(f"- `{path}`" for path in missing_paths)
    )
    st.info(
        "By default the app loads `data/demo/signals_scored_portfolio.csv` and "
        "`data/demo/video_scores_portfolio.csv`. To use full local data, set "
        "`SIGNALS_PATH` and `VIDEO_SCORES_PATH` before running Streamlit. "
        "If `data/processed/signals_intent.csv` already exists, regenerate "
        "local scored outputs with: `python -m src.score`."
    )
    return False


def validate_columns(
    signals_df: pd.DataFrame,
    video_scores_df: pd.DataFrame,
) -> bool:
    """Validate expected dashboard columns and show actionable messages."""
    missing_signal_columns = REQUIRED_SIGNAL_COLUMNS - set(signals_df.columns)
    missing_video_columns = REQUIRED_VIDEO_COLUMNS - set(video_scores_df.columns)

    if not missing_signal_columns and not missing_video_columns:
        return True

    st.warning("Dashboard inputs were found, but some expected columns are missing.")
    if missing_signal_columns:
        st.markdown(
            f"Missing columns in `{SIGNALS_PATH}`: "
            f"`{sorted(missing_signal_columns)}`"
        )
    if missing_video_columns:
        st.markdown(
            f"Missing columns in `{VIDEO_SCORES_PATH}`: "
            f"`{sorted(missing_video_columns)}`"
        )
    st.info(
        "Regenerate the dashboard inputs by running the pipeline through sentiment "
        "→ intent → score. If `data/processed/signals_intent.csv` already exists, "
        "run: `python -m src.score`."
    )
    return False


def render_header() -> None:
    """Render title and project framing."""
    st.title("Kingdom Hearts IV Demand Intelligence")
    st.write(
        "A lightweight portfolio dashboard for exploring Reddit and YouTube "
        "community discussion around Kingdom Hearts IV as sentiment, intent, "
        "activation, risk, and heuristic demand signals."
    )
    with st.expander("About this project", expanded=False):
        st.markdown(
            "**Problem:** Fan sentiment alone does not reveal whether discussion "
            "signals purchase intent, reactivation, confusion, or churn risk.\n\n"
            "**Method:** Public community discussion is cleaned, labelled with an "
            "evaluated rule-based intent taxonomy, scored, and aggregated into a "
            "recruiter-friendly Streamlit dashboard.\n\n"
            "**Key finding:** Positive sentiment is not the same as behavioural "
            "intent; nostalgia, new-customer interest, frustration, and confusion "
            "need separate treatment.\n\n"
            "**Validation:** A 200-row manual audit and regression tests enforce "
            "precision floors for the highest-value intent classes."
        )


def render_sidebar_filters(signals_df: pd.DataFrame) -> dict[str, list[str]]:
    """Render optional sidebar filters and return selected values."""
    st.sidebar.header("Filters")
    filters: dict[str, list[str]] = {}

    for column in ["source", "intent_label", "sentiment_label", "parent_id"]:
        values = sorted_unique_values(signals_df, column)
        if values:
            filters[column] = st.sidebar.multiselect(
                label=column,
                options=values,
                default=[],
            )

    return filters


def render_kpis(filtered_df: pd.DataFrame) -> None:
    """Render headline KPI cards."""
    total_signals = len(filtered_df)
    activation_signals = int(filtered_df["activation_score"].sum())
    risk_signals = int(filtered_df["risk_score"].sum())

    columns = st.columns(5)
    columns[0].metric("Total signals", f"{total_signals:,}")
    columns[1].metric("Positive share", f"{positive_share(filtered_df):.1%}")
    columns[2].metric("Negative share", f"{negative_share(filtered_df):.1%}")
    columns[3].metric("Activation signals", f"{activation_signals:,}")
    columns[4].metric("Risk signals", f"{risk_signals:,}")


def render_charts(filtered_df: pd.DataFrame, video_scores_df: pd.DataFrame) -> None:
    """Render distribution and score charts."""
    st.header("Signal overview")

    chart_columns = st.columns(2)
    sentiment_counts = filtered_df["sentiment_label"].value_counts().sort_index()
    intent_counts = filtered_df["intent_label"].value_counts().sort_values(ascending=False)

    with chart_columns[0]:
        st.subheader("Sentiment distribution")
        st.bar_chart(sentiment_counts)

    with chart_columns[1]:
        st.subheader("Intent distribution")
        st.bar_chart(intent_counts)

    st.subheader("Activation vs risk summary")
    activation_risk = pd.DataFrame(
        {
            "signal_type": ["activation", "risk"],
            "count": [
                int(filtered_df["activation_score"].sum()),
                int(filtered_df["risk_score"].sum()),
            ],
        }
    ).set_index("signal_type")
    st.bar_chart(activation_risk)

    st.subheader("Overall top videos by mean demand score")
    top_videos = video_scores_df.sort_values(
        "mean_demand_score",
        ascending=False,
    ).head(15)
    st.bar_chart(top_videos.set_index("parent_id")["mean_demand_score"])


def render_tables(filtered_df: pd.DataFrame, video_scores_df: pd.DataFrame) -> None:
    """Render row-level and video-level tables."""
    st.header("Priority tables")

    st.subheader("Top 25 highest demand rows")
    st.dataframe(
        filtered_df.sort_values("demand_score", ascending=False).head(25),
        use_container_width=True,
    )

    st.subheader("Top 25 risk rows")
    risk_rows = filtered_df[filtered_df["risk_score"] > 0]
    st.dataframe(
        risk_rows.sort_values("demand_score", ascending=False).head(25),
        use_container_width=True,
    )

    st.subheader("Video-level score table")
    st.dataframe(
        video_scores_df.sort_values("mean_demand_score", ascending=False),
        use_container_width=True,
    )


def render_explainability() -> None:
    """Render concise scoring explanation."""
    st.header("How to read the scores")
    st.markdown(
        "- `demand_score` is a heuristic score calculated as "
        "`(intent_weight + sentiment_weight) * "
        "(1 + log1p(clipped_engagement) / 5)`, so engagement amplifies "
        "the intent/sentiment signal without turning the table into a raw "
        "popularity leaderboard.\n"
        "- `activation_score` marks rows with purchase, reactivation, or new-player "
        "interest signals.\n"
        "- `risk_score` marks rows with frustration, confusion, fatigue, or expectation "
        "decay signals.\n"
        "- This is a heuristic decision-support prototype for portfolio review and "
        "analysis, not a forecast or production demand model."
    )


def model_status() -> str:
    """Read the model evaluation status from run/artefact metadata."""
    for path in [ARTIFACT_METADATA_PATH, ML_REPORT_DIR / "run_metadata.json"]:
        if path.exists():
            status = json.loads(path.read_text()).get("evaluation_status", "")
            if status == "final_held_out":
                return "Final held-out evaluation"
            if status:
                return "Development / preliminary evaluation"
    return "Development / preliminary evaluation"


def _show_markdown_report(path: Path, missing_hint: str) -> bool:
    if not path.exists():
        st.info(missing_hint)
        return False
    st.markdown(path.read_text())
    return True


def render_demand_tab(signals_df: pd.DataFrame, video_scores_df: pd.DataFrame) -> None:
    """Original demand-intelligence dashboard content."""
    filters = render_sidebar_filters(signals_df)
    filtered_df = filter_dataframe(signals_df, filters)

    if filtered_df.empty:
        st.warning("No rows match the selected filters. Clear filters to see results.")
        return

    render_kpis(filtered_df)
    render_charts(filtered_df, video_scores_df)
    render_tables(filtered_df, video_scores_df)
    render_explainability()


def render_model_performance_tab() -> None:
    st.header("Model performance")
    st.caption(f"Model status: {model_status()}")
    summary_path = ML_REPORT_DIR / "results_summary.csv"
    if not summary_path.exists():
        st.info(
            "No benchmark results found. Run `python -m src.ml.benchmark` to "
            "generate reports under `reports/ml/`."
        )
        return
    summary = pd.read_csv(summary_path)
    headline_columns = [
        column
        for column in [
            "model",
            "n_seeds",
            "stage1_f1_mean",
            "stage1_f1_std",
            "stage1_pr_auc_mean",
            "stage2_macro_f1_mean",
            "flat_macro_f1_mean",
            "e2e_macro_f1_mean",
            "e2e_macro_f1_std",
            "lift_at_10pct_mean",
        ]
        if column in summary.columns
    ]
    st.subheader("Benchmark summary (mean over seeds)")
    st.dataframe(summary[headline_columns], use_container_width=True)

    figure_columns = st.columns(2)
    calibration_path = ML_REPORT_DIR / "figures" / "calibration_stage1.png"
    gains_path = ML_REPORT_DIR / "figures" / "cumulative_gains.png"
    if calibration_path.exists():
        figure_columns[0].image(str(calibration_path), caption="Stage 1 calibration")
    if gains_path.exists():
        figure_columns[1].image(str(gains_path), caption="Cumulative gains")
    st.warning(
        "Preliminary development benchmark on the current 200-row audit "
        "corpus — not final held-out performance."
    )


def render_model_comparison_tab() -> None:
    st.header("Model comparison")
    _show_markdown_report(
        ML_REPORT_DIR / "model_comparison.md",
        "No model-comparison report yet. Run `python -m src.ml.benchmark`.",
    )
    _show_markdown_report(
        ML_REPORT_DIR / "rule_model_comparison.md",
        "No rule-vs-model report yet. Run `python -m src.ml.error_analysis`.",
    )


def render_error_analysis_tab() -> None:
    st.header("Error analysis")
    tables_dir = ML_REPORT_DIR / "tables"
    shown = False
    for filename, title in [
        ("false_positives.csv", "Stage 1 false positives"),
        ("false_negatives.csv", "Stage 1 false negatives"),
        ("most_confident_errors.csv", "Most confident errors"),
    ]:
        path = tables_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            st.subheader(f"{title} ({len(frame)})")
            st.dataframe(frame, use_container_width=True)
            shown = True
    if not shown:
        st.info("No error exports yet. Run `python -m src.ml.error_analysis`.")


def render_drift_tab() -> None:
    st.header("Offline distribution-shift analysis")
    _show_markdown_report(
        ML_REPORT_DIR / "drift_report.md",
        "No drift report yet. Run `python -m src.ml.drift`.",
    )


def render_events_tab() -> None:
    st.header("Event-relative sentiment")
    events_signals_path = Path(os.getenv("EVENTS_SIGNALS_PATH", "data/processed/signals_events.csv"))
    deltas_path = Path(os.getenv("EVENT_DELTAS_PATH", "reports/tables/event_sentiment_deltas.csv"))

    missing = [path for path in [events_signals_path, deltas_path] if not path.exists()]
    if missing:
        st.info(
            "No event-relative outputs yet. Run the full pipeline "
            "(`python scripts/run_pipeline.py`) or `python -m src.events` to "
            "generate:\n" + "\n".join(f"- `{path}`" for path in missing)
        )
        return

    deltas = load_csv(deltas_path)
    st.subheader("Sentiment by event window")
    st.caption(
        "Windows are defined by the nearest prior event in "
        "`data/reference/events.csv` (e.g. pre-announcement vs post-D23); "
        "`delta_vs_prev_window` shows the mean VADER compound shift between "
        "consecutive windows."
    )
    st.dataframe(deltas, use_container_width=True)

    if "mean_vader_compound" in deltas.columns:
        st.bar_chart(deltas.set_index("event_window")["mean_vader_compound"])

    signals = load_csv(events_signals_path)
    required = {"event_window", "days_since_event", "vader_compound"}
    missing_columns = required - set(signals.columns)
    if missing_columns:
        st.warning(
            f"Missing columns in `{events_signals_path}`: `{sorted(missing_columns)}`. "
            "Regenerate with `python -m src.events`."
        )
        return

    st.subheader("Sentiment vs days since event")
    windows = sorted_unique_values(signals, "event_window")
    selected = st.multiselect("Event windows", options=windows, default=windows)
    subset = signals[signals["event_window"].astype(str).isin(selected)]
    subset = subset.dropna(subset=["days_since_event"])
    if subset.empty:
        st.info("No rows with event-relative timestamps for the selected windows.")
        return
    st.scatter_chart(subset, x="days_since_event", y="vader_compound", color="event_window")


def render_live_inference_tab() -> None:
    st.header("Live inference")
    st.caption(f"Model status: {model_status()}")
    st.warning(
        "Development model trained on a limited manually labelled portfolio "
        "dataset. Outputs are illustrative, not production predictions."
    )
    text = st.text_area(
        "Enter a comment to classify",
        value="Still waiting for news, Square please give us something",
    )
    if not st.button("Classify"):
        return
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    from src.ml.inference import predict_text
    from src.score import INTENT_WEIGHTS, SENTIMENT_WEIGHTS
    from src.sentiment import label_from_compound

    result = predict_text(text)
    compound = SentimentIntensityAnalyzer().polarity_scores(str(text))["compound"]
    sentiment_label = label_from_compound(compound)
    intent_label = result["intent_label"]
    demand_score = INTENT_WEIGHTS.get(intent_label, 0.0) + SENTIMENT_WEIGHTS.get(
        sentiment_label, 0.0
    )

    columns = st.columns(3)
    if "actionable_probability" in result:
        columns[0].metric("Actionable probability", f"{result['actionable_probability']:.2f}")
    elif "rule_confidence_indicator" in result:
        columns[0].metric("Rule confidence indicator", f"{result['rule_confidence_indicator']:.1f}")
    columns[1].metric("Predicted intent", intent_label)
    if "intent_probability" in result:
        columns[2].metric("Intent probability", f"{result['intent_probability']:.2f}")

    st.write(
        {
            "is_actionable": result.get("is_actionable"),
            "sentiment": sentiment_label,
            "demand_score_base": round(demand_score, 2),
            "activation_flag": intent_label
            in {"high_intent", "nostalgia_reactivation", "new_customer_interest"},
            "risk_flag": intent_label
            in {
                "frustrated_demand",
                "confusion_barrier",
                "content_drought_fatigue",
                "expectation_decay",
            },
            "model_name": result.get("model_name"),
        }
    )


# --- Forecasting, monitoring and marketing-science views -------------------
# These read artefacts generated by the experiment scripts. The dashboard never
# fits a Bayesian model at page-load time, so a lightweight deployment works
# without PyMC installed.


def _forecast_artefacts():
    """Load forecasting results, or None with guidance when absent."""
    from src.services import ArtefactMissingError, load_forecast_artefacts

    try:
        return load_forecast_artefacts()
    except ArtefactMissingError as exc:
        st.info(str(exc))
        return None


def _render_provenance_banner(metadata: dict) -> None:
    """Say plainly whether these results describe the full dataset or a sample."""
    if metadata.get("sample_only"):
        st.error(
            "**Demo smoke run — not a substantive result.** These figures come "
            "from the class-stratified portfolio extract, which is not "
            "volume-representative. They exist to show the pipeline runs. "
            "Measured results require a run over the full local dataset: "
            "`python scripts/build_temporal_dataset.py --input "
            "data/processed/signals_scored.csv && python scripts/run_forecasting.py`."
        )
    else:
        st.caption(
            f"Run `{metadata.get('run_id')}` · commit `{metadata.get('git_commit')}` · "
            f"evaluation status `{metadata.get('evaluation_status')}`"
        )


def render_forecasting_tab() -> None:
    st.header("Forecasting")
    st.caption(
        "Forecasts of observable behavioural demand proxies derived from public "
        "community discussion. These are not sales, revenue, or unit forecasts."
    )
    artefacts = _forecast_artefacts()
    if artefacts is None:
        return
    _render_provenance_banner(artefacts.metadata)

    targets = list(artefacts.metadata.get("targets", {}))
    if not targets:
        st.info("No targets were modelled in this run.")
        return
    target = st.selectbox("Target", options=targets, key="forecast_target")
    details = artefacts.metadata["targets"][target]

    if not details.get("usable"):
        st.warning(f"`{target}` was not modelled: {details.get('reason')}")
        return

    columns = st.columns(4)
    columns[0].metric("Selected model", str(details.get("selected_model")))
    columns[1].metric("Likelihood", str(details.get("likelihood")))
    columns[2].metric("Forecast origins", len(details.get("origins") or []))
    columns[3].metric("Non-empty periods", details.get("non_empty_periods"))

    st.markdown(
        "`actionable_probability_mass` is the sum of calibrated P(actionable) over "
        "the comments observed in a period: an expected actionable-signal volume, "
        "not expected purchases."
    )

    observed = artefacts.observed
    forecast = artefacts.forecast
    if not observed.empty and "target" in observed.columns:
        series = observed[observed["target"] == target].copy()
        if not series.empty:
            st.subheader("Observed series")
            if "period_start" in series.columns:
                series["period_start"] = pd.to_datetime(series["period_start"], utc=True)
                st.line_chart(series.set_index("period_start")["actual"])
            else:
                st.line_chart(series["actual"])

    if not forecast.empty and "target" in forecast.columns:
        rows = forecast[forecast["target"] == target]
        if not rows.empty:
            st.subheader("Forecast with interval")
            st.dataframe(
                rows[["period_start", "horizon_step", "forecast", "lower", "upper"]],
                use_container_width=True,
            )
            chart_frame = rows.set_index("horizon_step")[["forecast", "lower", "upper"]]
            st.line_chart(chart_frame)

    if not artefacts.comparison.empty:
        st.subheader("Model comparison (rolling-origin backtest)")
        st.caption(
            "Selection uses WAPE first, then mean absolute error, signed bias and "
            "interval coverage. A model that wins on WAPE but forecasts "
            "persistently in one direction is demoted."
        )
        subset = artefacts.comparison
        if "target" in subset.columns:
            subset = subset[subset["target"] == target]
        st.dataframe(subset, use_container_width=True)

    for reason in details.get("selection_reasoning") or []:
        st.markdown(f"- {reason}")

    diagnostics = details.get("bayesian_diagnostics") or {}
    if diagnostics:
        st.subheader("Bayesian diagnostics")
        columns = st.columns(4)
        columns[0].metric("Max R-hat", f"{diagnostics.get('max_r_hat', float('nan')):.4f}")
        columns[1].metric("Min bulk ESS", f"{diagnostics.get('min_ess_bulk', float('nan')):.0f}")
        columns[2].metric("Divergences", diagnostics.get("divergences"))
        columns[3].metric("Converged", "yes" if diagnostics.get("converged") else "no")
        for warning in diagnostics.get("warnings", []):
            st.warning(warning)


def render_forecast_monitoring_tab() -> None:
    st.header("Forecast monitoring")
    st.caption(
        "A forecast that is consistently wrong in the same direction costs more "
        "in planning than one with a slightly larger but unbiased error."
    )
    artefacts = _forecast_artefacts()
    if artefacts is None:
        return
    _render_provenance_banner(artefacts.metadata)

    monitoring = artefacts.monitoring
    if monitoring.empty:
        st.info("No monitoring table yet. Run `python scripts/run_forecasting.py`.")
        return

    targets = sorted(monitoring["target"].unique()) if "target" in monitoring.columns else []
    target = st.selectbox("Target", options=targets, key="monitoring_target") if targets else None
    subset = monitoring[monitoring["target"] == target] if target else monitoring

    models = sorted(subset["model"].unique())
    model = st.selectbox("Model", options=models, key="monitoring_model")
    rows = subset[subset["model"] == model].sort_values(["origin", "horizon_step"])

    st.subheader("Forecast versus actual")
    st.line_chart(rows.set_index("origin")[["actual", "forecast"]])

    st.subheader("Rolling error and signed bias")
    st.line_chart(rows.set_index("origin")[["rolling_mae", "rolling_bias"]])

    flags = artefacts.bias_flags.get(target, {}).get("models", {}).get(model, {})
    if flags:
        persistent = flags.get("persistent_bias")
        sufficient = flags.get("sufficient_evidence", True)
        columns = st.columns(4)
        columns[0].metric("Mean signed error", f"{flags.get('mean_signed_error', float('nan')):.3f}")
        coverage = flags.get("interval_coverage")
        columns[1].metric(
            "Interval coverage", "—" if coverage is None else f"{coverage:.0%}"
        )
        # "Unknown" is a third state, not a quiet "no": too few forecasts to judge.
        columns[2].metric(
            "Persistent bias",
            "unknown" if persistent is None else ("yes" if persistent else "no"),
        )
        columns[3].metric("Forecasts", flags.get("n_forecasts", "—"))
        st.caption(f"Interval calibration: {flags.get('interval_calibration')}")

        if not sufficient:
            st.warning(
                f"Only {flags.get('n_forecasts')} forecasts for this model — too few "
                "to call bias or interval calibration either way. A run of four "
                "same-signed errors in six happens about one time in five by "
                "chance, so the figures above are observations, not verdicts."
            )
            for run in flags.get("observed_bias_runs", []):
                st.caption(
                    f"Observed (not concluded): {run['direction'].replace('_', ' ')} "
                    f"over {run['length']} consecutive periods."
                )
        elif persistent:
            for run in flags.get("bias_runs", []):
                st.error(
                    f"Persistent {run['direction'].replace('_', ' ')} over "
                    f"{run['length']} consecutive periods "
                    f"(mean signed error {run['mean_signed_error']:.3f})."
                )
        else:
            st.success("No persistent directional bias detected at the configured thresholds.")

    with st.expander("Monitoring table", expanded=False):
        st.dataframe(rows, use_container_width=True)


def render_marketing_lab_tab() -> None:
    st.header("Marketing Science Lab")
    st.error(
        "**SYNTHETIC DATA.** Everything on this tab is simulated by "
        "`src/mmm/synthetic.py`. **This is not Square Enix data.** No figure here "
        "describes real media spend, pricing, or commercial performance, and none "
        "of it comes from the Kingdom Hearts IV community signals analysed on the "
        "other tabs."
    )
    from src.services import ArtefactMissingError, load_mmm_artefacts

    try:
        artefacts = load_mmm_artefacts()
    except ArtefactMissingError as exc:
        st.info(str(exc))
        return

    metadata = artefacts["metadata"]
    st.caption(
        f"Run `{metadata.get('run_id')}` · generator seed `{metadata.get('generator_seed')}` · "
        f"recovery checks passed {metadata.get('recovery_checks_passed')}/"
        f"{metadata.get('recovery_checks_total')}"
    )
    st.markdown(
        "Kingdom Hearts IV is unreleased, so no spend, price or sales data exists "
        "for it. Simulating the data from a known process makes the marketing-science "
        "methods demonstrable **and** checkable: the model is scored on whether it "
        "recovers the parameters that generated the data."
    )

    contributions = artefacts["contributions"]
    if not contributions.empty:
        st.subheader("Posterior channel contribution (simulated sales)")
        st.dataframe(contributions, use_container_width=True)
        st.bar_chart(contributions.set_index("channel")["contribution_mean"])
        st.caption(
            "Credible intervals on contribution are wide and overlapping. Point "
            "estimates should never be quoted without them."
        )

    curves = artefacts["response_curves"]
    if not curves.empty:
        st.subheader("Media response curves (adstock then saturation)")
        pivot = curves.pivot_table(index="spend", columns="channel", values="response_mean")
        st.line_chart(pivot)
        st.caption(
            "Each curve rises and flattens: the marginal return falls as spend "
            "grows, which is why moving budget between channels does not move "
            "simulated sales proportionally."
        )

    recovery = artefacts["recovery"]
    if not recovery.empty:
        st.subheader("Recovery against known ground truth")
        st.dataframe(recovery, use_container_width=True)

    diagnostics = metadata.get("diagnostics_summary", {})
    if diagnostics:
        st.subheader("Diagnostics")
        columns = st.columns(4)
        columns[0].metric("Max R-hat", f"{diagnostics.get('max_r_hat', float('nan')):.4f}")
        columns[1].metric("Min bulk ESS", f"{diagnostics.get('min_ess_bulk', float('nan')):.0f}")
        columns[2].metric("Divergences", diagnostics.get("divergences"))
        columns[3].metric("Converged", "yes" if diagnostics.get("converged") else "no")
        for warning in diagnostics.get("warnings", []):
            st.warning(warning)

    for note in metadata.get("identifiability_notes", []):
        st.markdown(f"- {note}")


def render_scenario_planner_tab() -> None:
    st.header("Scenario planner")
    st.caption(
        "Two separate modes. The behavioural mode asks what observable community "
        "demand would look like under a hypothetical; the marketing mode asks a "
        "budget question of a wholly synthetic market. They are never combined."
    )
    from src.services import ArtefactMissingError, behavioural_scenario, mmm_scenario

    mode = st.radio(
        "Scenario mode",
        options=["REAL BEHAVIOURAL SCENARIO", "SYNTHETIC MARKETING SCENARIO"],
        key="scenario_mode",
    )

    if mode == "REAL BEHAVIOURAL SCENARIO":
        st.info(
            "Real public Kingdom Hearts IV discussion signals. The output is a "
            "change in expected actionable-signal volume, not sales."
        )
        target = st.text_input("Target", value="actionable_probability_mass")
        level_shift = st.slider("Sustained change in actionable discussion", 0.5, 2.0, 1.2, 0.05)
        event_pulse = st.slider("Event pulse multiplier", 1.0, 4.0, 1.0, 0.1)
        pulse_periods = st.slider("Pulse duration (weeks)", 0, 8, 1)
        if st.button("Run behavioural scenario"):
            try:
                result = behavioural_scenario(
                    level_shift=level_shift,
                    event_pulse=event_pulse,
                    pulse_periods=pulse_periods,
                    target=target,
                    label="dashboard_scenario",
                )
            except ArtefactMissingError as exc:
                st.warning(str(exc))
                return
            if result.get("sample_only"):
                st.error(
                    "These posterior samples come from the demo smoke run over the "
                    "stratified sample and are not a substantive result."
                )
            columns = st.columns(3)
            columns[0].metric("Baseline total", f"{result['total_baseline']:.2f}")
            columns[1].metric("Scenario total", f"{result['total_scenario']:.2f}")
            columns[2].metric("Difference", f"{result['total_difference_mean']:+.2f}")
            st.caption(
                f"90% interval on the difference: {result['total_difference_lower']:.2f} "
                f"to {result['total_difference_upper']:.2f}"
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "horizon_step": range(1, result["horizon"] + 1),
                        "baseline": result["baseline_mean"],
                        "scenario": result["scenario_mean"],
                        "difference": result["difference_mean"],
                    }
                ),
                use_container_width=True,
            )
            for warning in result.get("warnings", []):
                st.warning(warning)
            st.caption(result["interpretation"])
        return

    st.error(
        "**SYNTHETIC DATA.** This mode simulates a marketing plan in a simulated "
        "market. **This is not Square Enix data** and it is not a forecast of real "
        "commercial performance."
    )
    from_channel = st.selectbox(
        "Move budget from", ["social_spend", "paid_search_spend", "video_spend"]
    )
    to_channel = st.selectbox(
        "Move budget to", ["video_spend", "paid_search_spend", "social_spend"]
    )
    fraction = st.slider("Share of budget to move", 0.05, 0.5, 0.20, 0.05)
    price_change = st.slider("Price change", -10.0, 10.0, 0.0, 0.5)
    promotion = st.selectbox("Promotion", ["unchanged", "on", "off"])

    if st.button("Run marketing scenario"):
        if from_channel == to_channel:
            st.warning("Choose two different channels to reallocate between.")
            return
        try:
            result = mmm_scenario(
                budget_changes={from_channel: 1.0 - fraction, to_channel: 1.0 + fraction},
                price_change=price_change,
                promotion={"unchanged": None, "on": True, "off": False}[promotion],
                label=f"move_{int(fraction * 100)}pct_{from_channel}_to_{to_channel}",
            )
        except ArtefactMissingError as exc:
            st.warning(str(exc))
            return

        columns = st.columns(3)
        columns[0].metric("Baseline simulated sales", f"{result['baseline_expected_sales']:,.0f}")
        columns[1].metric("Scenario simulated sales", f"{result['scenario_expected_sales']:,.0f}")
        columns[2].metric("Difference", f"{result['difference_mean']:+,.0f}")
        st.caption(
            f"{int(result['interval_level'] * 100)}% interval: "
            f"{result['difference_lower']:,.0f} to {result['difference_upper']:,.0f} "
            f"· probability of an increase {result['probability_of_increase']:.0%}"
        )
        st.subheader("Channel contribution change")
        st.bar_chart(pd.Series(result["channel_contribution_change"]))
        for warning in result.get("warnings", []):
            st.warning(warning)
        st.caption(result["interpretation"])


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="KH4 Demand Intelligence",
        page_icon="👑",
        layout="wide",
    )
    render_header()
    st.caption(f"Model status: {model_status()}")

    tabs = st.tabs(
        [
            "Demand Intelligence",
            "Model Performance",
            "Model Comparison",
            "Error Analysis",
            "Drift",
            "Events",
            "Forecasting",
            "Forecast Monitoring",
            "Marketing Science Lab",
            "Scenario Planner",
            "Live Inference",
        ]
    )

    with tabs[0]:
        if require_files():
            signals_df = load_csv(SIGNALS_PATH)
            video_scores_df = load_csv(VIDEO_SCORES_PATH)
            if validate_columns(signals_df, video_scores_df):
                render_demand_tab(signals_df, video_scores_df)
    with tabs[1]:
        render_model_performance_tab()
    with tabs[2]:
        render_model_comparison_tab()
    with tabs[3]:
        render_error_analysis_tab()
    with tabs[4]:
        render_drift_tab()
    with tabs[5]:
        render_events_tab()
    with tabs[6]:
        render_forecasting_tab()
    with tabs[7]:
        render_forecast_monitoring_tab()
    with tabs[8]:
        render_marketing_lab_tab()
    with tabs[9]:
        render_scenario_planner_tab()
    with tabs[10]:
        render_live_inference_tab()

    st.caption(
        "Unofficial portfolio project. Not affiliated with Square Enix, Disney, "
        "or the Kingdom Hearts franchise. Data sourced from public community discussion."
    )


if __name__ == "__main__":
    main()

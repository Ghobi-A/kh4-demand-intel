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
        render_live_inference_tab()

    st.caption(
        "Unofficial portfolio project. Not affiliated with Square Enix, Disney, "
        "or the Kingdom Hearts franchise. Data sourced from public community discussion."
    )


if __name__ == "__main__":
    main()

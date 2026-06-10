"""Streamlit dashboard for Kingdom Hearts IV demand intelligence."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_SIGNALS_PATH = "data/demo/signals_scored_demo.csv"
DEFAULT_VIDEO_SCORES_PATH = "data/demo/video_scores_demo.csv"
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
        "By default the app loads `data/demo/signals_scored_demo.csv` and "
        "`data/demo/video_scores_demo.csv`. To use full local data, set "
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


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="KH4 Demand Intelligence",
        page_icon="👑",
        layout="wide",
    )
    render_header()

    if not require_files():
        return

    signals_df = load_csv(SIGNALS_PATH)
    video_scores_df = load_csv(VIDEO_SCORES_PATH)

    if not validate_columns(signals_df, video_scores_df):
        return

    filters = render_sidebar_filters(signals_df)
    filtered_df = filter_dataframe(signals_df, filters)

    if filtered_df.empty:
        st.warning("No rows match the selected filters. Clear filters to see results.")
        return

    render_kpis(filtered_df)
    render_charts(filtered_df, video_scores_df)
    render_tables(filtered_df, video_scores_df)
    render_explainability()


if __name__ == "__main__":
    main()

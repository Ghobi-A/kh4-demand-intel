"""Event-relative sentiment analysis for KH4 signals.

Annotates scored signals with the nearest prior announcement event
(from ``data/reference/events.csv``) and aggregates sentiment deltas
between event windows (e.g. pre-announcement vs post-D23).
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_EVENTS_PATH = Path("data/reference/events.csv")
DEFAULT_INPUT_PATH = Path("data/processed/signals_scored.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/signals_events.csv")
DEFAULT_DELTA_OUTPUT_PATH = Path("reports/tables/event_sentiment_deltas.csv")
DEFAULT_FIGURE_PATH = Path("reports/figures/sentiment_by_event.png")

PRE_EVENT_LABEL = "pre_announcement"

REQUIRED_EVENT_COLUMNS = {"event_date", "event_name"}


def _slugify(name: str) -> str:
    """Return a lowercase snake_case slug for an event name."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return slug or "event"


def load_events(path: Path = DEFAULT_EVENTS_PATH) -> pd.DataFrame:
    """Load and validate the event reference table, sorted by date."""
    events = pd.read_csv(path)

    missing = REQUIRED_EVENT_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"Expected columns {sorted(REQUIRED_EVENT_COLUMNS)}; missing {sorted(missing)}")

    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce", utc=True)
    unparseable = events["event_date"].isna()
    if unparseable.any():
        LOGGER.warning("Dropping %s event rows with unparseable event_date", int(unparseable.sum()))
        events = events[~unparseable]

    if events.empty:
        raise ValueError(f"No valid events found in {path}")

    events = events.sort_values("event_date").reset_index(drop=True)
    events["event_window"] = "post_" + events["event_name"].map(_slugify)
    return events


def annotate_events_dataframe(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with nearest-prior-event columns added."""
    if "timestamp" not in df.columns:
        raise ValueError("Expected column 'timestamp' in input data")

    annotated = df.copy()
    annotated["timestamp"] = pd.to_datetime(annotated["timestamp"], errors="coerce", utc=True)

    events = events.sort_values("event_date")
    event_lookup = events[["event_date", "event_name", "event_window"]].rename(
        columns={"event_name": "last_event_name"}
    )

    order_column = "_events_original_order"
    annotated[order_column] = range(len(annotated))
    with_ts = annotated[annotated["timestamp"].notna()].sort_values("timestamp")
    without_ts = annotated[annotated["timestamp"].isna()]

    merged = pd.merge_asof(
        with_ts,
        event_lookup,
        left_on="timestamp",
        right_on="event_date",
        direction="backward",
    )

    result = pd.concat([merged, without_ts], ignore_index=True).sort_values(order_column)
    result = result.drop(columns=[order_column]).reset_index(drop=True)

    result["days_since_event"] = (result["timestamp"] - result["event_date"]).dt.total_seconds() / 86400.0
    result["last_event_name"] = result["last_event_name"].fillna(PRE_EVENT_LABEL)
    result["event_window"] = result["event_window"].fillna(PRE_EVENT_LABEL)
    result = result.drop(columns=["event_date"])

    return result


def aggregate_event_deltas(annotated: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment metrics per event window with deltas vs the previous window."""
    required = {"event_window", "sentiment_label"}
    missing = required - set(annotated.columns)
    if missing:
        raise ValueError(f"Expected columns {sorted(required)}; missing {sorted(missing)}")

    aggregations: dict[str, tuple[str, object]] = {
        "n_comments": ("event_window", "size"),
        "positive_share": ("sentiment_label", lambda s: (s == "positive").mean()),
        "negative_share": ("sentiment_label", lambda s: (s == "negative").mean()),
    }
    if "vader_compound" in annotated.columns:
        aggregations["mean_vader_compound"] = ("vader_compound", "mean")
    if "demand_score" in annotated.columns:
        aggregations["mean_demand_score"] = ("demand_score", "mean")

    summary = annotated.groupby("event_window", dropna=False).agg(**aggregations)

    # Order windows chronologically: pre_announcement first, then by first comment timestamp.
    if "timestamp" in annotated.columns:
        first_seen = pd.to_datetime(annotated["timestamp"], errors="coerce", utc=True).groupby(
            annotated["event_window"]
        ).min()
        order = first_seen.sort_values().index.tolist()
    else:
        order = summary.index.tolist()
    if PRE_EVENT_LABEL in order:
        order = [PRE_EVENT_LABEL] + [w for w in order if w != PRE_EVENT_LABEL]

    summary = summary.reindex(order).reset_index()
    if "mean_vader_compound" in summary.columns:
        summary["delta_vs_prev_window"] = summary["mean_vader_compound"].diff()
    return summary


def plot_event_timeline(
    annotated: pd.DataFrame,
    events: pd.DataFrame,
    output_path: Path = DEFAULT_FIGURE_PATH,
) -> Path | None:
    """Plot daily mean vader_compound with event markers; returns path or None if skipped."""
    if "vader_compound" not in annotated.columns:
        LOGGER.warning("Skipping event timeline figure: 'vader_compound' column missing")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("Skipping event timeline figure: matplotlib is not installed")
        return None

    df = annotated.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    daily = df.set_index("timestamp")["vader_compound"].resample("D").mean().dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily.index, daily.values, marker="o", linewidth=1.5)
    for _, event in events.iterrows():
        ax.axvline(event["event_date"], linestyle="--", alpha=0.6, color="tab:red")
        ax.text(
            event["event_date"],
            ax.get_ylim()[1],
            str(event["event_name"]),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )
    ax.set_title("Daily mean VADER compound sentiment with announcement events")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean vader_compound")
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Wrote event timeline figure to %s", output_path)
    return output_path


def run_events_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    events_path: Path = DEFAULT_EVENTS_PATH,
    delta_output_path: Path = DEFAULT_DELTA_OUTPUT_PATH,
    figure_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate scored signals with event context and save window-level deltas."""
    df = pd.read_csv(input_path)
    events = load_events(events_path)

    annotated = annotate_events_dataframe(df, events)
    deltas = aggregate_event_deltas(annotated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    delta_output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated.to_csv(output_path, index=False)
    deltas.to_csv(delta_output_path, index=False)

    LOGGER.info("Rows annotated: %s", len(annotated))
    LOGGER.info("Event windows: %s", deltas["event_window"].tolist())
    LOGGER.info("Wrote annotated signals to %s", output_path)
    LOGGER.info("Wrote event sentiment deltas to %s", delta_output_path)

    if figure_path is not None:
        plot_event_timeline(annotated, events, output_path=figure_path)

    return annotated, deltas


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate KH4 signals with event-relative context")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--delta-output", type=Path, default=DEFAULT_DELTA_OUTPUT_PATH)
    parser.add_argument(
        "--figure",
        action="store_true",
        help=f"Also write a sentiment timeline figure to {DEFAULT_FIGURE_PATH}",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_parser().parse_args()
    run_events_pipeline(
        input_path=args.input,
        output_path=args.output,
        events_path=args.events,
        delta_output_path=args.delta_output,
        figure_path=DEFAULT_FIGURE_PATH if args.figure else None,
    )


if __name__ == "__main__":
    main()

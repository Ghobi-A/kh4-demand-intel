"""Weekly temporal aggregation of behavioural signals.

Turns row-level scored signals into a period-level table that the forecasting
layer can model. The headline column is the **behavioural demand proxy**:

    actionable_probability_mass_t = sum over comments i observed in period t
                                    of P(actionable_i)

This is an expected actionable-signal volume. It measures how much
actionable-looking public discussion occurred, not how many people bought
anything: Kingdom Hearts IV is unreleased and no purchase data exists.

Rows without a usable creation timestamp are excluded from every period and
counted in the metadata; they are never assigned to a bucket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from src.events import (
    DEFAULT_EVENT_HALF_LIFE_WEEKS,
    DEFAULT_EVENT_WINDOW_WEEKS,
    DEFAULT_EVENTS_PATH,
    event_features_for_periods,
    load_events,
)
from src.ml.probabilities import (
    PROBABILITY_COLUMN,
    PROVENANCE_COLUMN,
    SOURCE_UNAVAILABLE,
    ProbabilityUnavailableError,
    build_probabilities,
)
from src.timestamps import (
    audit_temporal_coverage,
    complexity_tier,
    parse_timestamps,
    period_range,
    period_start,
    serialise_timestamps,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_FREQ = "W-MON"
DEFAULT_INPUT_PATH = Path("data/processed/signals_scored.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/weekly_demand_signals.csv")
DEMO_INPUT_PATH = Path("data/demo/signals_scored_portfolio.csv")
DEMO_OUTPUT_PATH = Path("data/demo/weekly_demand_signals_demo.csv")

ACTIONABLE_INTENTS = {
    "high_intent",
    "nostalgia_reactivation",
    "new_customer_interest",
    "frustrated_demand",
    "content_drought_fatigue",
    "confusion_barrier",
    "expectation_decay",
}

EVALUATION_STATUS_DEMO = "demo_only"
EVALUATION_STATUS_FULL = "full_dataset"
EVALUATION_STATUS_SEED_CAPPED = "seed_capped_dataset"

DEMO_BANNER = (
    "Generated from the class-stratified 304-row portfolio extract. This is a "
    "pipeline smoke output only: the sample is not volume-representative, so "
    "its weekly volumes and any metric derived from them are not substantive "
    "results."
)


def _rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Share, with NaN where the denominator is zero.

    A week with no comments carries no information about a rate; recording 0%
    there would invent an observation of "no actionable demand".
    """
    denominator = denominator.astype(float)
    return numerator.astype(float).div(denominator).where(denominator > 0)


def dataframe_hash(df: pd.DataFrame) -> str:
    """Deterministic content hash for provenance metadata."""
    payload = pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _youtube_collection_meta(
    manifest_path: Path = Path("reports/tables/youtube_fetch_manifest.csv"),
) -> dict:
    """Summarise YouTube collection provenance when a fetch manifest exists.

    A seed-list scrape can be a real observed dataset without being volume
    representative. If any fetched video hits the requested ``max_results``
    cap, the weekly comment volumes are right-censored by collection design.
    """
    if not manifest_path.exists():
        return {"manifest_path": str(manifest_path), "manifest_found": False}

    manifest = pd.read_csv(manifest_path)
    meta = {
        "manifest_path": str(manifest_path),
        "manifest_found": True,
        "videos": int(len(manifest)),
        "rows_fetched": int(manifest.get("rows_fetched", pd.Series(dtype=int)).fillna(0).sum()),
        "failed_videos": int((manifest.get("status", pd.Series(dtype=str)) == "failed").sum()),
    }
    if {"rows_fetched", "max_results"}.issubset(manifest.columns):
        fetched = pd.to_numeric(manifest["rows_fetched"], errors="coerce").fillna(0)
        caps = pd.to_numeric(manifest["max_results"], errors="coerce")
        capped = caps.notna() & (fetched >= caps)
        meta["max_results_per_video"] = int(caps.dropna().max()) if caps.notna().any() else None
        meta["cap_hit_videos"] = int(capped.sum())
        meta["seed_capped"] = bool(capped.any())
    else:
        meta["seed_capped"] = None

    return meta


def _should_use_youtube_manifest(input_path: Path) -> bool:
    """Avoid letting a repo-local manifest contaminate external/tmp datasets."""
    try:
        input_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return False
    return True


def aggregate_weekly(
    df: pd.DataFrame,
    freq: str = DEFAULT_FREQ,
    events: pd.DataFrame | None = None,
    window_weeks: int = DEFAULT_EVENT_WINDOW_WEEKS,
    half_life_weeks: float = DEFAULT_EVENT_HALF_LIFE_WEEKS,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate row-level signals into a complete, gap-free period table.

    Only columns supported by the input are emitted. Empty periods are kept
    with zero counts so the series has no implicit holes, but their rates stay
    NaN rather than being read as observed zeros.
    """
    parsed, report = parse_timestamps(df["timestamp"]) if "timestamp" in df.columns else (
        pd.Series(dtype="datetime64[ns, UTC]"),
        None,
    )
    frame = df.reset_index(drop=True).copy()
    parsed = parsed.reset_index(drop=True)
    usable = frame[parsed.notna()].copy()
    dropped = int(parsed.isna().sum())
    if dropped:
        LOGGER.warning("Excluding %s rows with no usable timestamp from aggregation", dropped)

    meta: dict = {
        "rows_in": int(len(frame)),
        "rows_aggregated": int(len(usable)),
        "rows_without_timestamp": dropped,
        "freq": freq,
        "timestamp_report": report.to_dict() if report is not None else None,
    }

    if usable.empty:
        return pd.DataFrame(columns=["period_start", "total_comments"]), meta

    # Align on index rather than assigning ``.values``: a raw ndarray drops the
    # UTC timezone, and the naive result then matches nothing on reindex.
    starts = period_start(parsed[parsed.notna()], freq)
    starts.index = usable.index
    usable["period_start"] = starts
    index = period_range(usable["period_start"].min(), usable["period_start"].max(), freq)
    grouped = usable.groupby("period_start", sort=True)

    out = pd.DataFrame({"period_start": index})
    out = out.set_index("period_start")
    out["total_comments"] = grouped.size().reindex(index, fill_value=0).astype(int)

    if "source" in usable.columns:
        source = usable["source"].astype(str).str.lower()
        for platform in ["youtube", "reddit"]:
            counts = (
                usable[source.str.contains(platform, na=False)]
                .groupby("period_start")
                .size()
                .reindex(index, fill_value=0)
            )
            out[f"{platform}_comments"] = counts.astype(int)

    if "intent_label" in usable.columns:
        intent = usable["intent_label"].astype(str)
        actionable = usable[intent.isin(ACTIONABLE_INTENTS)]
        out["actionable_count"] = (
            actionable.groupby("period_start").size().reindex(index, fill_value=0).astype(int)
        )
        out["actionable_rate"] = _rate(out["actionable_count"], out["total_comments"])
        for label in ["high_intent", "frustrated_demand", "nostalgia_reactivation"]:
            subset = usable[intent == label]
            counts = subset.groupby("period_start").size().reindex(index, fill_value=0).astype(int)
            out[f"{label}_count"] = counts
            out[f"{label}_rate"] = _rate(counts, out["total_comments"])

    if PROBABILITY_COLUMN in usable.columns:
        provenance = (
            usable[PROVENANCE_COLUMN]
            if PROVENANCE_COLUMN in usable.columns
            else pd.Series("", index=usable.index)
        )
        scored = usable[usable[PROBABILITY_COLUMN].notna() & (provenance != SOURCE_UNAVAILABLE)]
        by_period = scored.groupby("period_start")[PROBABILITY_COLUMN]
        out["actionable_probability_mass"] = (
            by_period.sum().reindex(index, fill_value=0.0).astype(float)
        )
        out["mean_actionable_probability"] = by_period.mean().reindex(index)
        out["n_with_probability"] = (
            by_period.size().reindex(index, fill_value=0).astype(int)
        )
        out["probability_coverage"] = _rate(out["n_with_probability"], out["total_comments"])

    if "vader_compound" in usable.columns:
        out["mean_sentiment"] = grouped["vader_compound"].mean().reindex(index)
    if "demand_score" in usable.columns:
        out["mean_demand_score"] = grouped["demand_score"].mean().reindex(index)

    out = out.reset_index()

    if events is not None and not events.empty:
        features = event_features_for_periods(
            out["period_start"],
            events,
            window_weeks=window_weeks,
            half_life_weeks=half_life_weeks,
        )
        # No origin is passed here: the descriptive table uses the full
        # retrospective calendar, which is why it is labelled as such.
        features["event_annotation"] = "retrospective"
        out = out.merge(features, on="period_start", how="left")

    meta["periods"] = int(len(out))
    meta["non_empty_periods"] = int((out["total_comments"] > 0).sum())
    meta["complexity_tier"] = complexity_tier(meta["non_empty_periods"])
    meta["period_start_min"] = out["period_start"].min().isoformat()
    meta["period_start_max"] = out["period_start"].max().isoformat()
    return out, meta


def build_temporal_dataset(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path | None = None,
    events_path: Path = DEFAULT_EVENTS_PATH,
    freq: str = DEFAULT_FREQ,
    demo: bool = False,
    artifact_dir: Path | None = None,
    seed: int = 42,
    use_oof: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build the weekly demand-signal table and its provenance metadata.

    ``demo`` routes outputs to the demo location and stamps every artefact so a
    smoke run over the stratified sample can never be mistaken for a measured
    result over the canonical dataset.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = DEMO_OUTPUT_PATH if demo else DEFAULT_OUTPUT_PATH
    output_path = Path(output_path)

    signals = pd.read_csv(input_path)
    source_hash = dataframe_hash(signals)

    probability_meta: dict = {}
    primary_target_available = True
    try:
        kwargs = {"seed": seed, "use_oof": use_oof}
        if artifact_dir is not None:
            kwargs["artifact_dir"] = Path(artifact_dir)
        signals, probability_meta = build_probabilities(signals, **kwargs)
    except ProbabilityUnavailableError as exc:
        primary_target_available = False
        probability_meta = {"error": str(exc)}
        LOGGER.warning(
            "No calibrated probabilities available (%s); actionable_probability_mass "
            "will be omitted and count targets retained",
            exc,
        )

    events = load_events(events_path) if Path(events_path).exists() else None
    weekly, agg_meta = aggregate_weekly(signals, freq=freq, events=events)

    coverage = audit_temporal_coverage(signals, freq=freq, events=events)

    youtube_collection = (
        _youtube_collection_meta()
        if _should_use_youtube_manifest(input_path)
        else {"manifest_found": False}
    )
    seed_capped = youtube_collection.get("seed_capped") is True
    volume_representative = bool(not demo and not seed_capped)

    meta = {
        "source_dataset": str(input_path),
        "source_dataset_hash": source_hash,
        "output": str(output_path),
        "generated_freq": freq,
        "sample_only": bool(demo),
        "volume_representative": volume_representative,
        "evaluation_status": (
            EVALUATION_STATUS_DEMO
            if demo
            else EVALUATION_STATUS_SEED_CAPPED
            if seed_capped
            else EVALUATION_STATUS_FULL
        ),
        "youtube_collection": youtube_collection,
        "primary_target": "actionable_probability_mass",
        "primary_target_available": bool(
            primary_target_available and "actionable_probability_mass" in weekly.columns
        ),
        "primary_target_definition": (
            "Sum of calibrated P(actionable) over comments observed in the period. "
            "An expected actionable-signal volume (behavioural demand proxy), "
            "not expected purchases or sales."
        ),
        "probabilities": probability_meta,
        "aggregation": agg_meta,
        "coverage": coverage.to_dict(),
    }
    if demo:
        meta["notice"] = DEMO_BANNER

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writable = weekly.copy()
    writable["period_start"] = serialise_timestamps(writable["period_start"])
    writable.to_csv(output_path, index=False)
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    LOGGER.info("Wrote %s periods to %s", len(weekly), output_path)
    LOGGER.info("Wrote metadata to %s", meta_path)
    return weekly, meta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the weekly demand-signal table")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--freq", default=DEFAULT_FREQ)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Smoke-test mode: reads the stratified portfolio extract and writes "
            "demo-stamped outputs that are not substantive results"
        ),
    )
    parser.add_argument("--no-oof", action="store_true", help="Skip out-of-fold probabilities")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_parser().parse_args()
    input_path = args.input or (DEMO_INPUT_PATH if args.demo else DEFAULT_INPUT_PATH)
    weekly, meta = build_temporal_dataset(
        input_path=input_path,
        output_path=args.output,
        events_path=args.events,
        freq=args.freq,
        demo=args.demo,
        seed=args.seed,
        use_oof=not args.no_oof,
    )
    print(f"Periods: {len(weekly)} ({meta['aggregation']['non_empty_periods']} non-empty)")
    print(f"Evaluation status: {meta['evaluation_status']}")
    print(f"Primary target available: {meta['primary_target_available']}")
    print(f"Output: {meta['output']}")


if __name__ == "__main__":
    main()

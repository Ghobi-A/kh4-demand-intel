"""Recover missing creation timestamps for already-collected signals.

Rows collected before creation times were handled correctly may carry a
missing ``timestamp`` while still carrying the source identifier needed to
recover it deterministically. This module re-fetches those creation times.

It is a manual, network-dependent enrichment path. It is never invoked by the
normal pipeline or by CI, and it never invents a timestamp: an identifier that
the source cannot resolve is reported as ``unresolved`` and its row keeps its
missing timestamp.

    python -m src.enrich_timestamps --input data/processed/signals_clean.csv \
        --output data/processed/signals_clean.csv

Reddit rows are enriched only when ``--allow-pullpush`` is passed, because
PullPush is a third-party community archive rather than a first-party source.
Timestamps already present in the raw Reddit data are always preferred.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from dotenv import load_dotenv

from src.timestamps import parse_timestamps, serialise_timestamps, timestamp_status

LOGGER = logging.getLogger(__name__)

YOUTUBE_COMMENTS_ENDPOINT = "https://www.googleapis.com/youtube/v3/comments"
PULLPUSH_COMMENT_ENDPOINT = "https://api.pullpush.io/reddit/search/comment/"
REQUEST_TIMEOUT_SECONDS = 15
YOUTUBE_BATCH_SIZE = 50

DEFAULT_ENRICHMENT_PATH = Path("data/processed/timestamp_enrichment.csv")

STATUS_RESOLVED = "resolved"
STATUS_UNRESOLVED = "unresolved"
STATUS_SKIPPED = "skipped_source_not_enabled"


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_youtube_comment_times(comment_ids: list[str], api_key: str) -> dict[str, str]:
    """Resolve YouTube comment creation times via comments.list (50 per call)."""
    resolved: dict[str, str] = {}
    for batch in _chunked(comment_ids, YOUTUBE_BATCH_SIZE):
        params = {"part": "snippet", "id": ",".join(batch), "key": api_key}
        url = f"{YOUTUBE_COMMENTS_ENDPOINT}?{urlencode(params)}"
        try:
            with urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - one bad batch must not abort enrichment
            LOGGER.warning("YouTube batch failed (%s ids): %s", len(batch), exc)
            continue
        for item in payload.get("items", []):
            published = item.get("snippet", {}).get("publishedAt")
            if item.get("id") and published:
                resolved[str(item["id"])] = published
    return resolved


def fetch_reddit_comment_times(comment_ids: list[str]) -> dict[str, str]:
    """Resolve Reddit comment creation times via the PullPush archive."""
    resolved: dict[str, str] = {}
    for batch in _chunked(comment_ids, 100):
        url = f"{PULLPUSH_COMMENT_ENDPOINT}?{urlencode({'ids': ','.join(batch)})}"
        try:
            with urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - archive availability varies
            LOGGER.warning("PullPush batch failed (%s ids): %s", len(batch), exc)
            continue
        for item in payload.get("data", []):
            created = item.get("created_utc")
            if item.get("id") and created:
                resolved[str(item["id"])] = str(created)
    return resolved


def build_enrichment_table(
    df: pd.DataFrame,
    youtube_resolver=fetch_youtube_comment_times,
    reddit_resolver=fetch_reddit_comment_times,
    api_key: str | None = None,
    allow_pullpush: bool = False,
) -> pd.DataFrame:
    """Return one row per signal needing a timestamp, with what was recovered."""
    parsed, _ = parse_timestamps(df["timestamp"]) if "timestamp" in df.columns else (
        pd.Series(pd.NaT, index=range(len(df)), dtype="datetime64[ns, UTC]"),
        None,
    )
    parsed.index = df.index
    needs = df[parsed.isna()]
    if needs.empty:
        return pd.DataFrame(columns=["id", "source", "timestamp", "status"])

    entries: list[dict] = []
    youtube_ids = needs.loc[needs["source"].astype(str) == "youtube", "id"].astype(str).tolist()
    reddit_ids = needs.loc[needs["source"].astype(str) == "reddit", "id"].astype(str).tolist()

    youtube_resolved: dict[str, str] = {}
    if youtube_ids:
        if api_key:
            youtube_resolved = youtube_resolver(youtube_ids, api_key)
        else:
            LOGGER.warning("No YOUTUBE_API_KEY: %s YouTube rows left unresolved", len(youtube_ids))

    reddit_resolved: dict[str, str] = {}
    if reddit_ids and allow_pullpush:
        reddit_resolved = reddit_resolver(reddit_ids)
    elif reddit_ids:
        LOGGER.info(
            "%s Reddit rows skipped; pass --allow-pullpush to use the third-party archive",
            len(reddit_ids),
        )

    for record_id, source in zip(needs["id"].astype(str), needs["source"].astype(str)):
        if source == "youtube":
            raw = youtube_resolved.get(record_id)
            status = STATUS_RESOLVED if raw else STATUS_UNRESOLVED
        elif source == "reddit":
            if not allow_pullpush:
                raw, status = None, STATUS_SKIPPED
            else:
                raw = reddit_resolved.get(record_id)
                status = STATUS_RESOLVED if raw else STATUS_UNRESOLVED
        else:
            raw, status = None, STATUS_UNRESOLVED
        entries.append({"id": record_id, "source": source, "raw_timestamp": raw, "status": status})

    table = pd.DataFrame(entries)
    resolved_ts, _ = parse_timestamps(table["raw_timestamp"])
    table["timestamp"] = serialise_timestamps(resolved_ts)
    return table[["id", "source", "timestamp", "status"]]


def apply_enrichment(df: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    """Fill only genuinely missing timestamps; never overwrite an observed one."""
    if enrichment.empty:
        return df.copy()
    merged = df.copy()
    parsed, _ = parse_timestamps(merged["timestamp"])
    parsed.index = merged.index

    lookup_series, _ = parse_timestamps(enrichment["timestamp"])
    lookup = dict(zip(enrichment["id"].astype(str), lookup_series))

    recovered = merged["id"].astype(str).map(lookup)
    fill_mask = parsed.isna() & recovered.notna()
    parsed.loc[fill_mask] = recovered[fill_mask]

    merged["timestamp"] = parsed
    merged["timestamp_status"] = timestamp_status(
        parsed.reset_index(drop=True), merged["timestamp"].reset_index(drop=True)
    ).values
    LOGGER.info("Filled %s previously missing timestamps", int(fill_mask.sum()))
    return merged


def run_enrichment(
    input_path: Path,
    output_path: Path | None = None,
    enrichment_path: Path = DEFAULT_ENRICHMENT_PATH,
    allow_pullpush: bool = False,
) -> pd.DataFrame:
    """Recover missing timestamps for a signals CSV and write both outputs."""
    load_dotenv()
    df = pd.read_csv(input_path)
    table = build_enrichment_table(
        df, api_key=os.getenv("YOUTUBE_API_KEY"), allow_pullpush=allow_pullpush
    )
    enrichment_path = Path(enrichment_path)
    enrichment_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(enrichment_path, index=False)
    LOGGER.info("Wrote enrichment record to %s (%s rows)", enrichment_path, len(table))

    enriched = apply_enrichment(df, table)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writable = enriched.copy()
        writable["timestamp"] = serialise_timestamps(writable["timestamp"])
        writable.to_csv(output_path, index=False)
        LOGGER.info("Wrote enriched signals to %s", output_path)
    return enriched


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover missing signal creation timestamps")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--enrichment-record", type=Path, default=DEFAULT_ENRICHMENT_PATH)
    parser.add_argument(
        "--allow-pullpush",
        action="store_true",
        help="Enable Reddit recovery via the third-party PullPush archive (optional)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_parser().parse_args()
    run_enrichment(args.input, args.output, args.enrichment_record, args.allow_pullpush)


if __name__ == "__main__":
    main()

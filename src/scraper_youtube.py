"""YouTube comment ingestion for KH4 demand intelligence.

Fetches top-level comments via YouTube Data API v3 commentThreads.list,
maps records into the shared SignalRecord schema, and persists to CSV.

Usage:
    python -m src.scraper_youtube --video-id <YT_VIDEO_ID> [--max-results 100]
    python -m src.scraper_youtube --seed-list [PATH] [--max-results 100] [--force]

The batch mode reads a seed list CSV (``video_id`` column required) and fetches
comments for every listed video, writing a provenance manifest to
``reports/tables/youtube_fetch_manifest.csv``.

Requires YOUTUBE_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from dotenv import load_dotenv

from src.schema import SignalRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

YOUTUBE_COMMENTS_ENDPOINT = "https://www.googleapis.com/youtube/v3/commentThreads"
YOUTUBE_VIDEOS_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_OUTPUT_DIR = Path("data/raw/youtube")
YOUTUBE_REQUEST_TIMEOUT_SECONDS = 15

DEFAULT_SEED_LIST_PATH = YOUTUBE_OUTPUT_DIR / "video_seed_list.csv"
EXAMPLE_SEED_LIST_PATH = YOUTUBE_OUTPUT_DIR / "video_seed_list.example.csv"
DEFAULT_MANIFEST_PATH = Path("reports/tables/youtube_fetch_manifest.csv")
REQUIRED_SEED_COLUMNS = {"video_id"}

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_SKIPPED = "skipped_existing"
STATUS_FAILED = "failed"


def _parse_iso8601(timestamp: str) -> datetime:
    """Parse YouTube API timestamps into timezone-aware datetimes."""
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def _fetch_comment_threads_page(
    video_id: str,
    api_key: str,
    max_results: int,
    page_token: str | None = None,
) -> dict:
    """Fetch one page from YouTube commentThreads.list."""
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max_results,
        "textFormat": "plainText",
        "key": api_key,
    }
    if page_token:
        params["pageToken"] = page_token

    request_url = f"{YOUTUBE_COMMENTS_ENDPOINT}?{urlencode(params)}"
    with urlopen(request_url, timeout=YOUTUBE_REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _save_records(records: list[SignalRecord], out_path: Path) -> None:
    """Write records to CSV with JSON metadata for portability."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        payload = asdict(record)
        payload["timestamp"] = record.timestamp.isoformat()
        payload["metadata"] = json.dumps(record.metadata)
        rows.append(payload)

    df = pd.DataFrame(rows)
    df["scraped_at"] = datetime.now(timezone.utc).isoformat()
    df.to_csv(out_path, index=False)


def _build_output_path(video_id: str, output_dir: Path | None = None) -> Path:
    """Build output path, namespaced by video_id to avoid overwrite."""
    if output_dir is None:
        output_dir = YOUTUBE_OUTPUT_DIR
    return output_dir / f"youtube_comments_{video_id}.csv"


def _resolve_api_key() -> str:
    """Return the YouTube API key from the environment, or raise."""
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("Missing YOUTUBE_API_KEY in .env")
    return api_key


def fetch_youtube_comments(
    video_id: str,
    max_results: int = 100,
    output_dir: Path | str | None = None,
) -> list[SignalRecord]:
    """Fetch top-level YouTube comments and persist them as raw CSV."""
    api_key = _resolve_api_key()

    records: list[SignalRecord] = []
    next_page_token: str | None = None
    pages_fetched = 0

    while len(records) < max_results:
        page_size = min(100, max_results - len(records))
        payload = _fetch_comment_threads_page(
            video_id=video_id,
            api_key=api_key,
            max_results=page_size,
            page_token=next_page_token,
        )
        pages_fetched += 1

        for item in payload.get("items", []):
            comment = item.get("snippet", {}).get("topLevelComment", {})
            comment_snippet = comment.get("snippet", {})
            comment_id = comment.get("id")

            if not comment_id or "publishedAt" not in comment_snippet:
                continue

            records.append(
                SignalRecord(
                    source="youtube",
                    record_type="comment",
                    id=comment_id,
                    text=comment_snippet.get("textDisplay", ""),
                    author=comment_snippet.get("authorDisplayName"),
                    timestamp=_parse_iso8601(comment_snippet["publishedAt"]),
                    engagement=int(comment_snippet.get("likeCount", 0)),
                    permalink=(
                        f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}"
                    ),
                    parent_id=video_id,
                    metadata={
                        "video_id": video_id,
                        "author_display_name": comment_snippet.get(
                            "authorDisplayName"
                        ),
                        "reply_count": item.get("snippet", {}).get(
                            "totalReplyCount", 0
                        ),
                    },
                )
            )

            if len(records) >= max_results:
                break

        next_page_token = payload.get("nextPageToken")
        if not next_page_token:
            break

    log.info(
        f"Fetched {len(records)} comments across {pages_fetched} page(s) "
        f"from video {video_id}"
    )
    resolved_output_dir = Path(output_dir) if output_dir else None
    output_path = _build_output_path(video_id, resolved_output_dir)
    _save_records(records, output_path)
    log.info(f"Saved to {output_path}")
    return records


def load_video_seed_list(path: Path | str = DEFAULT_SEED_LIST_PATH) -> pd.DataFrame:
    """Load and validate a video seed list CSV.

    Only ``video_id`` is required. Any ``title``/``channel``/``notes`` columns are
    hand-written and unverified, so callers should treat them as seed metadata
    rather than as fact about the video.
    """
    seed_path = Path(path)
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Seed list not found at {seed_path}. Copy the committed template first: "
            f"cp {EXAMPLE_SEED_LIST_PATH} {DEFAULT_SEED_LIST_PATH}"
        )

    seeds = pd.read_csv(seed_path)
    missing = REQUIRED_SEED_COLUMNS - set(seeds.columns)
    if missing:
        raise ValueError(f"Seed list {seed_path} missing required columns {sorted(missing)}")

    seeds["video_id"] = seeds["video_id"].astype(str).str.strip()
    blank = seeds["video_id"].isin(["", "nan", "None"])
    if blank.any():
        log.warning("Dropping %s seed rows with a blank video_id", int(blank.sum()))
        seeds = seeds[~blank]

    duplicated = seeds["video_id"].duplicated()
    if duplicated.any():
        log.warning("Dropping %s duplicate video_id rows", int(duplicated.sum()))
        seeds = seeds[~duplicated]

    if seeds.empty:
        raise ValueError(f"Seed list {seed_path} contains no usable video_id rows")

    return seeds.reset_index(drop=True)


def _resolve_seed_list_path(path: Path | str | None) -> Path:
    """Resolve the seed list path, falling back to the committed example."""
    if path is not None:
        return Path(path)
    if DEFAULT_SEED_LIST_PATH.exists():
        return DEFAULT_SEED_LIST_PATH
    if EXAMPLE_SEED_LIST_PATH.exists():
        log.warning(
            "%s not found; falling back to the committed template %s",
            DEFAULT_SEED_LIST_PATH,
            EXAMPLE_SEED_LIST_PATH,
        )
        return EXAMPLE_SEED_LIST_PATH
    return DEFAULT_SEED_LIST_PATH


def _should_skip(video_id: str, output_dir: Path | str | None, force: bool) -> bool:
    """Return True when this video already has an output CSV and force is off."""
    if force:
        return False
    resolved_dir = Path(output_dir) if output_dir else None
    return _build_output_path(video_id, resolved_dir).exists()


def _fetch_video_metadata(video_id: str, api_key: str) -> dict:
    """Fetch the real title/channel via videos.list; return {} on any failure.

    commentThreads.list carries no video-level metadata, so this is the only way
    to record what a video actually is rather than trusting the seed row.
    """
    params = {"part": "snippet", "id": video_id, "key": api_key}
    request_url = f"{YOUTUBE_VIDEOS_ENDPOINT}?{urlencode(params)}"
    try:
        with urlopen(request_url, timeout=YOUTUBE_REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
        items = payload.get("items", [])
        if not items:
            return {}
        snippet = items[0].get("snippet", {})
        return {
            "api_title": snippet.get("title", ""),
            "api_channel": snippet.get("channelTitle", ""),
        }
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort provenance
        log.warning("Could not fetch video metadata for %s: %s", video_id, exc)
        return {}


def _existing_row_count(output_path: Path) -> int | None:
    """Return the row count of an already-fetched CSV, or None if unreadable."""
    try:
        return len(pd.read_csv(output_path))
    except Exception as exc:  # noqa: BLE001 - a stale/corrupt file must not abort the batch
        log.warning("Could not read existing output %s: %s", output_path, exc)
        return None


def fetch_youtube_comments_batch(
    seed_list_path: Path | str | None = None,
    max_results: int = 100,
    output_dir: Path | str | None = None,
    manifest_path: Path | str | None = DEFAULT_MANIFEST_PATH,
    force: bool = False,
    fetch_metadata: bool = False,
) -> pd.DataFrame:
    """Fetch comments for every video in the seed list and return a manifest.

    A failure on one video never aborts the batch; it is recorded in the manifest
    so a partial run stays auditable and re-runnable.
    """
    resolved_seed_path = _resolve_seed_list_path(seed_list_path)
    seeds = load_video_seed_list(resolved_seed_path)

    # Fail fast on a missing key rather than once per video.
    api_key = _resolve_api_key()

    resolved_dir = Path(output_dir) if output_dir else None
    entries: list[dict] = []

    for row in seeds.itertuples(index=False):
        video_id = row.video_id
        entry = {
            "video_id": video_id,
            "seed_title": getattr(row, "title", ""),
            "seed_channel": getattr(row, "channel", ""),
            "status": "",
            "rows_fetched": 0,
            "output_path": str(_build_output_path(video_id, resolved_dir)),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "error": "",
        }

        if _should_skip(video_id, output_dir, force):
            existing = _existing_row_count(_build_output_path(video_id, resolved_dir))
            entry["status"] = STATUS_SKIPPED
            entry["rows_fetched"] = existing if existing is not None else 0
            log.info("Skipping %s; output already exists (use --force to re-fetch)", video_id)
        else:
            try:
                records = fetch_youtube_comments(video_id, max_results, output_dir)
            except Exception as exc:  # noqa: BLE001 - HTTP/URL/timeout/JSON all mean
                # "this one video is unusable"; isolate it and keep the batch going.
                log.warning("Failed to fetch %s: %s", video_id, exc)
                entry["status"] = STATUS_FAILED
                entry["error"] = f"{type(exc).__name__}: {exc}"
            else:
                entry["rows_fetched"] = len(records)
                entry["status"] = STATUS_OK if records else STATUS_EMPTY

        if fetch_metadata:
            entry.update(_fetch_video_metadata(video_id, api_key))

        entries.append(entry)

    manifest = pd.DataFrame(entries)

    if manifest_path is not None:
        resolved_manifest = Path(manifest_path)
        resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(resolved_manifest, index=False)
        log.info("Wrote fetch manifest to %s", resolved_manifest)

    counts = manifest["status"].value_counts()
    log.info(
        "Batch complete: %s ok, %s skipped, %s failed, %s comments total",
        int(counts.get(STATUS_OK, 0)),
        int(counts.get(STATUS_SKIPPED, 0)),
        int(counts.get(STATUS_FAILED, 0)),
        int(manifest["rows_fetched"].sum()),
    )
    failed_ids = manifest.loc[manifest["status"] == STATUS_FAILED, "video_id"].tolist()
    if failed_ids:
        log.warning("Failed video ids: %s", failed_ids)

    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape YouTube comments for KH4 demand intelligence"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--video-id",
        help="YouTube video ID (the bit after v= in the URL)",
    )
    mode.add_argument(
        "--seed-list",
        nargs="?",
        const=str(DEFAULT_SEED_LIST_PATH),
        help=(
            "Fetch every video in a seed list CSV. Pass a path, or use the flag "
            f"bare to read {DEFAULT_SEED_LIST_PATH}"
        ),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Max comments to fetch per video (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(YOUTUBE_OUTPUT_DIR),
        help="Directory for output CSV files (default: data/raw/youtube)",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help=f"Batch mode only: manifest CSV path (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Batch mode only: re-fetch videos that already have an output CSV",
    )
    parser.add_argument(
        "--fetch-metadata",
        action="store_true",
        help="Batch mode only: also record each video's real title/channel via videos.list",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.max_results <= 0:
        raise ValueError("--max-results must be a positive integer")

    if args.seed_list is not None:
        log.info(
            "Starting YouTube batch scrape seed_list=%s max_results=%s output_dir=%s",
            args.seed_list,
            args.max_results,
            args.output_dir,
        )
        manifest = fetch_youtube_comments_batch(
            seed_list_path=args.seed_list,
            max_results=args.max_results,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            force=args.force,
            fetch_metadata=args.fetch_metadata,
        )
        # Exit non-zero on partial failure so quota/permission problems are visible.
        if (manifest["status"] == STATUS_FAILED).any():
            raise SystemExit(1)
        return

    log.info(
        "Starting YouTube scrape for video_id=%s max_results=%s output_dir=%s",
        args.video_id,
        args.max_results,
        args.output_dir,
    )
    fetch_youtube_comments(args.video_id, args.max_results, args.output_dir)


if __name__ == "__main__":
    main()

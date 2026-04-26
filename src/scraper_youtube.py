"""YouTube comment ingestion for KH4 demand intelligence.

Fetches top-level comments via YouTube Data API v3 commentThreads.list,
maps records into the shared SignalRecord schema, and persists to CSV.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from dotenv import load_dotenv

from src.schema import SignalRecord

YOUTUBE_COMMENTS_ENDPOINT = "https://www.googleapis.com/youtube/v3/commentThreads"
YOUTUBE_OUTPUT_PATH = Path("data/raw/youtube/youtube_comments.csv")


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
    with urlopen(request_url) as response:
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


def fetch_youtube_comments(video_id: str, max_results: int = 100) -> list[SignalRecord]:
    """Fetch top-level YouTube comments and persist them as raw CSV."""
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("Missing YOUTUBE_API_KEY in .env")

    records: list[SignalRecord] = []
    next_page_token: str | None = None

    while len(records) < max_results:
        page_size = min(100, max_results - len(records))
        payload = _fetch_comment_threads_page(
            video_id=video_id,
            api_key=api_key,
            max_results=page_size,
            page_token=next_page_token,
        )

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

    _save_records(records, YOUTUBE_OUTPUT_PATH)
    return records

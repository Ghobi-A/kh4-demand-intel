"""
Reddit scraper for Kingdom Hearts IV demand intelligence.

Uses PullPush (https://pullpush.io), a community-maintained Reddit archive
that mirrors the historical Pushshift API. No authentication required.

Why PullPush instead of PRAW:
- PRAW requires Reddit OAuth credentials, which depend on Reddit's
  app-creation form. That form has been silently failing for many users
  in 2025-2026 (CAPTCHA validates but POST never completes).
- PullPush is the de facto research replacement post-Pushshift shutdown
  and is actively maintained as of 2026.
- For a *historical* analysis project (KH4 reveal: June 2022), PullPush
  is actually the better tool — PRAW's search is capped to recent results,
  whereas PullPush returns the full archive.

PRAW path is preserved and will be re-enabled in v2 once Reddit's app
creation flow is working again.

Usage:
    python -m src.scraper_reddit [--limit 100] [--top-n 50]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.schema import SignalRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


PULLPUSH_BASE = "https://api.pullpush.io/reddit/search"

# Subreddits and search queries scoped for MVP
SUBREDDITS = ["KingdomHearts", "SquareEnix", "JRPG"]
QUERIES = [
    "kingdom hearts 4",
    "kh4",
    "kingdom hearts iv",
]

# PullPush is community-run; be polite
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "kh4-demand-intel/0.1 (educational portfolio project)"


def fetch_pullpush(endpoint: str, params: dict) -> list[dict]:
    """Hit a PullPush endpoint and return the raw 'data' list.

    endpoint: 'submission' for posts, 'comment' for comments.
    params: PullPush query params (q, subreddit, size, before, after, etc.)
    """
    url = f"{PULLPUSH_BASE}/{endpoint}/"
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(
            url, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        log.warning(f"PullPush request failed for {url} {params}: {e}")
        return []


def scrape_posts(limit_per_query: int) -> list[SignalRecord]:
    """Pull posts matching each query across each subreddit. Dedupe by ID."""
    seen_ids: set[str] = set()
    records: list[SignalRecord] = []

    for sub_name in SUBREDDITS:
        for query in QUERIES:
            log.info(f"Searching r/{sub_name} for '{query}' (limit={limit_per_query})")
            params = {
                "q": query,
                "subreddit": sub_name,
                "size": min(limit_per_query, 100),  # PullPush hard cap is 100/req
                "sort": "desc",
                "sort_type": "created_utc",
            }
            raw_posts = fetch_pullpush("submission", params)

            for post in raw_posts:
                post_id = post.get("id")
                if not post_id or post_id in seen_ids:
                    continue
                seen_ids.add(post_id)

                title = post.get("title", "") or ""
                selftext = post.get("selftext", "") or ""
                permalink = post.get("permalink", "")
                full_permalink = (
                    f"https://reddit.com{permalink}"
                    if permalink and not permalink.startswith("http")
                    else permalink
                )

                records.append(SignalRecord(
                    source="reddit",
                    record_type="post",
                    id=post_id,
                    text=f"{title}\n\n{selftext}".strip(),
                    author=post.get("author"),
                    timestamp=datetime.fromtimestamp(
                        post.get("created_utc", 0), tz=timezone.utc
                    ),
                    engagement=int(post.get("score", 0) or 0),
                    permalink=full_permalink,
                    parent_id=None,
                    metadata={
                        "subreddit": sub_name,
                        "title": title,
                        "num_comments": post.get("num_comments", 0),
                        "query": query,
                    },
                ))

            time.sleep(REQUEST_DELAY_SECONDS)

    log.info(f"Collected {len(records)} unique posts")
    return records


def scrape_comments(
    posts: list[SignalRecord],
    top_n: int = 50,
) -> list[SignalRecord]:
    """Pull comments for the top N posts by engagement."""
    top_posts = sorted(posts, key=lambda r: r.engagement, reverse=True)[:top_n]
    comment_records: list[SignalRecord] = []

    for i, post in enumerate(top_posts, 1):
        title_preview = str(post.metadata.get("title", post.id))[:60]
        log.info(f"[{i}/{len(top_posts)}] Comments for: {title_preview}")
        params = {
            "link_id": post.id,
            "size": 100,
            "sort": "desc",
            "sort_type": "score",
        }
        raw_comments = fetch_pullpush("comment", params)

        for comment in raw_comments:
            comment_id = comment.get("id")
            body = comment.get("body", "") or ""
            if not comment_id or not body or body in ("[deleted]", "[removed]"):
                continue

            permalink = comment.get("permalink", "")
            full_permalink = (
                f"https://reddit.com{permalink}"
                if permalink and not permalink.startswith("http")
                else permalink
            )

            comment_records.append(SignalRecord(
                source="reddit",
                record_type="comment",
                id=comment_id,
                text=body,
                author=comment.get("author"),
                timestamp=datetime.fromtimestamp(
                    comment.get("created_utc", 0), tz=timezone.utc
                ),
                engagement=int(comment.get("score", 0) or 0),
                permalink=full_permalink,
                parent_id=post.id,
                metadata={
                    "subreddit": post.metadata.get("subreddit"),
                },
            ))

        time.sleep(REQUEST_DELAY_SECONDS)

    log.info(f"Collected {len(comment_records)} comments")
    return comment_records


def save_records(records: list[SignalRecord], out_path: Path) -> None:
    """Write records to CSV. Metadata dict is JSON-encoded for portability."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        d = asdict(r)
        d["timestamp"] = r.timestamp.isoformat()
        d["metadata"] = json.dumps(r.metadata)
        rows.append(d)
    df = pd.DataFrame(rows)
    df["scraped_at"] = datetime.now(timezone.utc).isoformat()
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} records to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Reddit (via PullPush) for KH4 demand intelligence"
    )
    parser.add_argument("--limit", type=int, default=100,
                        help="Posts per subreddit/query (max 100, default: 100)")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Top N posts to scrape comments from (default: 50)")
    parser.add_argument("--output", type=Path,
                        default=Path("data/raw/reddit/reddit_kh4.csv"))
    parser.add_argument("--skip-comments", action="store_true",
                        help="Skip comment scraping (posts only)")
    args = parser.parse_args()
    if args.limit > 100:
        log.info("PullPush max page size is 100; capping --limit to 100.")
        args.limit = 100

    log.info("Using PullPush API (no Reddit auth required)")
    posts = scrape_posts(limit_per_query=args.limit)

    if not posts:
        log.error("No posts collected. Check your network or PullPush status.")
        return

    if args.skip_comments:
        all_records = posts
    else:
        comments = scrape_comments(posts, top_n=args.top_n)
        all_records = posts + comments

    save_records(all_records, args.output)


if __name__ == "__main__":
    main()

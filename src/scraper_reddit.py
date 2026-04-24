"""
Reddit scraper for Kingdom Hearts IV demand intelligence.

Scrapes posts and comments mentioning KH4 across relevant subreddits and
saves to data/raw/reddit/reddit_kh4.csv using the unified schema defined
in src/schema.py — so YouTube and other sources can plug in later
without pipeline changes.

Usage:
    python src/scraper_reddit.py [--limit 200] [--top-n 50]

Requires Reddit API credentials in .env (see .env.example).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
from dotenv import load_dotenv

from src.schema import SignalRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# Subreddits and search queries scoped for MVP
SUBREDDITS = ["KingdomHearts", "SquareEnix", "JRPG"]
QUERIES = [
    "kingdom hearts 4",
    "kh4",
    "kingdom hearts iv",
    "sora kh4",
]


def get_reddit_client() -> praw.Reddit:
    """Build a read-only PRAW client from env vars."""
    load_dotenv()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, user_agent]):
        log.error(
            "Missing Reddit credentials. Copy .env.example to .env and fill in."
        )
        sys.exit(1)

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def scrape_posts(reddit: praw.Reddit, limit: int) -> list[SignalRecord]:
    """Search each subreddit for each query, deduplicate, return posts."""
    seen_ids: set[str] = set()
    records: list[SignalRecord] = []

    for sub_name in SUBREDDITS:
        subreddit = reddit.subreddit(sub_name)
        for query in QUERIES:
            log.info(f"Searching r/{sub_name} for '{query}'")
            try:
                for post in subreddit.search(query, limit=limit, sort="relevance"):
                    if post.id in seen_ids:
                        continue
                    seen_ids.add(post.id)
                    records.append(SignalRecord(
                        source="reddit",
                        record_type="post",
                        id=post.id,
                        text=f"{post.title}\n\n{post.selftext or ''}".strip(),
                        author=str(post.author) if post.author else None,
                        timestamp=datetime.fromtimestamp(
                            post.created_utc, tz=timezone.utc
                        ),
                        engagement=post.score,
                        permalink=f"https://reddit.com{post.permalink}",
                        parent_id=None,
                        metadata={
                            "subreddit": sub_name,
                            "title": post.title,
                            "num_comments": post.num_comments,
                        },
                    ))
            except Exception as e:
                log.warning(f"Failed on r/{sub_name} query '{query}': {e}")

    log.info(f"Collected {len(records)} unique posts")
    return records


def scrape_comments(
    reddit: praw.Reddit,
    posts: list[SignalRecord],
    top_n: int = 50,
) -> list[SignalRecord]:
    """Pull top-level comments from the top N posts by engagement."""
    top_posts = sorted(posts, key=lambda r: r.engagement, reverse=True)[:top_n]
    comment_records: list[SignalRecord] = []

    for i, post in enumerate(top_posts, 1):
        title_preview = str(post.metadata.get("title", post.id))[:60]
        log.info(f"[{i}/{len(top_posts)}] Comments for: {title_preview}")
        try:
            submission = reddit.submission(id=post.id)
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                comment_records.append(SignalRecord(
                    source="reddit",
                    record_type="comment",
                    id=comment.id,
                    text=comment.body,
                    author=str(comment.author) if comment.author else None,
                    timestamp=datetime.fromtimestamp(
                        comment.created_utc, tz=timezone.utc
                    ),
                    engagement=comment.score,
                    permalink=f"https://reddit.com{comment.permalink}",
                    parent_id=post.id,
                    metadata={
                        "subreddit": post.metadata.get("subreddit"),
                    },
                ))
        except Exception as e:
            log.warning(f"Failed comments for post {post.id}: {e}")

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
        description="Scrape Reddit for KH4 demand intelligence signals"
    )
    parser.add_argument("--limit", type=int, default=200,
                        help="Posts per subreddit/query (default: 200)")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Top N posts to scrape comments from (default: 50)")
    parser.add_argument("--output", type=Path,
                        default=Path("data/raw/reddit/reddit_kh4.csv"))
    args = parser.parse_args()

    reddit = get_reddit_client()
    posts = scrape_posts(reddit, limit=args.limit)
    comments = scrape_comments(reddit, posts, top_n=args.top_n)
    all_records = posts + comments
    save_records(all_records, args.output)


if __name__ == "__main__":
    main()

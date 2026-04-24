"""Smoke tests for unified schema and Reddit scraper.

Real tests for scraper logic, preprocessing, and scoring will be added
in Week 3 once the pipeline matures.
"""

from datetime import datetime, timezone

from src.schema import SignalRecord


def test_signal_record_works_for_reddit_post() -> None:
    """SignalRecord should accept a Reddit post."""
    record = SignalRecord(
        source="reddit",
        record_type="post",
        id="abc123",
        text="When is KH4 coming out?",
        author="testuser",
        timestamp=datetime.now(timezone.utc),
        engagement=42,
        permalink="https://reddit.com/r/KingdomHearts/abc123",
        parent_id=None,
        metadata={"subreddit": "KingdomHearts", "title": "KH4 release"},
    )
    assert record.source == "reddit"
    assert record.record_type == "post"


def test_signal_record_works_for_youtube_comment() -> None:
    """Same schema must accept a future YouTube comment without changes."""
    record = SignalRecord(
        source="youtube",
        record_type="comment",
        id="ugxyz",
        text="Hyped for the new trailer!",
        author="fanaccount",
        timestamp=datetime.now(timezone.utc),
        engagement=120,
        permalink="https://youtube.com/watch?v=xyz",
        parent_id="video_id_here",
        metadata={"video_title": "KH4 official trailer"},
    )
    assert record.source == "youtube"
    assert record.record_type == "comment"


def test_subreddits_and_queries_are_defined() -> None:
    """Sanity: scraper has subreddits and queries to work with."""
    from src.scraper_reddit import QUERIES, SUBREDDITS
    assert len(SUBREDDITS) > 0
    assert len(QUERIES) > 0
    assert "KingdomHearts" in SUBREDDITS

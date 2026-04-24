"""
Unified signal schema for all data sources.

The same SignalRecord shape works for Reddit posts, Reddit comments,
YouTube comments, and any future source. New sources just need their
own scraper that emits SignalRecord — the rest of the pipeline
(preprocessing, sentiment, topic modelling, scoring) is source-agnostic.

This is the core abstraction that lets v1 ship Reddit-only without
needing a rewrite when v2 adds YouTube.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


SourceType = Literal["reddit", "youtube", "twitter", "trends"]
RecordType = Literal["post", "comment", "video", "trend"]


@dataclass
class SignalRecord:
    """A single piece of demand-signal content from any source.

    Attributes:
        source: Which platform the data came from.
        record_type: What kind of content (post, comment, video, etc.).
        id: Unique ID within the source platform.
        text: The actual content for sentiment/topic analysis.
        author: Username if available; None for anonymous or deleted.
        timestamp: When the content was created (UTC).
        engagement: Score/upvotes/likes — interpretation varies by source.
        permalink: Direct URL to the content.
        parent_id: ID of parent post for comments; None for top-level.
        metadata: Source-specific extras (subreddit, video title, etc.).
    """
    source: SourceType
    record_type: RecordType
    id: str
    text: str
    author: str | None
    timestamp: datetime
    engagement: int
    permalink: str
    parent_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

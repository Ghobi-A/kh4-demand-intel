from __future__ import annotations

import json

import pandas as pd
import pytest

from src import scraper_youtube


def test_fetch_youtube_comments_maps_to_signal_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")

    responses = [
        {
            "items": [
                {
                    "snippet": {
                        "totalReplyCount": 2,
                        "topLevelComment": {
                            "id": "comment_1",
                            "snippet": {
                                "textDisplay": "KH4 hype is real",
                                "authorDisplayName": "SoraFan",
                                "publishedAt": "2026-01-01T12:00:00Z",
                                "likeCount": 17,
                            },
                        },
                    }
                }
            ],
            "nextPageToken": "NEXT_PAGE",
        },
        {
            "items": [
                {
                    "snippet": {
                        "totalReplyCount": 0,
                        "topLevelComment": {
                            "id": "comment_2",
                            "snippet": {
                                "textDisplay": "Need release date soon",
                                "authorDisplayName": "KairiTalks",
                                "publishedAt": "2026-01-02T12:00:00Z",
                                "likeCount": 3,
                            },
                        },
                    }
                }
            ]
        },
    ]

    call_count = {"count": 0}

    def mock_page_fetch(
        video_id: str,
        api_key: str,
        max_results: int,
        page_token: str | None = None,
    ) -> dict:
        assert video_id == "video123"
        assert api_key == "test-key"
        assert max_results <= 100

        if call_count["count"] == 0:
            assert page_token is None
        if call_count["count"] == 1:
            assert page_token == "NEXT_PAGE"

        response = responses[call_count["count"]]
        call_count["count"] += 1
        return response

    monkeypatch.setattr(
        scraper_youtube,
        "_fetch_comment_threads_page",
        mock_page_fetch,
    )

    records = scraper_youtube.fetch_youtube_comments(
        video_id="video123",
        max_results=2,
        output_dir=tmp_path,
    )

    assert len(records) == 2
    assert records[0].source == "youtube"
    assert records[0].record_type == "comment"
    assert records[0].engagement == 17
    assert (
        records[0].permalink
        == "https://www.youtube.com/watch?v=video123&lc=comment_1"
    )
    assert records[0].metadata == {
        "video_id": "video123",
        "author_display_name": "SoraFan",
        "reply_count": 2,
    }
    assert records[1].id == "comment_2"
    assert records[1].metadata["reply_count"] == 0

    out_df = pd.read_csv(tmp_path / "youtube_comments_video123.csv")
    assert len(out_df) == 2
    assert out_df.loc[0, "source"] == "youtube"
    assert json.loads(out_df.loc[0, "metadata"])["video_id"] == "video123"


def test_fetch_youtube_comments_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr(scraper_youtube, "load_dotenv", lambda: None)

    with pytest.raises(ValueError, match="YOUTUBE_API_KEY"):
        scraper_youtube.fetch_youtube_comments(video_id="video123")


def test_build_output_path_uses_default_pattern() -> None:
    path = scraper_youtube._build_output_path("abc123")
    assert path == scraper_youtube.YOUTUBE_OUTPUT_DIR / "youtube_comments_abc123.csv"

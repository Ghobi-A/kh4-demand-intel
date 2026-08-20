from __future__ import annotations

import json
import urllib.error

import pandas as pd
import pytest

from src import scraper_youtube
from src.preprocess import REQUIRED_RAW_SIGNAL_COLUMNS


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


def _write_seed_csv(path, rows: list[str]) -> None:
    header = "video_id,title,channel,notes"
    path.write_text("\n".join([header, *rows]) + "\n")


def _stub_page_fetch(monkeypatch, comments_per_video: dict[str, int], failures=()):
    """Patch the page fetcher to return N synthetic comments per video id."""
    calls: list[str] = []

    def mock_page_fetch(video_id, api_key, max_results, page_token=None):
        calls.append(video_id)
        if video_id in failures:
            raise urllib.error.HTTPError(
                url="https://example.test", code=403, msg="commentsDisabled", hdrs=None, fp=None
            )
        items = [
            {
                "snippet": {
                    "totalReplyCount": 0,
                    "topLevelComment": {
                        "id": f"{video_id}_c{index}",
                        "snippet": {
                            "textDisplay": f"comment {index} on {video_id}",
                            "authorDisplayName": "Fan",
                            "publishedAt": "2026-08-16T12:00:00Z",
                            "likeCount": index,
                        },
                    },
                }
            }
            for index in range(comments_per_video.get(video_id, 0))
        ]
        return {"items": items}

    monkeypatch.setattr(scraper_youtube, "_fetch_comment_threads_page", mock_page_fetch)
    return calls


def test_load_video_seed_list_reads_video_ids(tmp_path) -> None:
    path = tmp_path / "seeds.csv"
    _write_seed_csv(path, ["abc123,A trailer,Nintendo,note", "def456,B trailer,KH,note"])

    seeds = scraper_youtube.load_video_seed_list(path)

    assert seeds["video_id"].tolist() == ["abc123", "def456"]
    assert "title" in seeds.columns


def test_load_video_seed_list_raises_on_missing_video_id_column(tmp_path) -> None:
    path = tmp_path / "seeds.csv"
    path.write_text("title,channel\nA trailer,Nintendo\n")

    with pytest.raises(ValueError, match="video_id"):
        scraper_youtube.load_video_seed_list(path)


def test_load_video_seed_list_raises_on_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        scraper_youtube.load_video_seed_list(tmp_path / "nope.csv")


def test_load_video_seed_list_drops_blank_and_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "seeds.csv"
    _write_seed_csv(
        path,
        [
            "abc123,A,Nintendo,note",
            " abc123 ,duplicate after strip,Nintendo,note",
            ",blank id,Nintendo,note",
            "def456,B,KH,note",
        ],
    )

    seeds = scraper_youtube.load_video_seed_list(path)

    assert seeds["video_id"].tolist() == ["abc123", "def456"]


def test_load_video_seed_list_raises_when_no_usable_rows(tmp_path) -> None:
    path = tmp_path / "seeds.csv"
    _write_seed_csv(path, [",blank,Nintendo,note"])

    with pytest.raises(ValueError, match="no usable"):
        scraper_youtube.load_video_seed_list(path)


def test_batch_fetches_every_seed_video(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    seed_path = tmp_path / "seeds.csv"
    _write_seed_csv(seed_path, ["vid1,First,Nintendo,note", "vid2,Second,KH,note"])
    _stub_page_fetch(monkeypatch, {"vid1": 2, "vid2": 3})

    manifest = scraper_youtube.fetch_youtube_comments_batch(
        seed_list_path=seed_path,
        max_results=10,
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifest.csv",
    )

    assert (tmp_path / "youtube_comments_vid1.csv").exists()
    assert (tmp_path / "youtube_comments_vid2.csv").exists()
    assert manifest["status"].tolist() == ["ok", "ok"]
    assert manifest["rows_fetched"].tolist() == [2, 3]
    assert manifest["seed_title"].tolist() == ["First", "Second"]


def test_batch_isolates_per_video_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    seed_path = tmp_path / "seeds.csv"
    _write_seed_csv(
        seed_path,
        ["vid1,First,Nintendo,note", "vid2,Second,KH,note", "vid3,Third,KH,note"],
    )
    _stub_page_fetch(monkeypatch, {"vid1": 1, "vid3": 1}, failures={"vid2"})

    manifest = scraper_youtube.fetch_youtube_comments_batch(
        seed_list_path=seed_path,
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifest.csv",
    )

    assert (tmp_path / "youtube_comments_vid1.csv").exists()
    assert (tmp_path / "youtube_comments_vid3.csv").exists()
    assert not (tmp_path / "youtube_comments_vid2.csv").exists()
    assert manifest["status"].tolist() == ["ok", "failed", "ok"]
    assert "HTTPError" in manifest.loc[1, "error"]


def test_batch_skips_existing_output(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    seed_path = tmp_path / "seeds.csv"
    _write_seed_csv(seed_path, ["vid1,First,Nintendo,note"])
    pd.DataFrame({"id": ["existing_1", "existing_2"]}).to_csv(
        tmp_path / "youtube_comments_vid1.csv", index=False
    )
    calls = _stub_page_fetch(monkeypatch, {"vid1": 5})

    manifest = scraper_youtube.fetch_youtube_comments_batch(
        seed_list_path=seed_path,
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifest.csv",
    )

    assert calls == []
    assert manifest.loc[0, "status"] == "skipped_existing"
    assert manifest.loc[0, "rows_fetched"] == 2


def test_batch_force_refetches(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    seed_path = tmp_path / "seeds.csv"
    _write_seed_csv(seed_path, ["vid1,First,Nintendo,note"])
    pd.DataFrame({"id": ["stale"]}).to_csv(tmp_path / "youtube_comments_vid1.csv", index=False)
    calls = _stub_page_fetch(monkeypatch, {"vid1": 3})

    manifest = scraper_youtube.fetch_youtube_comments_batch(
        seed_list_path=seed_path,
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifest.csv",
        force=True,
    )

    assert calls == ["vid1"]
    assert manifest.loc[0, "status"] == "ok"
    refetched = pd.read_csv(tmp_path / "youtube_comments_vid1.csv")
    assert len(refetched) == 3
    assert "stale" not in refetched.get("id", pd.Series(dtype=str)).tolist()


def test_batch_writes_manifest_that_is_not_signal_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    seed_path = tmp_path / "seeds.csv"
    _write_seed_csv(seed_path, ["vid1,First,Nintendo,note"])
    _stub_page_fetch(monkeypatch, {"vid1": 1})
    manifest_path = tmp_path / "manifest.csv"

    scraper_youtube.fetch_youtube_comments_batch(
        seed_list_path=seed_path,
        output_dir=tmp_path,
        manifest_path=manifest_path,
    )

    saved = pd.read_csv(manifest_path)
    expected = {
        "video_id",
        "seed_title",
        "seed_channel",
        "status",
        "rows_fetched",
        "output_path",
        "fetched_at",
        "error",
    }
    assert expected <= set(saved.columns)
    # The manifest must never be mistaken for raw signal data by the preprocessor.
    assert not REQUIRED_RAW_SIGNAL_COLUMNS <= set(saved.columns)


def test_batch_requires_api_key_before_any_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr(scraper_youtube, "load_dotenv", lambda: None)
    seed_path = tmp_path / "seeds.csv"
    _write_seed_csv(seed_path, ["vid1,First,Nintendo,note"])
    calls = _stub_page_fetch(monkeypatch, {"vid1": 1})

    with pytest.raises(ValueError, match="YOUTUBE_API_KEY"):
        scraper_youtube.fetch_youtube_comments_batch(
            seed_list_path=seed_path,
            output_dir=tmp_path,
            manifest_path=tmp_path / "manifest.csv",
        )

    assert calls == []


def test_parser_rejects_both_modes_together() -> None:
    parser = scraper_youtube._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--video-id", "abc", "--seed-list", "seeds.csv"])


def test_parser_requires_a_mode() -> None:
    parser = scraper_youtube._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_bare_seed_list_uses_default_path() -> None:
    args = scraper_youtube._build_parser().parse_args(["--seed-list"])
    assert args.seed_list == str(scraper_youtube.DEFAULT_SEED_LIST_PATH)


def test_fetch_video_metadata_returns_api_title_and_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"items": [{"snippet": {"title": "Real Title", "channelTitle": "Real Channel"}}]}

    class _Response:
        def read(self):
            return json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(scraper_youtube, "urlopen", lambda *a, **k: _Response())

    assert scraper_youtube._fetch_video_metadata("vid1", "test-key") == {
        "api_title": "Real Title",
        "api_channel": "Real Channel",
    }

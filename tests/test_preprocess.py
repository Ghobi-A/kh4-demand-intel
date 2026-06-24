"""Tests for raw-signal preprocessing."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.preprocess import clean_dataframe, load_raw_signals, run_preprocess_pipeline


RAW_SIGNAL_ROW = {
    "source": "youtube",
    "record_type": "comment",
    "id": "nested-1",
    "text": "Hello KH4 fans",
    "timestamp": "2024-01-01T00:00:00Z",
    "engagement": 7,
    "permalink": "https://example.com/comment/nested-1",
    "parent_id": "video-1",
    "metadata": '{"k":"v"}',
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_clean_dataframe_filters_and_normalizes() -> None:
    df = pd.DataFrame(
        [
            {
                "source": "reddit",
                "id": "keep-1",
                "text": "Hello KH4 fans",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": '{"k":"v"}',
            },
            {
                "source": "reddit",
                "id": "drop-deleted",
                "text": " [deleted] ",
                "timestamp": "2024-01-01T00:00:01Z",
                "metadata": '{"k":"v"}',
            },
            {
                "source": "youtube",
                "id": "drop-empty",
                "text": "   ",
                "timestamp": "2024-01-01T00:00:02Z",
                "metadata": '{"k":"v"}',
            },
            {
                "source": "youtube",
                "id": "keep-1",
                "text": "duplicate id should drop",
                "timestamp": "2024-01-01T00:00:03Z",
                "metadata": '{"k":"v"}',
            },
            {
                "source": "reddit",
                "id": "keep-2",
                "text": "See   https://a.com and https://b.com   now",
                "timestamp": "2024-01-01T00:00:04Z",
                "metadata": '{"k":"v"}',
            },
        ]
    )

    cleaned = clean_dataframe(df)

    assert list(cleaned["id"]) == ["keep-1", "keep-2"]
    assert list(cleaned["text"]) == ["Hello KH4 fans", "See and now"]
    assert pd.api.types.is_datetime64tz_dtype(cleaned["timestamp"])
    assert cleaned.loc[cleaned["id"] == "keep-1", "metadata"].iloc[0] == '{"k":"v"}'


def test_load_raw_signals_discovers_nested_source_csv(tmp_path: Path) -> None:
    nested_csv = tmp_path / "youtube" / "trailer_2026_06_09" / "comments.csv"
    _write_csv(nested_csv, [RAW_SIGNAL_ROW])

    loaded = load_raw_signals(tmp_path)

    assert len(loaded) == 1
    assert loaded.loc[0, "id"] == "nested-1"


def test_run_preprocess_pipeline_ignores_schema_invalid_reference_csv(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    valid_csv = tmp_path / "youtube" / "trailer_2026_06_09" / "comments.csv"
    reference_csv = tmp_path / "youtube" / "video_seed_list.example.csv"
    _write_csv(valid_csv, [RAW_SIGNAL_ROW])
    _write_csv(reference_csv, [{"video_id": "abc123", "url": "https://example.com/video"}])

    output_path = tmp_path / "processed" / "signals_clean.csv"
    with caplog.at_level(logging.INFO):
        cleaned = run_preprocess_pipeline(input_dir=tmp_path, output_path=output_path)

    assert list(cleaned["id"]) == ["nested-1"]
    assert output_path.exists()
    assert "video_seed_list.example.csv" in caplog.text
    assert "Skipping non-signal CSV" in caplog.text


def test_run_preprocess_pipeline_raises_clear_error_without_usable_raw_files(
    tmp_path: Path,
) -> None:
    _write_csv(
        tmp_path / "youtube" / "video_seed_list.example.csv",
        [{"video_id": "abc123", "url": "https://example.com/video"}],
    )
    output_path = tmp_path / "processed" / "signals_clean.csv"

    with pytest.raises(ValueError, match="No usable raw signal CSV files found"):
        run_preprocess_pipeline(input_dir=tmp_path, output_path=output_path)

    assert not output_path.exists()

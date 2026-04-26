"""Tests for raw-signal preprocessing."""

import pandas as pd

from src.preprocess import clean_dataframe


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

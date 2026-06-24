"""Week 2 preprocessing pipeline for raw multi-source KH4 signals."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/processed/signals_clean.csv")
URL_PATTERN = re.compile(r"https?://\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")
REQUIRED_RAW_SIGNAL_COLUMNS = {
    "source",
    "record_type",
    "id",
    "text",
    "timestamp",
    "engagement",
    "permalink",
    "parent_id",
    "metadata",
}


def _csv_paths_for_source(input_dir: Path, source_name: str) -> list[Path]:
    """Return all CSV files under a source directory, including nested captures."""
    return sorted((input_dir / source_name).rglob("*.csv"))


def _is_usable_raw_signal_csv(csv_path: Path) -> bool:
    """Return True when a CSV has the expected raw signal schema."""
    header = pd.read_csv(csv_path, nrows=0)
    missing_columns = REQUIRED_RAW_SIGNAL_COLUMNS - set(header.columns)
    if missing_columns:
        LOGGER.info(
            "Skipping non-signal CSV %s; missing required columns: %s",
            csv_path,
            sorted(missing_columns),
        )
        return False

    return True


def load_raw_signals(input_dir: Path = DEFAULT_INPUT_DIR) -> pd.DataFrame:
    """Load and concatenate all usable raw Reddit and YouTube CSV files."""
    source_to_paths = {
        "reddit": _csv_paths_for_source(input_dir, "reddit"),
        "youtube": _csv_paths_for_source(input_dir, "youtube"),
    }

    frames: list[pd.DataFrame] = []
    for source, paths in source_to_paths.items():
        source_rows = 0
        source_files = 0
        for csv_path in paths:
            if not _is_usable_raw_signal_csv(csv_path):
                continue

            frame = pd.read_csv(csv_path)
            frames.append(frame)
            source_rows += len(frame)
            source_files += 1
        LOGGER.info(
            "Loaded %s rows from %s usable %s CSV files",
            source_rows,
            source_files,
            source,
        )

    if not frames:
        raise ValueError(
            f"No usable raw signal CSV files found under {input_dir}. "
            f"Expected CSVs with columns: {sorted(REQUIRED_RAW_SIGNAL_COLUMNS)}"
        )

    return pd.concat(frames, ignore_index=True)


def _normalize_text(text: str) -> str:
    without_urls = URL_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", without_urls).strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw signals into a unified, analysis-ready DataFrame."""
    if df.empty:
        return df.copy()

    cleaned = df.copy()

    text_series = cleaned["text"].fillna("").astype(str)
    stripped = text_series.str.strip()

    empty_mask = stripped.eq("")
    deleted_mask = stripped.str.casefold().isin({"[deleted]", "[removed]"})
    duplicate_mask = cleaned.duplicated(subset=["id"], keep="first")

    LOGGER.info("Dropped %s rows with empty text", int(empty_mask.sum()))
    LOGGER.info("Dropped %s rows marked deleted/removed", int(deleted_mask.sum()))
    LOGGER.info("Dropped %s duplicate id rows", int(duplicate_mask.sum()))

    keep_mask = ~(empty_mask | deleted_mask | duplicate_mask)
    cleaned = cleaned.loc[keep_mask].copy()

    cleaned["text"] = cleaned["text"].fillna("").astype(str).map(_normalize_text)
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True, errors="coerce")

    return cleaned


def run_preprocess_pipeline(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run full preprocessing pipeline and save cleaned signals CSV."""
    raw_df = load_raw_signals(input_dir)
    cleaned_df = clean_dataframe(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    LOGGER.info("Final row count: %s", len(cleaned_df))
    LOGGER.info("Wrote cleaned signals to %s", output_path)

    return cleaned_df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess raw KH4 demand signals")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_parser().parse_args()
    run_preprocess_pipeline(input_dir=args.input_dir, output_path=args.output)


if __name__ == "__main__":
    main()

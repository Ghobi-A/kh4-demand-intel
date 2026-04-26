"""Week 2 VADER sentiment scoring pipeline for cleaned KH4 signals."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/processed/signals_clean.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/signals_sentiment.csv")


def label_from_compound(compound: float) -> str:
    """Map VADER compound score to a sentiment class."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def score_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Return a copy of DataFrame with VADER scores and sentiment label."""
    if text_column not in df.columns:
        raise ValueError(f"Expected column '{text_column}' in input data")

    analyzer = SentimentIntensityAnalyzer()
    scored = df.copy()

    scores = scored[text_column].fillna("").astype(str).map(analyzer.polarity_scores)
    scores_df = pd.DataFrame(list(scores), index=scored.index)

    scored["vader_neg"] = scores_df["neg"]
    scored["vader_neu"] = scores_df["neu"]
    scored["vader_pos"] = scores_df["pos"]
    scored["vader_compound"] = scores_df["compound"]
    scored["sentiment_label"] = scored["vader_compound"].map(label_from_compound)

    return scored


def run_sentiment_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Read cleaned signals, compute VADER sentiment, and save results."""
    df = pd.read_csv(input_path)
    scored = score_dataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    LOGGER.info("Rows scored: %s", len(scored))
    LOGGER.info("Sentiment label counts:\n%s", scored["sentiment_label"].value_counts())

    return scored


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score KH4 signals with VADER sentiment")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_parser().parse_args()
    run_sentiment_pipeline(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()

"""Run the full KH4 demand intelligence data pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.intent import run_intent_pipeline
from src.preprocess import run_preprocess_pipeline
from src.score import run_scoring_pipeline
from src.sentiment import run_sentiment_pipeline

LOGGER = logging.getLogger(__name__)

CLEAN_OUTPUT = Path("data/processed/signals_clean.csv")
SENTIMENT_OUTPUT = Path("data/processed/signals_sentiment.csv")
INTENT_OUTPUT = Path("data/processed/signals_intent.csv")
SCORED_OUTPUT = Path("data/processed/signals_scored.csv")
VIDEO_SCORES_OUTPUT = Path("reports/tables/video_scores.csv")


def main() -> None:
    """Run preprocessing, sentiment, intent, and scoring stages."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    LOGGER.info("Starting preprocess stage")
    run_preprocess_pipeline(output_path=CLEAN_OUTPUT)

    LOGGER.info("Starting sentiment stage")
    run_sentiment_pipeline(input_path=CLEAN_OUTPUT, output_path=SENTIMENT_OUTPUT)

    LOGGER.info("Starting intent stage")
    run_intent_pipeline(input_path=SENTIMENT_OUTPUT, output_path=INTENT_OUTPUT)

    LOGGER.info("Starting score stage")
    scored_df, _video_scores = run_scoring_pipeline(
        input_path=INTENT_OUTPUT,
        output_path=SCORED_OUTPUT,
        video_output_path=VIDEO_SCORES_OUTPUT,
    )

    print("Pipeline complete.")
    print(f"Scored rows: {len(scored_df)}")
    print("Outputs:")
    for output_path in [
        CLEAN_OUTPUT,
        SENTIMENT_OUTPUT,
        INTENT_OUTPUT,
        SCORED_OUTPUT,
        VIDEO_SCORES_OUTPUT,
    ]:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()

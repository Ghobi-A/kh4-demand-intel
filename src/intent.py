"""Rule-based intent classification for KH4 demand signals.

Reads sentiment-processed signals and adds two fields:
- intent_label
- intent_confidence

This is intentionally a simple MVP baseline built on keyword matching.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/signals_sentiment.csv")
OUTPUT_PATH = Path("data/processed/signals_intent.csv")

INTENT_KEYWORDS: dict[str, list[str]] = {
    "high_intent": [
        "day one",
        "preorder",
        "pre-order",
        "buying",
        "instant buy",
        "must buy",
        "can't wait",
    ],
    "frustrated_demand": [
        "where is",
        "still waiting",
        "4 years",
        "no news",
        "silence",
        "release date",
        "forgot this existed",
    ],
    "nostalgia_reactivation": [
        "kh1",
        "kh2",
        "childhood",
        "nostalgia",
        "ps2",
        "old kingdom hearts",
        "bring back",
    ],
    "new_customer_interest": [
        "marvel",
        "star wars",
        "pixar",
        "dreamworks",
        "disney plus",
        "disneyland",
        "new players",
    ],
    "confusion_barrier": [
        "confusing",
        "story makes no sense",
        "too complicated",
        "don't understand",
        "need recap",
    ],
}


def _matched_groups(text: str) -> list[str]:
    """Return the list of intent groups whose keywords appear in text."""
    normalized = text.casefold()
    matches: list[str] = []

    for label, keywords in INTENT_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            matches.append(label)

    return matches


def classify_intent(text: str) -> tuple[str, float]:
    """Classify one text string into an intent label and confidence."""
    matches = _matched_groups(text)

    if not matches:
        return "general_discussion", 0.3

    if len(matches) == 1:
        return matches[0], 0.7

    return matches[0], 1.0


def add_intent_columns(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Return a copy with `intent_label` and `intent_confidence` columns added."""
    if text_column not in df.columns:
        raise ValueError(f"Expected column '{text_column}' in input data")

    result = df.copy()
    classified = result[text_column].fillna("").astype(str).apply(classify_intent)
    result["intent_label"] = classified.str[0]
    result["intent_confidence"] = classified.str[1]
    return result


def run_intent_pipeline(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Load sentiment signals, classify intent, and save enriched output."""
    df = pd.read_csv(input_path)
    enriched = add_intent_columns(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)

    print(f"Rows processed: {len(enriched)}")
    print("Intent label counts:")
    print(enriched["intent_label"].value_counts())

    return enriched


if __name__ == "__main__":
    run_intent_pipeline()

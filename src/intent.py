"""Rule-based intent classification for KH4 demand signals.

Reads sentiment-processed signals and adds two fields:
- intent_label
- intent_confidence

This is intentionally a simple MVP baseline built on phrase/regex matching.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/signals_sentiment.csv")
OUTPUT_PATH = Path("data/processed/signals_intent.csv")

# Priority order from manual audit:
# 1) expectation_decay
# 2) high_intent
# 3) new_customer_interest
# 4) confusion_barrier
# 5) content_drought_fatigue
# 6) frustrated_demand
# 7) nostalgia_reactivation
INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "expectation_decay",
        [
            r"\bdon'?t care anymore\b",
            r"\blost interest\b",
            r"\bi(?: have|'ve) moved on\b",
            r"\bstopped caring\b",
            r"\bnot buying (?:it|this)\b",
            r"\bi(?: am|'m) not buying\b",
            r"\bno faith\b",
            r"\bno trust\b",
            r"\bdoesn'?t exist to me\b",
        ],
    ),
    (
        "high_intent",
        [
            r"\bbuy(?:ing)?\b",
            r"\bbought\b",
            r"\bpre[ -]?order\b",
            r"\bday one\b",
            r"\breplay(?:ing)?\b",
            r"\bplay again\b",
            r"\b100%\b",
            r"\bachievements?\b",
            r"\bcomplete (?:the )?series\b",
            r"\bi(?: am|'m) going to play\b",
            r"\bi(?: wi)?ll play\b",
            r"\bi(?: am|'m) getting (?:it|this)\b",
            r"\bi(?: have|'ve) bought (?:this|the) series\b",
        ],
    ),
    (
        "new_customer_interest",
        [
            r"\bnever played\b",
            r"\bfirst time playing\b",
            r"\bfinally (?:able|get) to play\b",
            r"\bnow (?:that|its|it is) on (?:steam|pc|switch)\b",
            r"\b(?:can|could) finally play\b",
            r"\bthinking about starting\b",
            r"\bnew to (?:kh|kingdom hearts)\b",
            r"\bentry point\b",
            r"\baccessible\b",
            r"\blocali[sz]ation\b",
        ],
    ),
    (
        "confusion_barrier",
        [
            r"\bconfusing\b",
            r"\bconvoluted\b",
            r"\bhard to follow\b",
            r"\bdon'?t understand\b",
            r"\bwhat order do i play\b",
            r"\bwhich games are required\b",
            r"\bmakes no sense\b",
            r"\btoo much information\b",
            r"\boverwhelmed\b",
            r"\b(?:i'?m|im) lost\b",
        ],
    ),
    (
        "content_drought_fatigue",
        [
            r"\bcopium\b",
            r"\bstarving\b",
            r"\bdrought\b",
            r"\bcrumbs\b",
            r"\bclown makeup\b",
            r"\b(?:we'?ll|i'?ll) take anything\b",
            r"\bat this point (?:we'?ll|i'?ll) take anything\b",
        ],
    ),
    (
        "frustrated_demand",
        [
            r"\bwhere is kh4\b",
            r"\bstill waiting\b",
            r"\bno (?:news|trailer|updates?)\b",
            r"\brelease date\b",
            r"\byears? of silence\b",
            r"\bgive us something\b",
            r"\bcontent drought\b",
            r"\b\d+ years?\b",
        ],
    ),
    (
        "nostalgia_reactivation",
        [
            r"\bwhen i was a kid\b",
            r"\b(?:was|is) (?:literally )?my childhood\b",
            r"\bgrew up with\b",
            r"\bsince i was \d+\b",
            r"\bremember playing\b",
            r"\bbrings me back\b",
        ],
    ),
]


def classify_intent(text: str) -> tuple[str, float]:
    """Classify one text string into an intent label and confidence."""
    normalized = text.casefold().strip()

    for label, patterns in INTENT_PATTERNS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return label, 0.7

    return "general_discussion", 0.3


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

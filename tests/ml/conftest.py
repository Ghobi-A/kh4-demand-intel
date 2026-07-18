"""Shared fixtures for ML tests.

The fixture corpus below is synthetic CI test data. It exists only so
tests never depend on scraping or on the real audit corpus, and it is
never treated as genuine human annotation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml.data import CANONICAL_COLUMNS

_PHRASES = {
    "high_intent": [
        "I will buy this day one for sure",
        "already pre ordered the deluxe edition",
        "taking time off work to play at launch",
        "getting it on release day no question",
    ],
    "frustrated_demand": [
        "still waiting for any news at all",
        "no trailer no updates nothing again",
        "years of silence is getting ridiculous",
        "give us something please square",
    ],
    "confusion_barrier": [
        "the timeline makes no sense to me",
        "what order do i play these games",
        "too convoluted to follow honestly",
        "completely lost about the story",
    ],
    "nostalgia_reactivation": [
        "grew up with these games as a kid",
        "this series was my whole childhood",
        "remember playing this every summer",
        "brings me back to simpler times",
    ],
    "general_discussion": [
        "the music in this scene is nice",
        "interesting theory about the character",
        "that boss design looks detailed",
        "the art style reminds me of the film",
        "cool video thanks for sharing",
        "the voice acting sounds different here",
        "nice breakdown of the trailer shots",
        "this channel always has good edits",
        "the lighting in that shot is pretty",
        "fun discussion in the comments today",
        "that cameo surprised everyone i think",
        "the soundtrack version here is orchestral",
    ],
}


def make_fixture_frame(groups_per_class: int = 3) -> pd.DataFrame:
    """Small labelled frame with real-looking group structure for split tests."""
    rows = []
    counter = 0
    for label, phrases in _PHRASES.items():
        for i, phrase in enumerate(phrases * groups_per_class):
            counter += 1
            rows.append(
                {
                    "record_id": f"r{counter}",
                    "text": f"{phrase} variant {counter}",
                    "corrected_label": label,
                    "source": "youtube" if counter % 3 else "reddit",
                    "parent_id": f"g{label[:4]}{i % groups_per_class}",
                    "group_id_source": "permalink_derived",
                    "created_at": None,
                    "annotator_id": "fixture",
                    "annotation_version": "ci_fixture",
                    "sample_strategy": "fixture",
                    "original_rule_label": "general_discussion",
                    "secondary_label": None,
                    "multi_intent_flag": None,
                }
            )
    frame = pd.DataFrame(rows)
    from src.ml.data import is_actionable

    frame["actionable"] = frame["corrected_label"].map(is_actionable)
    return frame[CANONICAL_COLUMNS]


@pytest.fixture
def fixture_frame() -> pd.DataFrame:
    return make_fixture_frame()

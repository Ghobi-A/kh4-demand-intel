from pathlib import Path

import pandas as pd

from src.intent import classify_intent


AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "audits"
    / "intent_audit_corrected.csv"
)

# Precision floors are derived from the manually audited corpus.
# high_intent and nostalgia_reactivation intentionally use lower
# acceptance floors due to semantic overlap with
# expectation_decay, frustration, and general discussion.

PRECISION_FLOORS = {
    "high_intent": 0.55,
    "nostalgia_reactivation": 0.65,
    "new_customer_interest": 0.70,
}


def _load_audit() -> pd.DataFrame:
    return pd.read_csv(AUDIT_PATH)


def test_precision_slices_meet_precision_floors() -> None:
    audit = _load_audit()
    target_classes = [
        "high_intent",
        "nostalgia_reactivation",
        "new_customer_interest",
    ]

    for target_class in target_classes:
        precision_slice = audit[
            (audit["sample_type"] == "precision")
            & (audit["intent_label"] == target_class)
        ]

        predictions = precision_slice["text"].apply(
            lambda text: classify_intent(str(text))[0]
        )

        precision = (
            (predictions == precision_slice["corrected_label"]).sum()
            / len(precision_slice)
        )

        assert precision >= PRECISION_FLOORS[target_class], (
            f"{target_class} precision below precision floor: {precision:.2%}"
        )


def test_recall_slice_false_positive_contamination() -> None:
    """Measure false-positive contamination of recall slice, not classical recall."""
    audit = _load_audit()
    recall_slice = audit[audit["sample_type"] == "recall"]

    predictions = recall_slice["text"].apply(lambda text: classify_intent(str(text))[0])

    miss_rate = (
        (
            (predictions != "general_discussion")
            & (recall_slice["corrected_label"] == "general_discussion")
        ).sum()
        / len(recall_slice)
    )

    assert miss_rate <= 0.20, (
        f"Recall-slice false-positive contamination too high: {miss_rate:.2%}"
    )

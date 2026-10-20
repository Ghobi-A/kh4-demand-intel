"""Tests for double annotation and adjudication.

All annotations below are synthetic test fixtures written for these tests.
They are not, and are never used as, labels for the real corpus.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml.adjudication import adjudication_queue, agreement_report, finalise, pair_annotations


def _annotator(name: str, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "queue_id": [f"q{i}" for i in range(len(labels))],
            "text": [f"synthetic comment {i}" for i in range(len(labels))],
            "human_label": labels,
            "annotator_id": name,
        }
    )


A = ["general_discussion", "frustrated_demand", "confusion_barrier", "general_discussion",
     "high_intent", "general_discussion"]
B = ["general_discussion", "frustrated_demand", "general_discussion", "general_discussion",
     "nostalgia_reactivation", "general_discussion"]


def test_agreement_report_counts_and_kappa() -> None:
    report = agreement_report(pair_annotations(_annotator("ann_a", A), _annotator("ann_b", B)), n_boot=50)
    assert report["items"] == 6
    assert report["raw_agreement"] == pytest.approx(4 / 6)
    assert report["disagreement_items"] == 2
    # high_intent vs nostalgia are both actionable, so the binary target agrees there.
    assert report["actionable_raw_agreement"] == pytest.approx(5 / 6)
    assert {row["class"] for row in report["per_class"]} >= {"confusion_barrier", "high_intent"}


def test_same_annotator_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="two people"):
        pair_annotations(_annotator("ann_a", A), _annotator("ann_a", B))


def test_labels_outside_taxonomy_are_rejected() -> None:
    with pytest.raises(ValueError, match="taxonomy"):
        pair_annotations(_annotator("ann_a", A[:-1] + ["spam"]), _annotator("ann_b", B))


def test_adjudication_queue_is_blind_to_annotators() -> None:
    paired = pair_annotations(_annotator("ann_a", A), _annotator("ann_b", B))
    queue = adjudication_queue(paired)
    assert len(queue) == 2
    assert not {"annotator_id", "annotator_id_a", "human_label_a"} & set(queue.columns)
    assert (queue["adjudicated_label"] == "").all()


def test_finalise_requires_independent_adjudicator_and_rationale() -> None:
    paired = pair_annotations(_annotator("ann_a", A), _annotator("ann_b", B))
    queue = adjudication_queue(paired)
    queue["adjudicated_label"] = ["confusion_barrier", "high_intent"]
    queue["rationale"] = ["obstacle to engagement", ""]
    queue["adjudicator_id"] = "adj_c"
    gold, report = finalise(paired, queue, version="gold_test")
    assert report["agreed"] == 4 and report["adjudicated"] == 1 and report["still_disputed"] == 1
    assert report["problems"]["missing_rationale"] == 1
    assert set(gold["label_source"]) == {"agreed", "adjudicated"}
    assert (gold["label_status"] == "gold_candidate_unmerged").all()

    queue["adjudicator_id"] = "ann_a"
    with pytest.raises(ValueError, match="independent"):
        finalise(paired, queue)

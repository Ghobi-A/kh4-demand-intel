"""Tests for the stratified annotation queue and its quality controls.

All rows below are synthetic test fixtures, never treated as annotation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ml.annotation_queue import (
    ANNOTATOR_COLUMNS,
    STRATUM_RANDOM,
    STRATUM_REDDIT,
    STRATUM_UNSEEN_GROUP,
    prepare_pool,
    score_pool,
    select_queue,
    validate_completed_batch,
)


def _labelled() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["l1", "l2", "l3"],
            "text": [
                "still waiting for any news at all",
                "what order do i play these games",
                "nice trailer music",
            ],
            "corrected_label": ["frustrated_demand", "confusion_barrier", "general_discussion"],
            "parent_id": ["yt:a", "yt:a", "yt:b"],
        }
    )


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "still waiting for any news at all",  # already labelled
                "Still waiting for any news at all!",  # near-duplicate of labelled
                "Pre-ordering the second it's available",
                "Pre-ordering the second it's available",  # duplicate in pool
                "the music in this trailer is great",
                "anyone else replaying the whole series",
                "this thread is about the new world reveal",
                "lol",
            ],
            "source": ["youtube", "youtube", "youtube", "youtube", "reddit", "reddit", "youtube", "youtube"],
            "parent_id": ["a", "a", "c", "c", "r1", "r2", "b", "d"],
        }
    )


def test_pool_excludes_labelled_duplicate_and_near_duplicate_rows() -> None:
    frame, excluded = prepare_pool(_pool(), _labelled())
    assert excluded["already_labelled"] == 1
    assert excluded["duplicate_in_pool"] == 1
    assert excluded["near_duplicate_of_labelled"] == 1
    assert frame["queue_id"].is_unique
    assert frame["parent_id"].str.contains(":").all()


def test_queue_strata_quotas_and_shortfalls_are_reported() -> None:
    labelled = _labelled()
    frame, _ = prepare_pool(_pool(), labelled)
    scored = score_pool(frame, model=None)
    quotas = {STRATUM_RANDOM: 1, STRATUM_REDDIT: 5, STRATUM_UNSEEN_GROUP: 5}
    queue, report = select_queue(scored, labelled, quotas=quotas, seed=1)

    assert queue["queue_id"].is_unique
    assert report[STRATUM_RANDOM]["selected"] == 1
    assert report[STRATUM_REDDIT]["shortfall"] >= 3  # only two reddit rows exist
    reddit_rows = queue[queue["stratum"] == STRATUM_REDDIT]
    assert (reddit_rows["source"] == "reddit").all()
    unseen = queue[queue["stratum"] == STRATUM_UNSEEN_GROUP]
    assert not unseen["parent_id"].isin(set(labelled["parent_id"])).any()


def test_annotator_file_is_blind_to_model_and_rule_outputs() -> None:
    assert not {"rule_label", "model_label", "model_probability"} & set(ANNOTATOR_COLUMNS)
    assert "human_label" in ANNOTATOR_COLUMNS


def test_validation_rejects_bad_rows_and_never_fills_labels() -> None:
    provenance = pd.DataFrame({"queue_id": ["q1", "q2", "q3", "q4"], "stratum": ["x"] * 4})
    completed = pd.DataFrame(
        {
            "queue_id": ["q1", "q2", "q3", "q4", "q1"],
            "text": ["a new comment", "another one", "still waiting for any news at all", "x", "a new comment"],
            "human_label": ["high_intent", "not_a_class", "general_discussion", "", "high_intent"],
            "annotator_id": ["ann_a", "ann_a", "ann_a", "ann_a", "ann_b"],
        }
    )
    accepted, report = validate_completed_batch(completed, provenance, _labelled())

    assert report["problems"]["label_not_in_taxonomy"] == 1
    assert report["problems"]["already_in_corpus"] == 1
    assert report["problems"]["unlabelled_rows"] == 1
    assert set(accepted["queue_id"]) == {"q1"}
    assert (accepted["label_status"] == "human_reviewed_unmerged").all()
    assert report["inter_annotator_agreement"]["items"] == 1
    assert report["inter_annotator_agreement"]["raw_agreement"] == 1.0


def test_group_cap_and_inclusion_probability_are_recorded() -> None:
    labelled = _labelled()
    pool = pd.DataFrame(
        {
            "text": [f"distinct synthetic comment number {i} about the game" for i in range(12)],
            "source": ["youtube"] * 12,
            "parent_id": ["big"] * 10 + ["small1", "small2"],
        }
    )
    frame, _ = prepare_pool(pool, labelled)
    scored = score_pool(frame, model=None)
    queue, report = select_queue(scored, labelled, quotas={STRATUM_RANDOM: 8}, seed=3, max_per_group=3)
    assert (queue["parent_id"].value_counts() <= 3).all()
    assert len(queue) == 5  # 3 from the big group + 2 singletons; the shortfall is reported
    assert report[STRATUM_RANDOM]["shortfall"] == 3
    assert queue["inclusion_probability"].iloc[0] == pytest.approx(5 / 12)

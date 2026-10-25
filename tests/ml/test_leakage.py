"""Synthetic near-duplicate groups must remain together at both split levels."""

import pandas as pd

from src.ml.leakage import link_near_duplicate_groups
from src.ml.probabilities import group_folds


def test_text_connected_groups_are_linked_without_using_labels():
    frame = pd.DataFrame({
        "record_id": ["a", "b", "c", "d"],
        "parent_id": ["video_a", "video_b", "video_b", "video_c"],
        "text": ["this is the exact same long repeated fan comment"] * 2
                + ["unrelated music discussion", "entirely different release question"],
    })
    linked, audit = link_near_duplicate_groups(frame)
    assert audit["original_groups"] == 3 and audit["effective_groups"] == 2
    assert linked.loc[:2, "parent_id"].nunique() == 1
    assert frame["parent_id"].nunique() == 3  # input remains untouched
    for held in group_folds(linked, n_folds=2):
        assert set(held) in ({0, 1, 2}, {3})

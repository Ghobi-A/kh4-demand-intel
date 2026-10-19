"""Tests for the dataset audit and group-aware CV helpers (synthetic fixtures)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.dataset_audit import near_duplicate_pairs, revision_evidence
from src.ml.group_cv import calibration_table, group_bootstrap, oof_predictions
from tests.ml.conftest import make_fixture_frame


def test_near_duplicates_flag_cross_group_pairs() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b", "c"],
            "text": ["trailers coming soon like mulan", "trailers coming soon like mulan!", "x y z"],
            "parent_id": ["yt:1", "yt:2", "yt:1"],
            "corrected_label": ["general_discussion"] * 3,
        }
    )
    pairs = near_duplicate_pairs(frame)
    assert len(pairs) == 1
    assert bool(pairs["crosses_groups"].iloc[0])


def test_revision_evidence_is_labelled_as_same_annotator() -> None:
    raw = pd.DataFrame(
        {
            "manual_label": ["high_intent", "general_discussion", "", "frustrated_demand"],
            "corrected_label": ["general_discussion", "general_discussion", "general_discussion", "frustrated_demand"],
        }
    )
    evidence = revision_evidence(raw)
    assert evidence["rows_with_both_passes"] == 3
    assert evidence["revised_rows"] == 1
    assert "not how reproducible" in evidence["interpretation"]


def test_calibration_table_counts_every_row() -> None:
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.05, 0.3, 0.5, 0.9, 1.0])
    table = calibration_table(y, p, n_bins=5)
    assert table["n"].sum() == len(y)


def test_oof_predictions_cover_rows_without_seeing_their_group() -> None:
    frame = make_fixture_frame(groups_per_class=4)
    oof = oof_predictions(frame, "logistic", seed=42, n_folds=4)
    scored = oof[oof["p_actionable"].notna()]
    assert len(scored) > 0.5 * len(frame)
    assert scored["p_actionable"].between(0, 1).all()
    merged = frame.merge(oof, on="record_id")
    # Every discussion group sits in exactly one outer fold.
    assert (merged[merged["fold"] >= 0].groupby("parent_id")["fold"].nunique() == 1).all()


def test_group_bootstrap_returns_ordered_intervals() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "parent_id": np.repeat([f"g{i}" for i in range(8)], 10),
            "true_actionable": rng.integers(0, 2, 80),
        }
    )
    frame["p_actionable"] = np.clip(frame["true_actionable"] * 0.4 + rng.random(80) * 0.6, 0, 1)
    intervals = group_bootstrap(frame, n_boot=50)
    low, high = intervals["pr_auc"]
    assert low <= high

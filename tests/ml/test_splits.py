import pandas as pd
import pytest

from src.ml.splits import (
    EVALUATION_MODE_DEV_ONLY,
    EVALUATION_MODE_GROUP,
    check_leakage,
    group_split,
    save_split_manifests,
    temporal_split,
)


def test_group_split_has_no_group_leakage(fixture_frame):
    split = group_split(fixture_frame, seed=42)
    assert check_leakage(split) == []
    assert split.evaluation_mode == EVALUATION_MODE_GROUP
    total = len(split.train) + len(split.validation) + len(split.test)
    assert total == len(fixture_frame)


def test_group_split_is_seed_reproducible(fixture_frame):
    first = group_split(fixture_frame, seed=42)
    second = group_split(fixture_frame, seed=42)
    assert first.train["record_id"].tolist() == second.train["record_id"].tolist()
    assert first.test["record_id"].tolist() == second.test["record_id"].tolist()

    different = group_split(fixture_frame, seed=99)
    assert (
        first.train["record_id"].tolist() != different.train["record_id"].tolist()
        or first.test["record_id"].tolist() != different.test["record_id"].tolist()
    )


def test_missing_groups_mark_development_only(fixture_frame):
    frame = fixture_frame.copy()
    frame.loc[frame.index[:10], "parent_id"] = None
    split = group_split(frame, seed=42)
    assert split.evaluation_mode == EVALUATION_MODE_DEV_ONLY


def test_bad_fractions_raise(fixture_frame):
    with pytest.raises(ValueError):
        group_split(fixture_frame, train_frac=0.9, validation_frac=0.3, test_frac=0.3)


def test_check_leakage_detects_shared_group(fixture_frame):
    split = group_split(fixture_frame, seed=42)
    leaked_row = split.train.head(1)
    split.test = pd.concat([split.test, leaked_row], ignore_index=True)
    problems = check_leakage(split)
    assert problems and any("shared" in p.lower() for p in problems)


def test_real_audit_split_has_no_leakage():
    from src.ml.data import load_labelled_data

    df = load_labelled_data()
    for seed in [42, 43, 44]:
        split = group_split(df, seed=seed)
        assert check_leakage(split) == []
        assert split.evaluation_mode == EVALUATION_MODE_GROUP


def test_temporal_split_requires_timestamps(fixture_frame):
    with pytest.raises(ValueError):
        temporal_split(fixture_frame)
    frame = fixture_frame.copy()
    frame["created_at"] = pd.date_range("2024-01-01", periods=len(frame), freq="D")
    split = temporal_split(frame)
    assert split.evaluation_mode == "temporal_holdout"
    assert split.train["created_at"].max() <= split.test["created_at"].min()


def test_save_split_manifests(tmp_path, fixture_frame):
    split = group_split(fixture_frame, seed=42)
    save_split_manifests(split, tmp_path)
    for name in ["train", "validation", "test", "split_summary"]:
        assert (tmp_path / f"{name}.csv").exists()

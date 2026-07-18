import pandas as pd
import pytest

from src.ml.data import (
    ACTIONABLE_CLASSES,
    NON_ACTIONABLE_CLASSES,
    TAXONOMY,
    dataset_hash,
    derive_group_id,
    is_actionable,
    load_labelled_data,
    validate_labelled_data,
)


def test_every_taxonomy_class_maps_to_exactly_one_binary_category():
    assert set(ACTIONABLE_CLASSES) & set(NON_ACTIONABLE_CLASSES) == set()
    for label in TAXONOMY:
        assert is_actionable(label) in (True, False)
    assert sorted(TAXONOMY) == sorted(set(ACTIONABLE_CLASSES) | set(NON_ACTIONABLE_CLASSES))


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        is_actionable("not_a_class")


def test_derive_group_id_youtube_and_reddit():
    yt = derive_group_id("https://www.youtube.com/watch?v=G5_3qdf19BU&lc=Ugz123", "youtube")
    assert yt == "yt:G5_3qdf19BU"
    rd = derive_group_id("https://reddit.com/r/JRPG/comments/1kmwpfq/title/msf9rc2/", "reddit")
    assert rd == "rd:1kmwpfq"
    assert derive_group_id(None, "youtube") is None
    assert derive_group_id("https://example.com/nothing", "youtube") is None


def test_load_real_audit_corpus_has_canonical_schema():
    df = load_labelled_data()
    assert len(df) == 200
    assert df["record_id"].is_unique
    assert df["parent_id"].notna().all()
    assert set(df["group_id_source"]) == {"permalink_derived"}
    assert df["actionable"].isin([True, False]).all()


def test_validate_flags_duplicates_and_conflicts(fixture_frame):
    report = validate_labelled_data(fixture_frame)
    assert report["valid"]

    corrupted = pd.concat([fixture_frame, fixture_frame.head(1)], ignore_index=True)
    corrupted.loc[len(corrupted) - 1, "corrected_label"] = "frustrated_demand"
    corrupted.loc[len(corrupted) - 1, "record_id"] = corrupted.loc[0, "record_id"]
    report = validate_labelled_data(corrupted)
    assert not report["valid"]
    assert report["duplicate_texts"] >= 1
    assert report["conflicting_labels"] >= 1


def test_validate_flags_missing_text_and_unknown_labels(fixture_frame):
    frame = fixture_frame.copy()
    frame.loc[0, "text"] = ""
    frame.loc[1, "corrected_label"] = "mystery_class"
    report = validate_labelled_data(frame)
    assert not report["valid"]
    assert any("missing or empty text" in issue for issue in report["issues"])
    assert any("unknown labels" in issue for issue in report["issues"])


def test_dataset_hash_is_deterministic_and_content_sensitive(fixture_frame):
    first = dataset_hash(fixture_frame)
    assert first == dataset_hash(fixture_frame.sample(frac=1.0, random_state=1))
    changed = fixture_frame.copy()
    changed.loc[0, "corrected_label"] = "general_discussion"
    assert dataset_hash(changed) != first

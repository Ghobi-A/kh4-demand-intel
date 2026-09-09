"""Tests for calibrated actionable probabilities and their provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.hierarchy import HierarchicalClassifier
from src.ml.probabilities import (
    PROBABILITY_COLUMN,
    SOURCE_EXCLUDED,
    PROVENANCE_COLUMN,
    SOURCE_OOF,
    SOURCE_PERSISTED,
    SOURCE_UNAVAILABLE,
    ProbabilityUnavailableError,
    attach_actionable_probabilities,
    build_probabilities,
    group_folds,
    oof_actionable_probabilities,
)
from tests.ml.conftest import make_fixture_frame


def test_group_folds_never_split_a_discussion_group() -> None:
    frame = make_fixture_frame()

    folds = group_folds(frame, n_folds=4, seed=42)

    assert sum(len(fold) for fold in folds) == len(frame)
    seen: dict[str, int] = {}
    for index, fold in enumerate(folds):
        for group in frame.iloc[fold]["parent_id"]:
            assert seen.setdefault(group, index) == index


def test_group_folds_are_deterministic_for_a_seed() -> None:
    frame = make_fixture_frame()

    first = [fold.tolist() for fold in group_folds(frame, seed=7)]
    second = [fold.tolist() for fold in group_folds(frame, seed=7)]

    assert first == second


def test_oof_probabilities_are_out_of_sample_for_every_row() -> None:
    frame = make_fixture_frame()

    probabilities = oof_actionable_probabilities(frame, n_folds=4, seed=42)

    assert probabilities
    assert set(probabilities).issubset(set(frame["record_id"].astype(str)))
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())
    # Not all identical: a constant would signal a degenerate fit.
    assert len(set(np.round(list(probabilities.values()), 6))) > 1


def test_oof_fold_model_never_sees_the_row_it_scores(monkeypatch) -> None:
    frame = make_fixture_frame()
    observed: list[tuple[set, set]] = []

    import src.ml.probabilities as probabilities_module

    original = probabilities_module.train_stage1

    def _record(model_name, split, config, seed):
        observed.append((set(split.train["record_id"]), set(split.validation["record_id"])))
        return original(model_name, split, config, seed)

    monkeypatch.setattr(probabilities_module, "train_stage1", _record)
    result = oof_actionable_probabilities(frame, n_folds=4, seed=42)

    assert result
    groups_by_record = dict(zip(frame["record_id"].astype(str), frame["parent_id"].astype(str)))
    for train_ids, validation_ids in observed:
        fitted_groups = {groups_by_record[str(r)] for r in train_ids | validation_ids}
        scored = set(groups_by_record) - {str(r) for r in train_ids | validation_ids}
        scored_groups = {groups_by_record[r] for r in scored}
        assert not (fitted_groups & scored_groups)


def test_labelled_rows_take_out_of_fold_values_not_in_sample_ones() -> None:
    signals = pd.DataFrame({"id": ["r1", "r2"], "text": ["buying it", "nice music"]})
    model = _StubModel(0.42)

    scored = attach_actionable_probabilities(signals, model, {"r1": 0.9})

    assert scored.loc[0, PROVENANCE_COLUMN] == SOURCE_OOF
    assert scored.loc[0, PROBABILITY_COLUMN] == pytest.approx(0.9)
    assert scored.loc[1, PROVENANCE_COLUMN] == SOURCE_PERSISTED
    assert scored.loc[1, PROBABILITY_COLUMN] == pytest.approx(0.42)


def test_rows_are_matched_by_text_when_identifiers_were_stripped() -> None:
    """The portfolio extract strips ids; without a text fallback those audit
    rows would be scored in-sample by a model that trained on them."""
    signals = pd.DataFrame({"text": ["  Buying   IT day one ", "unrelated comment"]})
    text_map = {"buying it day one": 0.88}

    scored = attach_actionable_probabilities(
        signals, _StubModel(0.42), {}, text_oof_map=text_map
    )

    assert scored.loc[0, PROVENANCE_COLUMN] == SOURCE_OOF
    assert scored.loc[0, PROBABILITY_COLUMN] == pytest.approx(0.88)
    assert scored.loc[1, PROVENANCE_COLUMN] == SOURCE_PERSISTED


def test_id_match_takes_priority_over_text_match() -> None:
    signals = pd.DataFrame({"id": ["r1"], "text": ["buying it"]})

    scored = attach_actionable_probabilities(
        signals, _StubModel(0.42), {"r1": 0.9}, text_oof_map={"buying it": 0.1}
    )

    assert scored.loc[0, PROBABILITY_COLUMN] == pytest.approx(0.9)


def test_out_of_fold_keys_can_be_generated_by_text() -> None:
    frame = make_fixture_frame()

    by_text = oof_actionable_probabilities(frame, n_folds=4, seed=42, by_text=True)

    assert by_text
    expected = {" ".join(str(t).split()).casefold() for t in frame["text"]}
    assert set(by_text).issubset(expected)


def test_rows_without_text_are_unavailable_rather_than_guessed() -> None:
    signals = pd.DataFrame({"id": ["r1"], "text": ["   "]})

    scored = attach_actionable_probabilities(signals, _StubModel(0.42), {})

    assert scored.loc[0, PROVENANCE_COLUMN] == SOURCE_UNAVAILABLE
    assert np.isnan(scored.loc[0, PROBABILITY_COLUMN])


def test_uncalibrated_model_yields_no_probabilities_instead_of_rule_confidence() -> None:
    signals = pd.DataFrame({"id": ["r1"], "text": ["buying it"]})

    scored = attach_actionable_probabilities(signals, _StubModel(0.42, probability=False), {})

    assert scored.loc[0, PROVENANCE_COLUMN] == SOURCE_UNAVAILABLE
    assert np.isnan(scored.loc[0, PROBABILITY_COLUMN])
    # The 0.7/0.3 rule confidence indicator must never appear as a probability.
    assert 0.7 not in set(scored[PROBABILITY_COLUMN].dropna())


def test_build_probabilities_reports_unavailable_rather_than_fabricating(monkeypatch) -> None:
    import src.ml.probabilities as probabilities_module

    def _fail(**kwargs):
        raise ProbabilityUnavailableError("no artefacts")

    monkeypatch.setattr(probabilities_module, "ensure_hierarchical_model", _fail)

    with pytest.raises(ProbabilityUnavailableError):
        build_probabilities(pd.DataFrame({"id": ["a"], "text": ["hi"]}))


class _StubModel(HierarchicalClassifier):
    """Minimal stand-in exposing the Stage 1 scoring contract."""

    def __init__(self, value: float, probability: bool = True):
        self.value = value
        self.calibrator = None
        self.stage1 = type(
            "Stage1", (), {"score_type": "probability" if probability else "margin"}
        )()

    def stage1_scores(self, texts):
        return np.full(len(list(texts)), self.value, dtype=float)


def test_training_row_without_oof_is_excluded_not_scored_in_sample() -> None:
    """The core provenance guarantee: no silent fallback to an in-sample score.

    A row the model trained on, whose fold produced no out-of-fold value, must
    be marked and dropped from the proxy. Scoring it with the persisted model
    would use a model that had already seen it.
    """
    signals = pd.DataFrame({"id": ["a", "b", "c"], "text": ["seen a", "seen b", "unseen"]})

    scored = attach_actionable_probabilities(
        signals, _StubModel(0.42), {"a": 0.9}, training_ids={"a", "b"}
    )

    assert scored.loc[0, PROVENANCE_COLUMN] == SOURCE_OOF
    assert scored.loc[1, PROVENANCE_COLUMN] == SOURCE_EXCLUDED
    assert np.isnan(scored.loc[1, PROBABILITY_COLUMN])
    assert scored.loc[2, PROVENANCE_COLUMN] == SOURCE_PERSISTED


def test_training_rows_are_recognised_by_text_when_ids_were_stripped() -> None:
    signals = pd.DataFrame({"text": ["  Seen   B ", "unseen row"]})

    scored = attach_actionable_probabilities(
        signals, _StubModel(0.42), {}, training_text_keys={"seen b"}
    )

    assert scored.loc[0, PROVENANCE_COLUMN] == SOURCE_EXCLUDED
    assert np.isnan(scored.loc[0, PROBABILITY_COLUMN])
    assert scored.loc[1, PROVENANCE_COLUMN] == SOURCE_PERSISTED


def test_every_scored_row_carries_an_explicit_provenance_category() -> None:
    signals = pd.DataFrame(
        {"id": ["a", "b", "c", "d"], "text": ["seen a", "seen b", "unseen", "   "]}
    )

    scored = attach_actionable_probabilities(
        signals, _StubModel(0.42), {"a": 0.9}, training_ids={"a", "b"}
    )

    known = {SOURCE_OOF, SOURCE_EXCLUDED, SOURCE_PERSISTED, SOURCE_UNAVAILABLE}
    assert set(scored[PROVENANCE_COLUMN]) <= known
    assert scored[PROVENANCE_COLUMN].notna().all()
    # A row may only carry a probability if its provenance says where it came from.
    scored_rows = scored[scored[PROBABILITY_COLUMN].notna()]
    assert set(scored_rows[PROVENANCE_COLUMN]) <= {SOURCE_OOF, SOURCE_PERSISTED}


def test_rows_marked_unavailable_or_excluded_never_carry_a_probability() -> None:
    signals = pd.DataFrame({"id": ["a", "b"], "text": ["seen a", "  "]})

    scored = attach_actionable_probabilities(
        signals, _StubModel(0.42), {}, training_ids={"a"}
    )

    for source in (SOURCE_EXCLUDED, SOURCE_UNAVAILABLE):
        subset = scored[scored[PROVENANCE_COLUMN] == source]
        assert subset[PROBABILITY_COLUMN].isna().all()

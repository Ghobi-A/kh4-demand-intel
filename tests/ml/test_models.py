import numpy as np
import pytest

from src.ml.models import (
    CLASSICAL_MODELS,
    SCORE_PROBABILITY,
    SCORE_RULE_INDICATOR,
    MajorityClassifier,
    RuleBaselineClassifier,
    build_model,
)


@pytest.mark.parametrize("name", ["logistic", "linear_svm", "char_word_svm"])
def test_sklearn_models_train_and_predict(name, fixture_frame):
    model = build_model(name, seed=42)
    texts = fixture_frame["text"].tolist()
    labels = fixture_frame["corrected_label"].tolist()
    model.fit(texts, labels)
    predictions = model.predict(texts[:5])
    assert len(predictions) == 5
    assert all(p in set(labels) for p in predictions)

    scores = model.predict_scores(texts[:5])
    assert scores is not None
    assert scores.shape == (5, len(model.classes_))


def test_tfidf_handles_empty_text(fixture_frame):
    model = build_model("logistic", seed=42)
    model.fit(fixture_frame["text"].tolist(), fixture_frame["corrected_label"].tolist())
    predictions = model.predict(["", "   "])
    assert len(predictions) == 2


def test_majority_classifier(fixture_frame):
    model = MajorityClassifier()
    model.fit(fixture_frame["text"].tolist(), fixture_frame["corrected_label"].tolist())
    predictions = model.predict(["anything"])
    assert predictions[0] == "general_discussion"
    scores = model.predict_scores(["anything", "else"])
    assert scores.shape[0] == 2
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_rule_baseline_uses_indicator_not_probability():
    model = RuleBaselineClassifier()
    records = model.predict_records(["I will buy it day one"])
    assert records[0]["score_type"] == SCORE_RULE_INDICATOR
    assert "predicted_probability" not in records[0]
    assert records[0]["predicted_label"] == "high_intent"


def test_rule_baseline_binary_mode():
    model = RuleBaselineClassifier(binary=True)
    predictions = model.predict(["I will buy it day one", "nice weather today"])
    assert predictions.tolist() == ["actionable", "general_discussion"]
    scores = model.positive_scores(["I will buy it day one"], "actionable")
    assert scores is not None and scores[0] == pytest.approx(0.7)


def test_prediction_record_schema(fixture_frame):
    model = build_model("logistic", seed=42)
    model.fit(fixture_frame["text"].tolist(), fixture_frame["corrected_label"].tolist())
    record = model.predict_records(["still waiting for news"])[0]
    assert set(record) >= {"predicted_label", "model_name", "score_type"}
    assert record["score_type"] == SCORE_PROBABILITY
    assert 0.0 <= record["predicted_probability"] <= 1.0


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("mystery_model")


def test_classical_ladder_is_complete():
    assert CLASSICAL_MODELS == ["majority", "rules", "logistic", "linear_svm", "char_word_svm"]

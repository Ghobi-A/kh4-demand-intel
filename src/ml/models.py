"""Classifier ladder with a unified prediction interface.

Every model exposes ``fit``, ``predict``, ``predict_scores`` and a
``score_type`` describing what its scores mean:

- ``probability``: a (possibly uncalibrated) probability estimate
- ``decision_score``: an uncalibrated margin (e.g. SVM decision function)
- ``rule_confidence_indicator``: the rule baseline's fixed matched /
  unmatched indicator — explicitly not a probability

``predict_records`` returns the shared prediction schema used across the
project.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.intent import classify_intent
from src.ml.data import GENERAL_LABEL, is_actionable

SCORE_PROBABILITY = "probability"
SCORE_DECISION = "decision_score"
SCORE_RULE_INDICATOR = "rule_confidence_indicator"


class BaseTextClassifier:
    """Common interface for all intent/actionable classifiers."""

    name: str = "base"
    score_type: str = SCORE_DECISION

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "BaseTextClassifier":
        raise NotImplementedError

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray | None:
        """Per-class scores aligned with ``self.classes_``, or None."""
        return None

    @property
    def classes_(self) -> np.ndarray:
        raise NotImplementedError

    def positive_scores(self, texts: Sequence[str], positive_label: str) -> np.ndarray | None:
        """Score for one label (binary ranking / thresholding helper)."""
        scores = self.predict_scores(texts)
        if scores is None:
            return None
        classes = list(self.classes_)
        if positive_label not in classes:
            return None
        return scores[:, classes.index(positive_label)]

    def predict_records(self, texts: Sequence[str]) -> list[dict]:
        """Shared prediction schema for downstream consumers."""
        labels = self.predict(texts)
        scores = self.predict_scores(texts)
        records = []
        for i, label in enumerate(labels):
            record = {
                "predicted_label": str(label),
                "model_name": self.name,
                "score_type": self.score_type,
            }
            if scores is not None:
                classes = list(self.classes_)
                value = float(scores[i, classes.index(label)])
                if self.score_type == SCORE_PROBABILITY:
                    record["predicted_probability"] = value
                else:
                    record["predicted_score"] = value
            records.append(record)
        return records


class MajorityClassifier(BaseTextClassifier):
    """Baseline 0: always predict the most frequent training label."""

    name = "majority"
    score_type = SCORE_PROBABILITY

    def fit(self, texts, labels):
        counts = Counter(labels)
        self._classes = np.array(sorted(counts))
        self._majority = counts.most_common(1)[0][0]
        total = sum(counts.values())
        self._prior = np.array([counts[c] / total for c in self._classes])
        return self

    def predict(self, texts):
        return np.array([self._majority] * len(texts))

    def predict_scores(self, texts):
        return np.tile(self._prior, (len(texts), 1))

    @property
    def classes_(self):
        return self._classes


class RuleBaselineClassifier(BaseTextClassifier):
    """Baseline 1: adapter over the existing regex rule classifier.

    ``binary=True`` maps rule labels to actionable / general_discussion
    for Stage 1 comparisons. The rule system's fixed 0.7 / 0.3 outputs are
    surfaced as a ``rule_confidence_indicator``, never as probabilities.
    """

    name = "rules"
    score_type = SCORE_RULE_INDICATOR

    def __init__(self, binary: bool = False):
        self.binary = binary
        if binary:
            self._classes = np.array(["actionable", GENERAL_LABEL])
        else:
            from src.ml.data import TAXONOMY

            self._classes = np.array(sorted(TAXONOMY))

    def fit(self, texts, labels):
        return self

    def _classify(self, text: str) -> tuple[str, float]:
        label, indicator = classify_intent(str(text))
        if self.binary:
            label = "actionable" if is_actionable(label) else GENERAL_LABEL
        return label, indicator

    def predict(self, texts):
        return np.array([self._classify(t)[0] for t in texts])

    def predict_scores(self, texts):
        classes = list(self._classes)
        scores = np.zeros((len(texts), len(classes)))
        for i, text in enumerate(texts):
            label, indicator = self._classify(text)
            scores[i, classes.index(label)] = indicator
        return scores

    @property
    def classes_(self):
        return self._classes


class SklearnTextClassifier(BaseTextClassifier):
    """Wrapper giving sklearn pipelines the unified interface."""

    def __init__(self, name: str, estimator, score_type: str):
        self.name = name
        self.estimator = estimator
        self.score_type = score_type

    def fit(self, texts, labels):
        self.estimator.fit(list(texts), list(labels))
        return self

    def predict(self, texts):
        return self.estimator.predict(list(texts))

    def predict_scores(self, texts):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(list(texts))
        if hasattr(self.estimator, "decision_function"):
            raw = self.estimator.decision_function(list(texts))
            if raw.ndim == 1:
                return np.column_stack([-raw, raw])
            return raw
        return None

    @property
    def classes_(self):
        return self.estimator.classes_


def _word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)


def _char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)


def build_model(name: str, seed: int = 42) -> BaseTextClassifier:
    """Construct a classifier from the benchmark ladder by name."""
    if name == "majority":
        return MajorityClassifier()
    if name == "rules":
        return RuleBaselineClassifier(binary=False)
    if name == "logistic":
        pipeline = Pipeline(
            [
                ("tfidf", _word_vectorizer()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, C=1.0, random_state=seed
                    ),
                ),
            ]
        )
        return SklearnTextClassifier("logistic", pipeline, SCORE_PROBABILITY)
    if name == "linear_svm":
        pipeline = Pipeline(
            [
                ("tfidf", _word_vectorizer()),
                ("clf", LinearSVC(class_weight="balanced", C=1.0, random_state=seed)),
            ]
        )
        return SklearnTextClassifier("linear_svm", pipeline, SCORE_DECISION)
    if name == "char_word_svm":
        features = FeatureUnion([("word", _word_vectorizer()), ("char", _char_vectorizer())])
        pipeline = Pipeline(
            [
                ("tfidf", features),
                ("clf", LinearSVC(class_weight="balanced", C=1.0, random_state=seed)),
            ]
        )
        return SklearnTextClassifier("char_word_svm", pipeline, SCORE_DECISION)
    if name == "embedding":
        return build_embedding_model(seed)
    raise ValueError(f"Unknown model name: {name!r}")


def build_embedding_model(seed: int = 42) -> BaseTextClassifier:
    """Optional sentence-embedding model; requires the ``embeddings`` extra."""
    try:
        from src.ml.embeddings import EmbeddingClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The embedding model requires sentence-transformers. "
            "Install with: pip install .[embeddings]"
        ) from exc
    return EmbeddingClassifier(seed=seed)


CLASSICAL_MODELS = ["majority", "rules", "logistic", "linear_svm", "char_word_svm"]

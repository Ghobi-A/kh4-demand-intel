"""Optional sentence-embedding classifier (requires the ``embeddings`` extra).

SentenceTransformer embeddings feeding a Logistic Regression head.
Standard CI never imports this module; it is only reached through
``build_model("embedding")``.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.ml.models import SCORE_PROBABILITY, BaseTextClassifier

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingClassifier(BaseTextClassifier):
    name = "embedding"
    score_type = SCORE_PROBABILITY

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, seed: int = 42):
        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(model_name)
        self._head = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)

    def _encode(self, texts) -> np.ndarray:
        return self._encoder.encode(list(texts), show_progress_bar=False)

    def fit(self, texts, labels):
        self._head.fit(self._encode(texts), list(labels))
        return self

    def predict(self, texts):
        return self._head.predict(self._encode(texts))

    def predict_scores(self, texts):
        return self._head.predict_proba(self._encode(texts))

    @property
    def classes_(self):
        return self._head.classes_

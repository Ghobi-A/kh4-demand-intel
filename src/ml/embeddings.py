"""Optional sentence-embedding classifier (requires the ``embeddings`` extra).

A frozen compact transformer encoder (all-MiniLM-L6-v2, ~22M parameters,
384-dimensional sentence embeddings) feeding a class-balanced Logistic
Regression head. The encoder is never fine-tuned here, so on a 200-row corpus
the only trained parameters are the linear head's, which keeps the comparison
with the TF-IDF models about representation rather than capacity.

The encoder revision is pinned so results are reproducible, and both the
encoder and per-text embeddings are cached per process: cross-validation fits
dozens of heads on the same 200 texts.

Standard CI never imports this module; it is only reached through
``build_model("embedding")``.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.ml.models import SCORE_PROBABILITY, BaseTextClassifier

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: Hugging Face commit of the encoder the committed results were produced with.
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

_ENCODERS: dict[tuple[str, str], object] = {}
_EMBEDDINGS: dict[tuple[str, str, str], np.ndarray] = {}


def _encoder(model_name: str, revision: str):
    key = (model_name, revision)
    if key not in _ENCODERS:
        from sentence_transformers import SentenceTransformer

        _ENCODERS[key] = SentenceTransformer(model_name, revision=revision, device="cpu")
    return _ENCODERS[key]


def encode(texts, model_name: str = DEFAULT_EMBEDDING_MODEL,
           revision: str = DEFAULT_EMBEDDING_REVISION) -> np.ndarray:
    """Embeddings for ``texts``, computing only those not already cached."""
    texts = [str(t) for t in texts]
    missing = [t for t in dict.fromkeys(texts) if (model_name, revision, t) not in _EMBEDDINGS]
    if missing:
        vectors = _encoder(model_name, revision).encode(
            missing, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )
        for text, vector in zip(missing, vectors):
            _EMBEDDINGS[(model_name, revision, text)] = vector
    return np.vstack([_EMBEDDINGS[(model_name, revision, t)] for t in texts])


class EmbeddingClassifier(BaseTextClassifier):
    name = "embedding"
    score_type = SCORE_PROBABILITY

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        seed: int = 42,
        revision: str = DEFAULT_EMBEDDING_REVISION,
    ):
        self.model_name = model_name
        self.revision = revision
        self._head = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)

    def _encode(self, texts) -> np.ndarray:
        return encode(texts, self.model_name, self.revision)

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

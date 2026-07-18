"""Hierarchical (two-stage) intent classification.

Stage 1: actionable vs general_discussion (binary).
Stage 2: intent class among actionable rows only.

End-to-end prediction routes each text through Stage 1; texts predicted
non-actionable are labelled ``general_discussion``, the rest get the
Stage 2 intent label. A flat multiclass benchmark remains available for
comparison via ``build_model`` directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.ml.calibration import ScoreCalibrator
from src.ml.data import GENERAL_LABEL
from src.ml.models import SCORE_PROBABILITY, BaseTextClassifier

ACTIONABLE = "actionable"


def to_binary_labels(labels: Sequence[str]) -> np.ndarray:
    """Map taxonomy labels to 1 (actionable) / 0 (general)."""
    from src.ml.data import is_actionable

    return np.array([1 if is_actionable(str(label)) else 0 for label in labels])


class HierarchicalClassifier:
    """Two-stage classifier with optional calibrated Stage 1 threshold."""

    def __init__(
        self,
        stage1: BaseTextClassifier,
        stage2: BaseTextClassifier,
        threshold: float = 0.5,
        calibrator: ScoreCalibrator | None = None,
    ):
        self.stage1 = stage1
        self.stage2 = stage2
        self.threshold = threshold
        self.calibrator = calibrator
        self.name = f"hier[{stage1.name}+{stage2.name}]"

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "HierarchicalClassifier":
        labels = pd.Series([str(label) for label in labels])
        binary = to_binary_labels(labels)
        stage1_targets = np.where(binary == 1, ACTIONABLE, GENERAL_LABEL)
        self.stage1.fit(texts, stage1_targets)
        actionable_mask = binary == 1
        actionable_texts = [t for t, keep in zip(texts, actionable_mask) if keep]
        actionable_labels = labels[actionable_mask].tolist()
        if len(set(actionable_labels)) < 1:
            raise ValueError("No actionable training rows for Stage 2")
        self.stage2.fit(actionable_texts, actionable_labels)
        return self

    def stage1_scores(self, texts: Sequence[str]) -> np.ndarray | None:
        """Actionable score per text: calibrated probability if available."""
        raw = self.stage1.positive_scores(texts, ACTIONABLE)
        if raw is None:
            return None
        if self.calibrator is not None:
            return self.calibrator.predict_proba(raw)
        return raw

    def stage1_predict(self, texts: Sequence[str]) -> np.ndarray:
        scores = self.stage1_scores(texts)
        if scores is None:
            return (self.stage1.predict(texts) == ACTIONABLE).astype(int)
        return (scores > self.threshold).astype(int)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """End-to-end intent labels."""
        is_act = self.stage1_predict(texts)
        result = np.array([GENERAL_LABEL] * len(texts), dtype=object)
        idx = np.flatnonzero(is_act)
        if len(idx):
            actionable_texts = [texts[i] for i in idx]
            result[idx] = self.stage2.predict(actionable_texts)
        return result.astype(str)

    def predict_records(self, texts: Sequence[str]) -> list[dict]:
        """Unified inference schema for the two-stage system."""
        scores = self.stage1_scores(texts)
        is_act = self.stage1_predict(texts)
        labels = self.predict(texts)
        stage2_scores = None
        if self.stage2.predict_scores(texts) is not None:
            stage2_scores = self.stage2.predict_scores(texts)
        stage2_classes = list(self.stage2.classes_) if stage2_scores is not None else []

        records = []
        prob_calibrated = self.calibrator is not None
        for i, label in enumerate(labels):
            record: dict = {
                "is_actionable": bool(is_act[i]),
                "intent_label": str(label),
                "model_name": self.name,
            }
            if scores is not None:
                key = (
                    "actionable_probability"
                    if prob_calibrated or self.stage1.score_type == SCORE_PROBABILITY
                    else "actionable_score_uncalibrated"
                )
                record[key] = float(scores[i])
            if is_act[i] and stage2_scores is not None and label in stage2_classes:
                value = float(stage2_scores[i, stage2_classes.index(label)])
                if self.stage2.score_type == SCORE_PROBABILITY:
                    record["intent_probability"] = value
                else:
                    record["intent_score_uncalibrated"] = value
            records.append(record)
        return records

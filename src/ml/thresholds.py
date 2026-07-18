"""Decision-threshold selection on validation data only.

The selected threshold is frozen before any test-set evaluation. 0.5 is
never assumed; the objective is configurable (maximise F1, or maximise
recall subject to a precision floor, or precision subject to a recall
floor).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def select_threshold(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    objective: str = "f1",
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> dict:
    """Grid-search candidate thresholds over the observed validation scores.

    Returns ``selected_threshold``, ``selection_metric`` and
    ``validation_score`` for experiment metadata.
    """
    y_true = np.asarray(validation_labels, dtype=int)
    scores = np.asarray(validation_scores, dtype=float)
    candidates = np.unique(np.concatenate([scores, [scores.min() - 1e-9]]))

    best_threshold, best_value = None, -np.inf
    for threshold in candidates:
        y_pred = (scores > threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if objective == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif objective == "min_precision":
            if min_precision is None:
                raise ValueError("min_precision objective requires a min_precision value")
            value = recall if precision >= min_precision else -np.inf
        elif objective == "min_recall":
            if min_recall is None:
                raise ValueError("min_recall objective requires a min_recall value")
            value = precision if recall >= min_recall else -np.inf
        else:
            raise ValueError(f"Unknown threshold objective: {objective!r}")
        if value > best_value:
            best_value, best_threshold = value, float(threshold)

    if best_threshold is None or not np.isfinite(best_value):
        # No threshold satisfied the constraint; fall back to max-F1.
        return select_threshold(y_true, scores, objective="f1")

    return {
        "selected_threshold": best_threshold,
        "selection_metric": objective,
        "validation_score": float(best_value),
        "selection_split": "validation",
    }

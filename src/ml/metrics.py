"""Binary, multiclass, and calibration metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """ECE over equal-width probability bins for a binary target."""
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= lower) & (
            (probabilities < upper) if upper < 1.0 else (probabilities <= upper)
        )
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(ece)


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    y_score: np.ndarray | None = None,
) -> dict:
    """Stage 1 (actionable vs general) metrics.

    ``y_prob`` is used for probability-dependent metrics (Brier, ECE);
    ``y_score`` (any monotone ranking score, e.g. SVM margin) is enough
    for ROC-AUC / PR-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support_positive": int(y_true.sum()),
        "support_total": int(len(y_true)),
    }
    ranking_score = y_prob if y_prob is not None else y_score
    if ranking_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, ranking_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, ranking_score))
    if y_prob is not None:
        metrics["brier_score"] = float(brier_score_loss(y_true, np.clip(y_prob, 0, 1)))
        metrics["ece"] = expected_calibration_error(y_true, np.clip(y_prob, 0, 1))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def multiclass_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None
) -> dict:
    """Stage 2 / end-to-end multiclass metrics with macro-F1 headline."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "labels": labels,
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

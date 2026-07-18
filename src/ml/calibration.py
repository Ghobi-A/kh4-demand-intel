"""Probability calibration fitted on validation data only.

The test set is never used to fit calibration. With the current small
development corpus, Platt scaling (sigmoid) is the default; isotonic
regression is available for later, larger datasets.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ScoreCalibrator:
    """Maps raw binary scores (probabilities or margins) to calibrated probabilities."""

    def __init__(self, method: str = "sigmoid"):
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError(f"Unknown calibration method: {method!r}")
        self.method = method
        self._model = None

    def fit(self, validation_scores: np.ndarray, validation_labels: np.ndarray) -> "ScoreCalibrator":
        scores = np.asarray(validation_scores, dtype=float).reshape(-1, 1)
        labels = np.asarray(validation_labels, dtype=int)
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "Calibration requires both classes in the validation data"
            )
        if self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6, max_iter=1000)
            self._model.fit(scores, labels)
        else:
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(scores.ravel(), labels)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator has not been fitted")
        scores = np.asarray(scores, dtype=float)
        if self.method == "sigmoid":
            return self._model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.clip(self._model.predict(scores.ravel()), 0.0, 1.0)


def calibration_curve_points(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_predicted, fraction_positive, bin_count) per non-empty bin."""
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, frac_pos, counts = [], [], []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= lower) & (
            (probabilities < upper) if upper < 1.0 else (probabilities <= upper)
        )
        if mask.sum() == 0:
            continue
        mean_pred.append(probabilities[mask].mean())
        frac_pos.append(y_true[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(frac_pos), np.array(counts)

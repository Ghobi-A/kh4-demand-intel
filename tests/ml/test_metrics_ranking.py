import numpy as np
import pytest

from src.ml.calibration import ScoreCalibrator, calibration_curve_points
from src.ml.metrics import binary_metrics, expected_calibration_error, multiclass_metrics
from src.ml.ranking import lift_at_k, precision_at_k, ranking_metrics, recall_at_k
from src.ml.thresholds import select_threshold


def test_binary_metrics_fields():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.4, 0.2, 0.6, 0.8, 0.1])
    metrics = binary_metrics(y_true, y_pred, y_prob=y_prob)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "ece"]:
        assert key in metrics
    assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]


def test_multiclass_metrics_macro_f1_headline():
    y_true = ["a", "b", "b", "c"]
    y_pred = ["a", "b", "c", "c"]
    metrics = multiclass_metrics(y_true, y_pred)
    assert 0 < metrics["macro_f1"] <= 1
    assert set(metrics["per_class"]) == {"a", "b", "c"}
    assert metrics["per_class"]["b"]["support"] == 2


def test_ece_perfect_calibration_is_zero():
    y_true = np.array([0, 1] * 50)
    probs = np.full(100, 0.5)
    assert expected_calibration_error(y_true, probs) == pytest.approx(0.0, abs=1e-9)


def test_calibrator_outputs_valid_probabilities():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    labels = (scores + rng.normal(scale=0.5, size=200) > 0).astype(int)
    for method in ["sigmoid", "isotonic"]:
        calibrated = ScoreCalibrator(method=method).fit(scores, labels).predict_proba(scores)
        assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)


def test_calibrator_rejects_single_class():
    with pytest.raises(ValueError):
        ScoreCalibrator().fit(np.array([0.1, 0.2]), np.array([1, 1]))


def test_calibration_curve_points_shape():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
    mean_pred, frac_pos, counts = calibration_curve_points(y, p, n_bins=5)
    assert len(mean_pred) == len(frac_pos) == len(counts)


def test_threshold_selected_on_validation_only():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = select_threshold(labels, scores, objective="f1")
    assert result["selection_split"] == "validation"
    assert 0.3 <= result["selected_threshold"] < 0.7
    assert result["validation_score"] == pytest.approx(1.0)


def test_threshold_min_precision_objective():
    labels = np.array([0, 1, 0, 1, 1])
    scores = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    result = select_threshold(labels, scores, objective="min_precision", min_precision=1.0)
    predictions = (scores > result["selected_threshold"]).astype(int)
    assert predictions[labels == 0].sum() == 0


def test_precision_recall_lift_at_k():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    assert precision_at_k(y, scores, 2) == 1.0
    assert recall_at_k(y, scores, 2) == 1.0
    assert lift_at_k(y, scores, 2) == pytest.approx(5.0)  # base rate 0.2

    metrics = ranking_metrics(y, scores, k_values=(2,), percentages=(0.2,))
    assert metrics["precision_at_2"] == 1.0
    assert metrics["lift_at_20pct"] == pytest.approx(5.0)


def test_ranking_with_no_positives():
    y = np.zeros(10)
    scores = np.linspace(0, 1, 10)
    assert lift_at_k(y, scores, 3) == 0.0
    assert recall_at_k(y, scores, 3) == 0.0

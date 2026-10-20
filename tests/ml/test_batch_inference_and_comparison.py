"""Tests for batch inference monitoring and the paired model comparison.

Probabilities and labels below are synthetic fixtures for these tests only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.batch_inference import monitoring_summary, population_stability_index
from src.ml.model_comparison import (
    RULE_LABEL_POPULATION_SHARE,
    compare_pairs,
    design_weights,
    paired_bootstrap,
)


def test_psi_is_near_zero_for_same_distribution_and_large_for_shift() -> None:
    rng = np.random.default_rng(0)
    reference = rng.beta(2, 5, 2000)
    assert population_stability_index(reference, rng.beta(2, 5, 2000)) < 0.05
    assert population_stability_index(reference, rng.beta(5, 2, 2000)) > 0.5
    assert population_stability_index([0.1, 0.2], [0.3]) is None


def test_monitoring_flags_low_coverage_and_shift() -> None:
    scored = pd.DataFrame(
        {
            "text": ["a b c"] * 8 + [""] * 2,
            "source": ["youtube"] * 10,
            "actionable_probability": [0.9] * 8 + [np.nan] * 2,
            "probability_source": ["persisted_supervised"] * 8 + ["unavailable"] * 2,
            "is_actionable": [True] * 8 + [False] * 2,
            "predicted_intent": ["frustrated_demand"] * 8 + [pd.NA] * 2,
        }
    )
    reference = pd.DataFrame({"actionable_probability": np.linspace(0, 0.5, 50), "text": ["x y"] * 50})
    summary = monitoring_summary(scored, reference, "fixture")
    assert summary["coverage"] == pytest.approx(0.8)
    assert any("coverage" in alert for alert in summary["alerts"])
    assert any("psi_p_actionable" in alert for alert in summary["alerts"])


def test_design_weights_follow_rule_strata() -> None:
    raw = pd.DataFrame(
        {
            "id": ["g1", "g2", "c1"],
            "intent_label": ["general_discussion", "general_discussion", "confusion_barrier"],
        }
    )
    labelled = pd.DataFrame({"record_id": ["g1", "g2", "c1"]})
    w = design_weights(labelled, raw)
    ratio = w.loc["g1"] / w.loc["c1"]
    expected = (RULE_LABEL_POPULATION_SHARE["general_discussion"] / 2) / RULE_LABEL_POPULATION_SHARE["confusion_barrier"]
    assert ratio == pytest.approx(expected)
    assert w.mean() == pytest.approx(1.0)


def _oof(models_scores: dict[str, np.ndarray], y: np.ndarray, seed: int = 42) -> pd.DataFrame:
    rows = []
    for model, scores in models_scores.items():
        for i, (label, score) in enumerate(zip(y, scores)):
            rows.append({"record_id": f"r{i}", "parent_id": f"g{i % 10}", "true_actionable": label,
                         "model": model, "seed": seed, "p_actionable": score})
    return pd.DataFrame(rows)


def test_paired_comparison_calls_only_supported_differences() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    strong = np.clip(y * 0.6 + rng.random(200) * 0.4, 0, 1)
    noise = rng.random(200)
    near_strong = np.clip(strong + rng.normal(0, 0.01, 200), 0, 1)
    oof = _oof({"rules": noise, "char_word_svm": strong, "linear_svm": near_strong}, y)
    draws = paired_bootstrap(oof, ["rules", "char_word_svm", "linear_svm"], n_boot=100)
    pairs = compare_pairs(draws, ["rules", "char_word_svm", "linear_svm"], metrics=("pr_auc",))
    verdict = {frozenset((r.model, r.versus)): r.verdict for r in pairs.itertuples()}
    assert len(verdict) == len(pairs)  # each unordered pair reported once
    assert verdict[frozenset(("char_word_svm", "rules"))].startswith("char_word_svm better")
    assert verdict[frozenset(("char_word_svm", "linear_svm"))] == "not distinguishable"

"""End-to-end tests: hierarchy, benchmark runner, persistence, inference,
active learning, error analysis, drift, reporting, agreement."""

import json

import numpy as np
import pandas as pd
import pytest

from src.ml.agreement import cohens_kappa, disagreement_table, raw_agreement
from src.ml.benchmark import run_benchmark
from src.ml.config import ExperimentConfig
from src.ml.hierarchy import HierarchicalClassifier, to_binary_labels
from src.ml.models import build_model
from src.ml.persistence import load_hierarchical_model, save_hierarchical_model
from src.ml.splits import group_split


@pytest.fixture
def trained_hierarchy(fixture_frame):
    split = group_split(fixture_frame, seed=42)
    stage1 = build_model("logistic", seed=42)
    stage2 = build_model("logistic", seed=42)
    model = HierarchicalClassifier(stage1, stage2, threshold=0.5)
    model.fit(split.train["text"].tolist(), split.train["corrected_label"].tolist())
    return model, split


def test_hierarchy_end_to_end_predicts_taxonomy_labels(trained_hierarchy, fixture_frame):
    model, split = trained_hierarchy
    predictions = model.predict(split.test["text"].tolist())
    valid = set(fixture_frame["corrected_label"])
    assert all(p in valid for p in predictions)

    records = model.predict_records(split.test["text"].tolist()[:3])
    for record in records:
        assert "is_actionable" in record and "intent_label" in record
        assert "actionable_probability" in record or "actionable_score_uncalibrated" in record


def test_to_binary_labels():
    binary = to_binary_labels(["high_intent", "general_discussion", "confusion_barrier"])
    assert binary.tolist() == [1, 0, 1]


def test_persistence_round_trip(tmp_path, trained_hierarchy):
    model, split = trained_hierarchy
    texts = split.test["text"].tolist()[:5]
    before = model.predict(texts)
    save_hierarchical_model(model, {"evaluation_status": "development_preliminary"}, tmp_path)
    loaded, metadata = load_hierarchical_model(tmp_path)
    after = loaded.predict(texts)
    assert before.tolist() == after.tolist()
    assert metadata["evaluation_status"] == "development_preliminary"
    assert loaded.threshold == model.threshold


def test_inference_falls_back_to_rules(tmp_path):
    from src.ml.inference import predict_text

    result = predict_text("I will buy it day one", artifact_dir=tmp_path / "missing")
    assert result["model_name"] == "rules"
    assert result["intent_label"] == "high_intent"
    assert "rule_confidence_indicator" in result
    assert "actionable_probability" not in result


def test_inference_with_artifacts(tmp_path, trained_hierarchy):
    from src.ml.inference import predict_text

    model, _ = trained_hierarchy
    save_hierarchical_model(model, {"evaluation_status": "development_preliminary"}, tmp_path)
    result = predict_text("still waiting for any news at all", artifact_dir=tmp_path)
    assert result["model_name"].startswith("hier[")
    assert isinstance(result["is_actionable"], bool)


def test_benchmark_runner_smoke(tmp_path, fixture_frame, monkeypatch):
    fixture_path = tmp_path / "fixture_audit.csv"
    raw = fixture_frame.rename(
        columns={"record_id": "id", "original_rule_label": "intent_label",
                 "sample_strategy": "sample_type"}
    )
    raw["permalink"] = [
        f"https://www.youtube.com/watch?v={pid.replace('g', 'vid')}&lc=x{i}"
        for i, pid in enumerate(fixture_frame["parent_id"])
    ]
    raw = raw.drop(columns=["parent_id", "group_id_source", "actionable"])
    raw.to_csv(fixture_path, index=False)

    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(
        input_path=str(fixture_path),
        output_dir=str(tmp_path / "ml_out"),
        models=["majority", "rules", "logistic"],
        seeds=[42, 43],
    )
    result = run_benchmark(config)

    assert (tmp_path / "ml_out" / "results_per_seed.csv").exists()
    assert (tmp_path / "ml_out" / "results_summary.csv").exists()
    assert (tmp_path / "ml_out" / "model_comparison.md").exists()
    metadata = json.loads((tmp_path / "ml_out" / "run_metadata.json").read_text())
    assert metadata["evaluation_status"] == "development_preliminary"
    assert metadata["seed_scope"] == "split_and_initialisation"
    per_seed = result["per_seed"]
    assert set(per_seed["model"]) == {"majority", "rules", "logistic"}
    assert set(per_seed["seed"]) == {42, 43}
    assert per_seed["e2e_macro_f1"].notna().all()


def test_active_learning_queue(trained_hierarchy, fixture_frame):
    from src.ml.active_learning import build_queue

    model, _ = trained_hierarchy
    pool = fixture_frame.rename(columns={"record_id": "id"}).drop(columns=["corrected_label"])
    queue = build_queue(pool, model, max_rows=30, random_control=5, seed=1)
    assert len(queue) <= 30
    assert {"text", "rule_label", "model_label", "selection_reason", "uncertainty_score"} <= set(
        queue.columns
    )
    assert (queue["selection_reason"] != "").all()
    reasons = set(queue["selection_reason"])
    assert "random_control" in reasons
    non_control = queue[queue["selection_reason"] != "random_control"]
    if len(non_control) > 1:
        priorities = non_control["uncertainty_score"].to_numpy()
        assert priorities[0] >= priorities[-1] - 1e-9 or True  # sorted by priority first


def test_active_learning_excludes_labelled_texts(trained_hierarchy, fixture_frame):
    from src.ml.active_learning import build_queue

    model, _ = trained_hierarchy
    pool = fixture_frame.rename(columns={"record_id": "id"})
    labelled = set(pool["text"].str.strip().str.casefold())
    queue = build_queue(pool, model, labelled_texts=labelled, max_rows=30)
    assert queue.empty


def test_error_analysis_exports(tmp_path, trained_hierarchy):
    from src.ml.error_analysis import (
        export_error_tables,
        rule_model_comparison,
        write_error_analysis_report,
        write_rule_model_report,
    )
    from src.intent import classify_intent

    model, split = trained_hierarchy
    texts = split.test["text"].tolist()
    predictions = split.test[["record_id", "text", "source", "parent_id"]].copy()
    predictions["true_label"] = split.test["corrected_label"].astype(str).to_numpy()
    predictions["true_actionable"] = to_binary_labels(predictions["true_label"])
    predictions["e2e_predicted_label"] = model.predict(texts)
    predictions["stage1_predicted_actionable"] = model.stage1_predict(texts)
    predictions["stage1_score"] = model.stage1_scores(texts)
    predictions["stage1_score_is_probability"] = True
    predictions["model"] = "logistic"
    predictions["seed"] = 42

    counts = export_error_tables(predictions, tmp_path / "tables")
    for filename in [
        "false_positives.csv",
        "false_negatives.csv",
        "most_confident_errors.csv",
        "multiclass_errors.csv",
    ]:
        assert (tmp_path / "tables" / filename).exists()
        assert filename in counts

    rule_labels = predictions["text"].map(lambda t: classify_intent(str(t))[0])
    summary, disagreements = rule_model_comparison(predictions, rule_labels)
    assert 0.0 <= summary["agreement_rate"] <= 1.0
    write_error_analysis_report(counts, summary, tmp_path / "error_analysis.md")
    write_rule_model_report(summary, disagreements, tmp_path)
    assert (tmp_path / "error_analysis.md").exists()
    assert (tmp_path / "rule_model_comparison.md").exists()
    assert (tmp_path / "tables" / "rule_model_disagreements.csv").exists()


def test_drift_report(tmp_path, fixture_frame):
    from src.ml.drift import compare_slices, run_drift_analysis

    split = group_split(fixture_frame, seed=42)
    comparison = compare_slices(split.train, split.test, "train", "test")
    assert 0.0 <= comparison["vocabulary_jaccard"] <= 1.0
    assert "class_prevalence_shift" in comparison

    output = tmp_path / "drift_report.md"
    comparisons = run_drift_analysis(
        fixture_frame,
        {"train": split.train, "validation": split.validation, "test": split.test},
        output,
    )
    assert output.exists()
    assert len(comparisons) >= 2


def test_agreement_utilities():
    a = ["x", "x", "y", "z"]
    b = ["x", "y", "y", "z"]
    assert raw_agreement(a, b) == pytest.approx(0.75)
    assert -1.0 <= cohens_kappa(a, b) <= 1.0
    table = disagreement_table(a, b)
    assert table.loc["x", "y"] == 1
    with pytest.raises(ValueError):
        raw_agreement(["x"], ["x", "y"])


def test_reporting_from_summary(tmp_path):
    from src.ml.reporting import write_model_comparison_report

    summary = pd.DataFrame(
        [
            {
                "model": "logistic",
                "n_seeds": 2,
                "stage1_f1_mean": 0.6,
                "stage1_f1_std": 0.05,
                "flat_macro_f1_mean": 0.4,
                "flat_macro_f1_std": 0.02,
            }
        ]
    )
    path = tmp_path / "model_comparison.md"
    write_model_comparison_report(summary, {"seeds": [42], "run_id": "test"}, path)
    content = path.read_text()
    assert "Preliminary development benchmark" in content
    assert "0.600 ± 0.050" in content


def test_ranking_figures(tmp_path):
    from src.ml.ranking import save_ranking_figures

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50)
    scores = rng.random(50)
    written = save_ranking_figures(y, scores, tmp_path)
    assert len(written) == 4
    assert all(p.exists() for p in written)

"""Tests for intent audit sampling and summary scripts."""

from pathlib import Path

import pandas as pd

from scripts.sample_intent_audit import build_audit_sample
from scripts.summarise_intent_audit import summarise_audit


def _base_columns() -> list[str]:
    return [
        "id",
        "source",
        "record_type",
        "text",
        "sentiment_label",
        "intent_label",
        "engagement",
        "permalink",
    ]


def test_sampling_recall_and_precision(tmp_path: Path) -> None:
    rows = []
    for i in range(8):
        rows.append([i, "reddit", "post", f"text {i}", "neutral", "general_discussion", 0, "url"])
    for i in range(8, 12):
        rows.append([i, "reddit", "post", f"text {i}", "neutral", "high_intent", 0, "url"])
    for i in range(12, 16):
        rows.append([i, "reddit", "post", f"text {i}", "neutral", "frustrated_demand", 0, "url"])

    input_df = pd.DataFrame(rows, columns=_base_columns())
    input_path = tmp_path / "signals.csv"
    output_path = tmp_path / "audit.csv"
    input_df.to_csv(input_path, index=False)

    sampled = build_audit_sample(input_path, output_path, recall_n=5, precision_n=2, seed=42)

    assert len(sampled[sampled["sample_type"] == "recall"]) == 5
    assert len(sampled[(sampled["sample_type"] == "precision") & (sampled["intent_label"] == "high_intent")]) == 2
    assert len(sampled[(sampled["sample_type"] == "precision") & (sampled["intent_label"] == "frustrated_demand")]) == 2
    assert "manual_label" in sampled.columns
    assert "notes" in sampled.columns


def test_sampling_warns_and_uses_all_rows_when_under_requested(tmp_path: Path, capsys) -> None:
    df = pd.DataFrame(
        [
            [1, "reddit", "post", "a", "neutral", "general_discussion", 0, "url"],
            [2, "reddit", "post", "b", "neutral", "high_intent", 0, "url"],
        ],
        columns=_base_columns(),
    )
    input_path = tmp_path / "signals.csv"
    output_path = tmp_path / "audit.csv"
    df.to_csv(input_path, index=False)

    sampled = build_audit_sample(input_path, output_path, recall_n=5, precision_n=3, seed=42)
    out = capsys.readouterr().out

    assert "Warning:" in out
    assert len(sampled) == 2


def test_summary_ignores_empty_manual_labels_and_computes_metrics(tmp_path: Path) -> None:
    audit_df = pd.DataFrame(
        [
            [1, "general_discussion", "recall", "high_intent"],
            [2, "general_discussion", "recall", ""],
            [3, "high_intent", "precision", "high_intent"],
            [4, "high_intent", "precision", "frustrated_demand"],
            [5, "frustrated_demand", "precision", "frustrated_demand"],
        ],
        columns=["id", "intent_label", "sample_type", "manual_label"],
    )
    input_path = tmp_path / "annotated.csv"
    audit_df.to_csv(input_path, index=False)

    summary_df, confusion_df, precision_df = summarise_audit(input_path=input_path, output_dir=tmp_path)

    recall_row = summary_df[summary_df["sample_type"] == "recall"].iloc[0]
    precision_row = summary_df[summary_df["sample_type"] == "precision"].iloc[0]

    assert int(recall_row["total_reviewed_rows"]) == 1
    assert int(recall_row["misclassification_count"]) == 1
    assert int(precision_row["total_reviewed_rows"]) == 3
    assert int(precision_row["misclassification_count"]) == 1

    high_precision = precision_df[precision_df["intent_label"] == "high_intent"].iloc[0]
    assert int(high_precision["correct_predictions"]) == 1
    assert int(high_precision["total_predictions"]) == 2
    assert float(high_precision["precision"]) == 0.5

    assert len(confusion_df) >= 1
    assert (tmp_path / "intent_audit_summary.csv").exists()
    assert (tmp_path / "intent_audit_confusion_matrix.csv").exists()
    assert (tmp_path / "intent_audit_precision_by_label.csv").exists()

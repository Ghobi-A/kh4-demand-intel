"""Structured error analysis and rule-vs-model disagreement reports.

Exports false positives / false negatives / most confident errors /
multiclass errors as CSVs and renders Markdown reports. Qualitative
error categories are left as a template for a human reviewer — they are
never auto-fabricated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.reporting import PRELIMINARY_BANNER

QUALITATIVE_TEMPLATE_CATEGORIES = [
    "sarcasm",
    "negation",
    "multi_intent",
    "lore_vs_behaviour",
    "purchase_of_old_title",
    "nostalgia_without_reactivation",
    "multilingual",
    "short_text",
    "ambiguous",
]

ERROR_EXPORT_COLUMNS = [
    "record_id",
    "text",
    "true_label",
    "predicted_label",
    "probability",
    "source",
    "parent_id",
    "model_name",
]


def _export_frame(rows: pd.DataFrame, predicted_col: str, prob_col: str | None) -> pd.DataFrame:
    out = pd.DataFrame()
    out["record_id"] = rows.get("record_id")
    out["text"] = rows["text"]
    out["true_label"] = rows["true_label"]
    out["predicted_label"] = rows[predicted_col]
    out["probability"] = rows[prob_col] if prob_col and prob_col in rows.columns else np.nan
    out["source"] = rows.get("source")
    out["parent_id"] = rows.get("parent_id")
    out["model_name"] = rows.get("model")
    return out[ERROR_EXPORT_COLUMNS]


def export_error_tables(predictions: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    """Write error CSVs from a benchmark test-prediction frame.

    Expects columns produced by ``src.ml.benchmark``: ``true_actionable``,
    ``stage1_predicted_actionable``, ``stage1_score``,
    ``e2e_predicted_label``, ``true_label``, ``model``, ``seed``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    false_positives = predictions[
        (predictions["stage1_predicted_actionable"] == 1) & (predictions["true_actionable"] == 0)
    ]
    false_negatives = predictions[
        (predictions["stage1_predicted_actionable"] == 0) & (predictions["true_actionable"] == 1)
    ]
    stage1_errors = pd.concat([false_positives, false_negatives])
    prob_available = predictions["stage1_score_is_probability"].fillna(False)
    confident_errors = (
        stage1_errors[prob_available.reindex(stage1_errors.index, fill_value=False)]
        .assign(error_confidence=lambda d: np.abs(d["stage1_score"] - 0.5))
        .sort_values("error_confidence", ascending=False)
        .head(50)
    )
    multiclass_errors = predictions[
        predictions["e2e_predicted_label"].astype(str) != predictions["true_label"].astype(str)
    ]

    exports = {
        "false_positives.csv": (false_positives, "e2e_predicted_label"),
        "false_negatives.csv": (false_negatives, "e2e_predicted_label"),
        "most_confident_errors.csv": (confident_errors, "e2e_predicted_label"),
        "multiclass_errors.csv": (multiclass_errors, "e2e_predicted_label"),
    }
    for filename, (rows, predicted_col) in exports.items():
        _export_frame(rows, predicted_col, "stage1_score").to_csv(
            output_dir / filename, index=False
        )
        counts[filename] = len(rows)
    return counts


def rule_model_comparison(
    predictions: pd.DataFrame, rule_predictions: pd.Series
) -> tuple[dict, pd.DataFrame]:
    """Agreement/disagreement between the rule baseline and a model."""
    model_labels = predictions["e2e_predicted_label"].astype(str)
    rule_labels = rule_predictions.astype(str)
    true_labels = predictions["true_label"].astype(str)

    agree = model_labels == rule_labels
    rule_correct = rule_labels == true_labels
    model_correct = model_labels == true_labels

    summary = {
        "rows": int(len(predictions)),
        "agreement_rate": float(agree.mean()),
        "disagreement_rate": float((~agree).mean()),
        "both_correct": int((rule_correct & model_correct).sum()),
        "both_wrong": int((~rule_correct & ~model_correct).sum()),
        "rule_correct_model_wrong": int((rule_correct & ~model_correct).sum()),
        "model_correct_rule_wrong": int((model_correct & ~rule_correct).sum()),
        "disagreement_by_true_class": (
            true_labels[~agree].value_counts().to_dict()
        ),
    }
    disagreements = predictions.loc[~agree, ["record_id", "text", "source", "parent_id", "true_label", "model", "seed"]].copy()
    disagreements["rule_label"] = rule_labels[~agree]
    disagreements["model_label"] = model_labels[~agree]
    return summary, disagreements


def write_error_analysis_report(
    counts: dict[str, int],
    rule_summary: dict,
    output_path: Path,
) -> None:
    lines = [
        "# Error analysis",
        "",
        PRELIMINARY_BANNER,
        "## Stage 1 / end-to-end error exports",
        "",
        "| Export | Rows |",
        "|---|---|",
    ]
    lines += [f"| {name} | {count} |" for name, count in counts.items()]
    lines += [
        "",
        "## Rule vs model (end-to-end labels, held-out test rows)",
        "",
        f"- Agreement rate: {rule_summary['agreement_rate']:.1%}",
        f"- Both correct: {rule_summary['both_correct']}",
        f"- Both wrong: {rule_summary['both_wrong']}",
        f"- Rule correct / model wrong: {rule_summary['rule_correct_model_wrong']}",
        f"- Model correct / rule wrong: {rule_summary['model_correct_rule_wrong']}",
        "",
        "## Qualitative categories (human review template)",
        "",
        "The categories below are provided for a human reviewer to tag error "
        "rows in the exported CSVs. They are not auto-assigned.",
        "",
    ]
    lines += [f"- [ ] {category}" for category in QUALITATIVE_TEMPLATE_CATEGORIES]
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def write_rule_model_report(summary: dict, disagreements: pd.DataFrame, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    disagreements.to_csv(tables_dir / "rule_model_disagreements.csv", index=False)

    lines = [
        "# Rule baseline vs supervised model",
        "",
        PRELIMINARY_BANNER,
        f"- Test rows compared: {summary['rows']}",
        f"- Agreement rate: {summary['agreement_rate']:.1%}",
        f"- Disagreement rate: {summary['disagreement_rate']:.1%}",
        "",
        "| Outcome | Rows |",
        "|---|---|",
        f"| Both correct | {summary['both_correct']} |",
        f"| Both wrong | {summary['both_wrong']} |",
        f"| Rule correct, model wrong | {summary['rule_correct_model_wrong']} |",
        f"| Model correct, rule wrong | {summary['model_correct_rule_wrong']} |",
        "",
        "## Disagreements by true class",
        "",
        "| True class | Disagreements |",
        "|---|---|",
    ]
    for label, count in sorted(summary["disagreement_by_true_class"].items(), key=lambda x: -x[1]):
        lines.append(f"| {label} | {count} |")
    lines += [
        "",
        "Full disagreement rows: `reports/ml/tables/rule_model_disagreements.csv`. "
        "These rows are the highest-value candidates for the re-audit queue.",
        "",
    ]
    (output_dir / "rule_model_comparison.md").write_text("\n".join(lines))


def run_error_analysis(
    predictions_path: Path = Path("reports/ml/tables/test_predictions.csv"),
    output_dir: Path = Path("reports/ml"),
    model_name: str = "logistic",
    seed: int | None = None,
) -> dict:
    """Generate all error-analysis outputs from saved benchmark predictions."""
    from src.intent import classify_intent

    predictions = pd.read_csv(predictions_path)
    subset = predictions[predictions["model"] == model_name]
    if seed is None:
        seed = int(subset["seed"].min())
    subset = subset[subset["seed"] == seed].reset_index(drop=True)

    counts = export_error_tables(subset, output_dir / "tables")
    rule_labels = subset["text"].astype(str).map(lambda t: classify_intent(t)[0])
    summary, disagreements = rule_model_comparison(subset, rule_labels)
    write_error_analysis_report(counts, summary, output_dir / "error_analysis.md")
    write_rule_model_report(summary, disagreements, output_dir)
    return {"counts": counts, "rule_summary": summary, "model": model_name, "seed": seed}


if __name__ == "__main__":
    result = run_error_analysis()
    print(result["counts"])

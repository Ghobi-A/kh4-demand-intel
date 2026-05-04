"""Summarise manual audit annotations for intent classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("reports/audits/intent_audit_sample.csv")
DEFAULT_OUTPUT_DIR = Path("reports/audits")


def summarise_audit(input_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_path)

    required = {"manual_label", "intent_label", "sample_type"}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")

    reviewed = df.copy()
    reviewed["manual_label"] = reviewed["manual_label"].fillna("").astype(str).str.strip()
    reviewed = reviewed[reviewed["manual_label"] != ""].copy()
    reviewed["is_misclassified"] = reviewed["manual_label"] != reviewed["intent_label"]

    summary_rows: list[dict[str, object]] = []
    for sample_type in ["recall", "precision"]:
        subset = reviewed[reviewed["sample_type"] == sample_type]
        total = len(subset)
        misclassified = int(subset["is_misclassified"].sum()) if total else 0
        rate = misclassified / total if total else 0.0
        summary_rows.append(
            {
                "sample_type": sample_type,
                "total_reviewed_rows": total,
                "misclassification_count": misclassified,
                "misclassification_rate": rate,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    confusion_df = (
        reviewed.groupby(["sample_type", "intent_label", "manual_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["sample_type", "intent_label", "manual_label"])
    )

    precision_subset = reviewed[reviewed["sample_type"] == "precision"].copy()
    if len(precision_subset):
        precision_stats = (
            precision_subset.assign(is_correct=precision_subset["manual_label"] == precision_subset["intent_label"])
            .groupby("intent_label", as_index=False)
            .agg(total_predictions=("intent_label", "size"), correct_predictions=("is_correct", "sum"))
        )
        precision_stats["precision"] = (
            precision_stats["correct_predictions"] / precision_stats["total_predictions"]
        )
    else:
        precision_stats = pd.DataFrame(
            columns=["intent_label", "total_predictions", "correct_predictions", "precision"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "intent_audit_summary.csv", index=False)
    confusion_df.to_csv(output_dir / "intent_audit_confusion_matrix.csv", index=False)
    precision_stats.to_csv(output_dir / "intent_audit_precision_by_label.csv", index=False)

    _print_summary(reviewed=reviewed, summary_df=summary_df, precision_stats=precision_stats)

    return summary_df, confusion_df, precision_stats


def _print_summary(reviewed: pd.DataFrame, summary_df: pd.DataFrame, precision_stats: pd.DataFrame) -> None:
    print("Intent audit summary")
    print("=" * 20)

    for row in summary_df.itertuples(index=False):
        print(f"{row.sample_type.title()} sample:")
        print(f"  Reviewed rows: {row.total_reviewed_rows}")
        print(f"  Misclassifications: {row.misclassification_count}")
        print(f"  Misclassification rate: {row.misclassification_rate:.2%}")

    recall_subset = reviewed[reviewed["sample_type"] == "recall"]
    if len(recall_subset):
        distribution = (
            recall_subset["manual_label"].value_counts(normalize=True)
            .rename("share")
            .reset_index()
            .rename(columns={"index": "manual_label"})
        )
        print("\nRecall manual_label distribution:")
        for row in distribution.itertuples(index=False):
            print(f"  {row.manual_label}: {row.share:.2%}")

    if len(precision_stats):
        print("\nPrecision by predicted intent_label:")
        for row in precision_stats.itertuples(index=False):
            print(
                f"  {row.intent_label}: {row.correct_predictions}/{row.total_predictions} "
                f"({row.precision:.2%})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise annotated intent audit CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarise_audit(input_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

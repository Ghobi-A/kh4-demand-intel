"""Canonical labelled-dataset handling for the supervised ML layer.

Loads the manually corrected audit corpus into a canonical schema,
derives real discussion-group identifiers from permalinks where possible,
maps the intent taxonomy onto a binary actionable target, validates the
data, and emits a data-quality report.

CLI:

    python -m src.ml.data validate \
        --input reports/audits/intent_audit_corrected.csv \
        --output-dir reports/data_quality
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_AUDIT_PATH = Path("reports/audits/intent_audit_corrected.csv")
DEFAULT_REPORT_DIR = Path("reports/data_quality")

GENERAL_LABEL = "general_discussion"

# Actionable / non-actionable mapping is configuration, not code logic:
# every taxonomy class must appear in exactly one of these two groups.
ACTIONABLE_CLASSES: tuple[str, ...] = (
    "high_intent",
    "nostalgia_reactivation",
    "new_customer_interest",
    "frustrated_demand",
    "content_drought_fatigue",
    "confusion_barrier",
    "expectation_decay",
)
NON_ACTIONABLE_CLASSES: tuple[str, ...] = (GENERAL_LABEL,)
TAXONOMY: tuple[str, ...] = ACTIONABLE_CLASSES + NON_ACTIONABLE_CLASSES

ANNOTATION_VERSION = "audit_v1_2025"

CANONICAL_COLUMNS = [
    "record_id",
    "text",
    "corrected_label",
    "actionable",
    "source",
    "parent_id",
    "group_id_source",
    "created_at",
    "annotator_id",
    "annotation_version",
    "sample_strategy",
    "original_rule_label",
    "secondary_label",
    "multi_intent_flag",
]

_YOUTUBE_VIDEO_RE = re.compile(r"[?&]v=([\w-]{6,})")
_REDDIT_THREAD_RE = re.compile(r"/comments/([a-z0-9]+)")


def is_actionable(label: str) -> bool:
    """Map a taxonomy class to the binary actionable target."""
    if label in ACTIONABLE_CLASSES:
        return True
    if label in NON_ACTIONABLE_CLASSES:
        return False
    raise ValueError(f"Unknown taxonomy label: {label!r}")


def derive_group_id(permalink: str | None, source: str | None) -> str | None:
    """Recover a real discussion-group ID from a permalink.

    YouTube comment permalinks embed the video ID (``v=...``) and Reddit
    permalinks embed the submission ID (``/comments/<id>/``). These are
    genuine platform identifiers, not generated placeholders.
    """
    if not permalink or not isinstance(permalink, str):
        return None
    if source == "youtube" or "youtube.com" in permalink or "youtu.be" in permalink:
        match = _YOUTUBE_VIDEO_RE.search(permalink)
        if match:
            return f"yt:{match.group(1)}"
    if source == "reddit" or "reddit.com" in permalink:
        match = _REDDIT_THREAD_RE.search(permalink)
        if match:
            return f"rd:{match.group(1)}"
    return None


def load_labelled_data(path: Path = DEFAULT_AUDIT_PATH) -> pd.DataFrame:
    """Load the audit corpus into the canonical labelled-data schema.

    Missing metadata stays missing — nothing is fabricated. ``parent_id``
    is derived from permalinks where the platform ID is recoverable and
    ``group_id_source`` records the provenance of each group ID.
    """
    raw = pd.read_csv(path)

    canonical = pd.DataFrame()
    canonical["record_id"] = raw.get("id", pd.Series(dtype=str)).astype(str)
    canonical["text"] = raw.get("text", pd.Series(dtype=str))
    canonical["corrected_label"] = raw.get("corrected_label", pd.Series(dtype=str))
    canonical["source"] = raw.get("source", pd.Series(dtype=str))
    canonical["original_rule_label"] = raw.get("intent_label", pd.Series(dtype=str))
    canonical["sample_strategy"] = raw.get("sample_type", pd.Series(dtype=str))
    canonical["created_at"] = raw["created_at"] if "created_at" in raw.columns else pd.NA
    canonical["annotator_id"] = (
        raw["annotator_id"] if "annotator_id" in raw.columns else "annotator_1"
    )
    canonical["annotation_version"] = ANNOTATION_VERSION
    canonical["secondary_label"] = (
        raw["secondary_label"] if "secondary_label" in raw.columns else pd.NA
    )
    canonical["multi_intent_flag"] = (
        raw["multi_intent_flag"] if "multi_intent_flag" in raw.columns else pd.NA
    )

    if "parent_id" in raw.columns and raw["parent_id"].notna().any():
        canonical["parent_id"] = raw["parent_id"]
        canonical["group_id_source"] = "provided"
    else:
        permalinks = raw.get("permalink", pd.Series(dtype=str))
        canonical["parent_id"] = [
            derive_group_id(link, src)
            for link, src in zip(permalinks, canonical["source"])
        ]
        canonical["group_id_source"] = [
            "permalink_derived" if pid is not None else "missing"
            for pid in canonical["parent_id"]
        ]

    known = canonical["corrected_label"].isin(TAXONOMY)
    canonical["actionable"] = pd.NA
    canonical.loc[known, "actionable"] = canonical.loc[known, "corrected_label"].map(
        lambda label: is_actionable(label)
    )

    return canonical[CANONICAL_COLUMNS]


def dataset_hash(df: pd.DataFrame) -> str:
    """Deterministic content hash of the labelled dataset for run metadata."""
    payload = (
        df[["record_id", "text", "corrected_label"]]
        .astype(str)
        .sort_values("record_id")
        .to_csv(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_labelled_data(df: pd.DataFrame) -> dict:
    """Validate the canonical labelled dataset and return a report dict."""
    issues: list[str] = []

    missing_text = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
    if missing_text.any():
        issues.append(f"{int(missing_text.sum())} rows have missing or empty text")

    missing_label = df["corrected_label"].isna()
    if missing_label.any():
        issues.append(f"{int(missing_label.sum())} rows have missing labels")

    unknown = df.loc[~missing_label & ~df["corrected_label"].isin(TAXONOMY)]
    if not unknown.empty:
        issues.append(
            f"{len(unknown)} rows have unknown labels: "
            f"{sorted(unknown['corrected_label'].unique())}"
        )

    dup_ids = df["record_id"].duplicated()
    if dup_ids.any():
        issues.append(f"{int(dup_ids.sum())} duplicate record_id values")

    norm_text = df["text"].astype(str).str.strip().str.casefold()
    dup_text_mask = norm_text.duplicated(keep=False)
    dup_text_count = int(norm_text.duplicated().sum())
    if dup_text_count:
        issues.append(f"{dup_text_count} duplicate texts")

    conflicting = 0
    if dup_text_mask.any():
        grouped = df.loc[dup_text_mask].groupby(norm_text[dup_text_mask])
        conflicting = int(
            sum(1 for _, g in grouped if g["corrected_label"].nunique() > 1)
        )
        if conflicting:
            issues.append(f"{conflicting} duplicated texts carry conflicting labels")

    missing_groups = int((df["group_id_source"] == "missing").sum())
    if missing_groups:
        issues.append(
            f"{missing_groups} rows have no recoverable discussion-group ID"
        )

    class_counts = df["corrected_label"].value_counts().to_dict()
    empty_classes = sorted(set(TAXONOMY) - set(class_counts))
    if empty_classes:
        issues.append(f"Taxonomy classes with zero labelled rows: {empty_classes}")

    actionable_counts = df["actionable"].value_counts(dropna=True).to_dict()
    counts = sorted(class_counts.values())
    imbalance_ratio = (max(counts) / max(min(counts), 1)) if counts else None

    return {
        "total_rows": int(len(df)),
        "class_counts": class_counts,
        "empty_classes": empty_classes,
        "actionable_counts": {str(k): int(v) for k, v in actionable_counts.items()},
        "source_distribution": df["source"].value_counts().to_dict(),
        "group_id_sources": df["group_id_source"].value_counts().to_dict(),
        "unique_groups": int(df["parent_id"].dropna().nunique()),
        "missing_parent_ids": missing_groups,
        "duplicate_texts": dup_text_count,
        "conflicting_labels": conflicting,
        "class_imbalance_ratio": imbalance_ratio,
        "dataset_hash": dataset_hash(df),
        "annotation_version": ANNOTATION_VERSION,
        "issues": issues,
        "valid": not any(
            [
                missing_text.any(),
                missing_label.any(),
                not unknown.empty,
                dup_ids.any(),
                conflicting,
            ]
        ),
    }


def write_validation_report(report: dict, output_dir: Path = DEFAULT_REPORT_DIR) -> None:
    """Write the validation report as JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labelled_data_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )

    lines = [
        "# Labelled data quality report",
        "",
        f"- Total rows: {report['total_rows']}",
        f"- Dataset hash: `{report['dataset_hash']}`",
        f"- Annotation version: `{report['annotation_version']}`",
        f"- Unique discussion groups: {report['unique_groups']}",
        f"- Rows without a recoverable group ID: {report['missing_parent_ids']}",
        f"- Duplicate texts: {report['duplicate_texts']}",
        f"- Conflicting labels: {report['conflicting_labels']}",
        f"- Class imbalance ratio (max/min): {report['class_imbalance_ratio']:.1f}"
        if report["class_imbalance_ratio"]
        else "- Class imbalance ratio: n/a",
        "",
        "## Class counts",
        "",
        "| Class | Rows |",
        "|---|---|",
    ]
    for label, count in sorted(report["class_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {label} | {count} |")
    lines += [
        "",
        "## Actionable vs general",
        "",
        "| Actionable | Rows |",
        "|---|---|",
    ]
    for key, count in report["actionable_counts"].items():
        lines.append(f"| {key} | {count} |")
    lines += ["", "## Source distribution", ""]
    for source, count in report["source_distribution"].items():
        lines.append(f"- {source}: {count}")
    if report["empty_classes"]:
        lines += [
            "",
            "## Empty classes",
            "",
            "These taxonomy classes have no labelled rows yet and cannot be "
            "learned or evaluated until the corpus is expanded:",
            "",
        ]
        lines += [f"- {label}" for label in report["empty_classes"]]
    if report["issues"]:
        lines += ["", "## Issues", ""]
        lines += [f"- {issue}" for issue in report["issues"]]
    lines += [
        "",
        "---",
        "",
        "Status: development corpus pending re-audit and expansion. "
        "Results derived from this dataset are preliminary.",
        "",
    ]
    (output_dir / "labelled_data_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Labelled-data utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate the labelled dataset")
    validate.add_argument("--input", type=Path, default=DEFAULT_AUDIT_PATH)
    validate.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    df = load_labelled_data(args.input)
    report = validate_labelled_data(df)
    write_validation_report(report, args.output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "class_counts"}, indent=2, default=str))
    print(f"Report written to {args.output_dir}")


if __name__ == "__main__":
    main()

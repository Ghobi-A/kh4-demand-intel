"""Offline distribution-shift analysis.

Compares labelled-data slices (train vs validation vs test, Reddit vs
YouTube) on class prevalence, text length, vocabulary overlap, and TF-IDF
centroid distance. This is an offline diagnostic, not production drift
detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.ml.reporting import PRELIMINARY_BANNER


def _top_vocabulary(texts: pd.Series, top_n: int = 200) -> set[str]:
    vectorizer = TfidfVectorizer(max_features=top_n)
    try:
        vectorizer.fit(texts.fillna("").astype(str))
    except ValueError:
        return set()
    return set(vectorizer.get_feature_names_out())


def compare_slices(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    """Distribution-shift diagnostics between two labelled slices."""
    result: dict = {"slice_a": name_a, "slice_b": name_b, "rows_a": len(a), "rows_b": len(b)}

    if "corrected_label" in a.columns and "corrected_label" in b.columns:
        prev_a = a["corrected_label"].value_counts(normalize=True)
        prev_b = b["corrected_label"].value_counts(normalize=True)
        labels = sorted(set(prev_a.index) | set(prev_b.index))
        result["class_prevalence_shift"] = {
            label: round(float(prev_a.get(label, 0.0) - prev_b.get(label, 0.0)), 4)
            for label in labels
        }
        result["max_prevalence_shift"] = max(
            abs(v) for v in result["class_prevalence_shift"].values()
        )

    lengths_a = a["text"].fillna("").astype(str).str.len()
    lengths_b = b["text"].fillna("").astype(str).str.len()
    result["text_length"] = {
        f"{name_a}_mean": float(lengths_a.mean()),
        f"{name_b}_mean": float(lengths_b.mean()),
        f"{name_a}_median": float(lengths_a.median()),
        f"{name_b}_median": float(lengths_b.median()),
    }

    vocab_a = _top_vocabulary(a["text"])
    vocab_b = _top_vocabulary(b["text"])
    union = vocab_a | vocab_b
    result["vocabulary_jaccard"] = float(len(vocab_a & vocab_b) / len(union)) if union else 0.0

    try:
        vectorizer = TfidfVectorizer(min_df=1)
        all_texts = pd.concat([a["text"], b["text"]]).fillna("").astype(str)
        matrix = vectorizer.fit_transform(all_texts)
        centroid_a = np.asarray(matrix[: len(a)].mean(axis=0)).ravel()
        centroid_b = np.asarray(matrix[len(a) :].mean(axis=0)).ravel()
        denom = np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b)
        result["tfidf_centroid_cosine_distance"] = (
            float(1 - centroid_a @ centroid_b / denom) if denom else None
        )
    except ValueError:
        result["tfidf_centroid_cosine_distance"] = None

    return result


def run_drift_analysis(
    labelled: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame] | None = None,
    output_path: Path = Path("reports/ml/drift_report.md"),
) -> list[dict]:
    """Generate the offline distribution-shift report."""
    comparisons: list[dict] = []
    if split_frames:
        if "train" in split_frames and "validation" in split_frames:
            comparisons.append(
                compare_slices(split_frames["train"], split_frames["validation"], "train", "validation")
            )
        if "train" in split_frames and "test" in split_frames:
            comparisons.append(
                compare_slices(split_frames["train"], split_frames["test"], "train", "test")
            )
    sources = labelled["source"].dropna().unique().tolist()
    if "reddit" in sources and "youtube" in sources:
        comparisons.append(
            compare_slices(
                labelled[labelled["source"] == "reddit"],
                labelled[labelled["source"] == "youtube"],
                "reddit",
                "youtube",
            )
        )

    lines = [
        "# Offline distribution-shift analysis",
        "",
        PRELIMINARY_BANNER,
        "This is an offline diagnostic over the current development corpus, "
        "not production drift detection.",
        "",
    ]
    for comparison in comparisons:
        lines += [
            f"## {comparison['slice_a']} vs {comparison['slice_b']}",
            "",
            f"- Rows: {comparison['rows_a']} vs {comparison['rows_b']}",
            f"- Vocabulary overlap (top-term Jaccard): {comparison['vocabulary_jaccard']:.3f}",
        ]
        distance = comparison.get("tfidf_centroid_cosine_distance")
        lines.append(
            f"- TF-IDF centroid cosine distance: {distance:.3f}"
            if distance is not None
            else "- TF-IDF centroid cosine distance: n/a"
        )
        lengths = comparison["text_length"]
        lines.append(
            "- Mean text length: "
            + " vs ".join(f"{v:.0f}" for k, v in lengths.items() if k.endswith("_mean"))
        )
        if "max_prevalence_shift" in comparison:
            lines.append(
                f"- Max class-prevalence shift: {comparison['max_prevalence_shift']:.3f}"
            )
            lines += ["", "| Class | Prevalence shift (a − b) |", "|---|---|"]
            for label, shift in sorted(
                comparison["class_prevalence_shift"].items(), key=lambda x: -abs(x[1])
            ):
                lines.append(f"| {label} | {shift:+.3f} |")
        small = min(comparison["rows_a"], comparison["rows_b"])
        if small < 30:
            lines.append(
                f"\n> Warning: smallest slice has only {small} rows; "
                "differences here are not statistically meaningful."
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    return comparisons


if __name__ == "__main__":
    from src.ml.data import load_labelled_data

    labelled = load_labelled_data()
    split_dir = Path("reports/splits")
    frames = None
    if (split_dir / "train.csv").exists():
        frames = {
            name: pd.read_csv(split_dir / f"{name}.csv")
            for name in ["train", "validation", "test"]
        }
    run_drift_analysis(labelled, frames)
    print("Wrote reports/ml/drift_report.md")

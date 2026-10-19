"""Dataset and annotation audit for the labelled corpus.

Goes beyond ``src.ml.data validate`` (schema, exact duplicates, empty classes)
to answer the questions that decide how far any supervised result can be
trusted:

* class support by source, by sampling strategy and by discussion group;
* how concentrated the corpus is in a few groups;
* near-duplicate texts, and whether any cross discussion groups (a group
  split cannot keep those apart);
* group isolation and test-split class support across the benchmark seeds;
* annotation reliability evidence that actually exists. There is one
  annotator. The raw audit file keeps a first-pass ``manual_label`` beside the
  final ``corrected_label``; their agreement is a *same-annotator revision
  rate*, not inter-annotator agreement, and is reported as such.

    python -m src.ml.dataset_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.data import DEFAULT_AUDIT_PATH, TAXONOMY, load_labelled_data
from src.ml.splits import check_leakage, group_split

DEFAULT_OUTPUT_DIR = Path("reports/data_quality")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
NEAR_DUPLICATE_THRESHOLD = 0.80
#: Fewer labelled rows than this and a class cannot be learned or evaluated
#: with any stability under a 70/15/15 group split.
MIN_ROWS_FOR_EVALUATION = 30


def near_duplicate_pairs(
    df: pd.DataFrame, threshold: float = NEAR_DUPLICATE_THRESHOLD
) -> pd.DataFrame:
    """Pairs of rows whose character n-gram TF-IDF cosine is >= ``threshold``."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = df["text"].fillna("").astype(str).str.casefold()
    if len(texts) < 2:
        return pd.DataFrame(columns=["record_id_a", "record_id_b", "similarity"])
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
    similarity = cosine_similarity(matrix)
    upper = np.triu(similarity, k=1)
    rows, cols = np.where(upper >= threshold)
    records = []
    for a, b in zip(rows, cols):
        records.append(
            {
                "record_id_a": df["record_id"].iloc[a],
                "record_id_b": df["record_id"].iloc[b],
                "group_a": df["parent_id"].iloc[a],
                "group_b": df["parent_id"].iloc[b],
                "label_a": df["corrected_label"].iloc[a],
                "label_b": df["corrected_label"].iloc[b],
                "similarity": round(float(upper[a, b]), 3),
                "crosses_groups": df["parent_id"].iloc[a] != df["parent_id"].iloc[b],
                "labels_differ": df["corrected_label"].iloc[a] != df["corrected_label"].iloc[b],
            }
        )
    return pd.DataFrame(records)


def revision_evidence(raw: pd.DataFrame) -> dict:
    """First-pass vs final label for the same annotator, where both exist."""
    if not {"manual_label", "corrected_label"}.issubset(raw.columns):
        return {"available": False}
    first = raw["manual_label"].astype("string").str.strip()
    final = raw["corrected_label"].astype("string").str.strip()
    both = first.notna() & (first != "") & final.notna()
    if not both.any():
        return {"available": False}
    from sklearn.metrics import cohen_kappa_score

    a, b = first[both].astype(str), final[both].astype(str)
    changed = a != b
    transitions = (
        pd.DataFrame({"first_pass": a[changed], "final": b[changed]})
        .value_counts()
        .reset_index(name="rows")
    )
    return {
        "available": True,
        "rows_with_both_passes": int(both.sum()),
        "rows_without_first_pass": int((~both).sum()),
        "unchanged_share": float((~changed).mean()),
        "revised_rows": int(changed.sum()),
        "first_pass_vs_final_kappa": float(cohen_kappa_score(a, b)),
        "top_revisions": transitions.head(10).to_dict(orient="records"),
        "interpretation": (
            "Same annotator, two passes. This measures how much the first pass was "
            "revised, not how reproducible the labels are between people. "
            "Inter-annotator agreement is unmeasured until a second, independent "
            "annotator labels an overlap sample (src.ml.agreement)."
        ),
    }


def split_support(df: pd.DataFrame, seeds=DEFAULT_SEEDS) -> tuple[pd.DataFrame, list[str]]:
    """Test-split class support and leakage checks for each benchmark seed."""
    rows, problems = [], []
    for seed in seeds:
        split = group_split(df, seed=seed)
        for problem in check_leakage(split):
            problems.append(f"seed {seed}: {problem}")
        counts = split.test["corrected_label"].value_counts()
        row = {"seed": seed, "test_rows": len(split.test), "test_groups": split.test["parent_id"].nunique()}
        row.update({label: int(counts.get(label, 0)) for label in TAXONOMY})
        rows.append(row)
    return pd.DataFrame(rows), problems


def audit_dataset(path: Path = DEFAULT_AUDIT_PATH, seeds=DEFAULT_SEEDS) -> dict:
    raw = pd.read_csv(path)
    df = load_labelled_data(path)

    by_source = pd.crosstab(df["corrected_label"], df["source"]).reindex(TAXONOMY, fill_value=0)
    by_strategy = pd.crosstab(df["corrected_label"], df["sample_strategy"]).reindex(
        TAXONOMY, fill_value=0
    )

    groups = df.groupby("parent_id")
    group_table = pd.DataFrame(
        {
            "rows": groups.size(),
            "actionable_share": groups["actionable"].mean().astype(float),
            "classes": groups["corrected_label"].nunique(),
            "source": groups["source"].first(),
        }
    ).sort_values("rows", ascending=False)
    top3_share = float(group_table["rows"].head(3).sum() / len(df))
    single_row_groups = int((group_table["rows"] == 1).sum())

    near = near_duplicate_pairs(df)
    support, leakage = split_support(df, seeds)

    class_counts = df["corrected_label"].value_counts()
    thin = sorted(
        label for label in TAXONOMY if class_counts.get(label, 0) < MIN_ROWS_FOR_EVALUATION
    )
    rule_agreement = (
        (df["original_rule_label"] == df["corrected_label"]).groupby(df["sample_strategy"]).mean()
    )

    test_zero = {
        label: int((support[label] == 0).sum()) for label in TAXONOMY if label in support
    }

    return {
        "rows": int(len(df)),
        "groups": int(df["parent_id"].nunique()),
        "annotators": sorted(df["annotator_id"].dropna().astype(str).unique().tolist()),
        "class_by_source": by_source.to_dict(),
        "class_by_sample_strategy": by_strategy.to_dict(),
        "classes_below_evaluation_minimum": thin,
        "min_rows_for_evaluation": MIN_ROWS_FOR_EVALUATION,
        "top3_group_share": top3_share,
        "single_row_groups": single_row_groups,
        "group_table": group_table.reset_index().to_dict(orient="records"),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicate_pairs": near.to_dict(orient="records"),
        "near_duplicates_crossing_groups": int(near["crosses_groups"].sum()) if len(near) else 0,
        "split_leakage_problems": leakage,
        "test_support_by_seed": support.to_dict(orient="records"),
        "seeds_with_zero_test_support": test_zero,
        "rule_agreement_by_sample_strategy": rule_agreement.to_dict(),
        "timestamps_available": int(df["created_at"].notna().sum()),
        "annotation_revision": revision_evidence(raw),
    }


def _table(frame: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    lines = [header, "|" + "---|" * len(frame.columns)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return lines


def write_audit_report(audit: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2, default=str))

    by_source = pd.DataFrame(audit["class_by_source"]).rename_axis("class").reset_index()
    by_strategy = pd.DataFrame(audit["class_by_sample_strategy"]).rename_axis("class").reset_index()
    support = pd.DataFrame(audit["test_support_by_seed"])
    groups = pd.DataFrame(audit["group_table"])
    groups["actionable_share"] = groups["actionable_share"].round(2)
    revision = audit["annotation_revision"]

    lines = [
        "# Labelled dataset and annotation audit",
        "",
        "> Generated by `python -m src.ml.dataset_audit`. Describes the development "
        "corpus as it stands; no label is changed by this report.",
        "",
        f"- Rows: {audit['rows']} across {audit['groups']} discussion groups",
        f"- Annotators: {', '.join(audit['annotators'])} (single annotator)",
        f"- Rows with a creation timestamp: {audit['timestamps_available']}"
        " (temporal holdout impossible on this corpus)",
        f"- Top 3 groups hold {audit['top3_group_share']:.0%} of rows; "
        f"{audit['single_row_groups']} groups have a single row",
        f"- Classes below {audit['min_rows_for_evaluation']} rows: "
        + ", ".join(f"`{c}`" for c in audit["classes_below_evaluation_minimum"]),
        "",
        "## Class support by source",
        "",
        *_table(by_source),
        "",
        "## Class support by sampling strategy",
        "",
        "Rows were sampled around the rule classifier's behaviour (precision and "
        "recall slices), so this is not a random sample of the signal set.",
        "",
        *_table(by_strategy),
        "",
        "Rule label agreement with the corrected label, by strategy: "
        + ", ".join(
            f"{k} {v:.0%}" for k, v in audit["rule_agreement_by_sample_strategy"].items()
        ),
        "",
        "## Test-split support across benchmark seeds",
        "",
        "Each seed regenerates the 70/15/15 group split. Counts are test rows per "
        "class; a class with 0–2 test rows contributes a near-arbitrary per-class F1 "
        "to macro averages.",
        "",
        *_table(support),
        "",
        "Group isolation and exact-duplicate checks across splits: "
        + ("**problems found**: " + "; ".join(audit["split_leakage_problems"])
           if audit["split_leakage_problems"] else "no shared groups or duplicate texts on any seed."),
        "",
        "## Near-duplicates",
        "",
        f"Character n-gram TF-IDF cosine ≥ {audit['near_duplicate_threshold']:.2f}: "
        f"{len(audit['near_duplicate_pairs'])} pair(s), "
        f"{audit['near_duplicates_crossing_groups']} crossing discussion groups. "
        "A group split cannot keep a cross-group near-duplicate apart, so such pairs "
        "can sit on both sides of a split.",
        "",
    ]
    if audit["near_duplicate_pairs"]:
        near = pd.DataFrame(audit["near_duplicate_pairs"])[
            ["record_id_a", "record_id_b", "group_a", "group_b", "label_a", "label_b", "similarity"]
        ]
        lines += [*_table(near), ""]

    lines += ["## Annotation reliability", ""]
    if revision.get("available"):
        lines += [
            f"- First-pass and final labels both present: {revision['rows_with_both_passes']} rows",
            f"- Unchanged between passes: {revision['unchanged_share']:.1%} "
            f"({revision['revised_rows']} rows revised)",
            f"- Cohen's kappa, first pass vs final: {revision['first_pass_vs_final_kappa']:.2f}",
            "",
            revision["interpretation"],
            "",
            "Most frequent revisions:",
            "",
            *_table(pd.DataFrame(revision["top_revisions"])),
            "",
        ]
    else:
        lines += ["No first-pass labels available; reliability is unmeasured.", ""]

    lines += [
        "## Groups",
        "",
        *_table(groups[["parent_id", "source", "rows", "classes", "actionable_share"]]),
        "",
    ]
    (output_dir / "dataset_audit.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the labelled corpus")
    parser.add_argument("--input", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    audit = audit_dataset(args.input)
    write_audit_report(audit, args.output_dir)
    print(f"Rows {audit['rows']}, groups {audit['groups']}")
    print(f"Near-duplicate pairs: {len(audit['near_duplicate_pairs'])}")
    revision = audit["annotation_revision"]
    if revision.get("available"):
        print(f"First-pass labels unchanged: {revision['unchanged_share']:.1%}")
    print(f"Report written to {args.output_dir}")


if __name__ == "__main__":
    main()

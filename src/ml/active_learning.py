"""Active-learning review-queue generation for the later human re-audit.

Prioritises rule-vs-model disagreements, model-vs-model disagreements,
low-margin / near-threshold predictions, rare predicted classes,
underrepresented sources and discussion groups, plus a random control
sample. Never overwrites manually labelled data — the queue is written to
``reports/annotation`` for human review.

    python -m src.ml.active_learning \
        --input data/processed/signals_scored.csv \
        --output reports/annotation/active_learning_queue.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.intent import classify_intent
from src.ml.hierarchy import HierarchicalClassifier

QUEUE_COLUMNS = [
    "record_id",
    "text",
    "source",
    "parent_id",
    "rule_label",
    "model_label",
    "model_probability",
    "uncertainty_score",
    "selection_reason",
    "reviewed_label",
    "secondary_label",
    "multi_intent_flag",
    "review_notes",
]

REVIEW_TEMPLATE_COLUMNS = [
    "record_id",
    "text",
    "source",
    "parent_id",
    "current_label",
    "rule_label",
    "model_label",
    "model_probability",
    "selection_reason",
    "reviewed_label",
    "secondary_label",
    "multi_intent_flag",
    "review_notes",
]


def build_queue(
    df: pd.DataFrame,
    model: HierarchicalClassifier,
    second_model: HierarchicalClassifier | None = None,
    labelled_texts: set[str] | None = None,
    max_rows: int = 150,
    random_control: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Rank unlabelled rows for manual review, most informative first."""
    frame = df.copy()
    if "record_id" not in frame.columns:
        frame["record_id"] = frame.get("id", pd.Series(range(len(frame)))).astype(str)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame[frame["text"].str.strip() != ""]
    if labelled_texts:
        normalised = frame["text"].str.strip().str.casefold()
        frame = frame[~normalised.isin(labelled_texts)]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=QUEUE_COLUMNS)

    texts = frame["text"].tolist()
    frame["rule_label"] = [classify_intent(t)[0] for t in texts]
    frame["model_label"] = model.predict(texts)
    scores = model.stage1_scores(texts)
    frame["model_probability"] = scores if scores is not None else np.nan

    threshold = model.threshold
    frame["uncertainty_score"] = (
        1.0 - np.abs(frame["model_probability"] - threshold)
        if scores is not None
        else 0.5
    )

    reasons = pd.Series([""] * len(frame))
    priority = pd.Series(np.zeros(len(frame)))

    def flag(mask: pd.Series, reason: str, weight: float) -> None:
        newly = mask & (reasons == "")
        reasons[newly] = reason
        priority[mask] = np.maximum(priority[mask], weight)

    rule_disagrees = frame["rule_label"] != frame["model_label"]
    flag(rule_disagrees, "rule_model_disagreement", 6.0)

    if second_model is not None:
        second_pred = second_model.predict(texts)
        model_disagrees = pd.Series(second_pred != frame["model_label"].to_numpy())
        flag(model_disagrees, "model_model_disagreement", 5.0)

    if scores is not None:
        near_threshold = (frame["model_probability"] - threshold).abs() < 0.10
        flag(near_threshold, "near_decision_threshold", 4.0)
        confident = frame["model_probability"] > 0.85
        flag(confident & rule_disagrees, "confident_model_rule_disagreement", 4.5)

    class_counts = frame["model_label"].value_counts()
    rare_classes = set(class_counts[class_counts < max(3, len(frame) // 50)].index)
    flag(frame["model_label"].isin(rare_classes), "rare_predicted_class", 3.0)

    if "source" in frame.columns:
        source_counts = frame["source"].value_counts()
        rare_sources = set(source_counts[source_counts < source_counts.max() * 0.25].index)
        flag(frame["source"].isin(rare_sources), "underrepresented_source", 2.0)

    if "parent_id" in frame.columns:
        group_counts = frame["parent_id"].value_counts()
        rare_groups = set(group_counts[group_counts <= 2].index)
        flag(frame["parent_id"].isin(rare_groups), "underrepresented_group", 1.5)

    frame["selection_reason"] = reasons
    frame["priority"] = priority + frame["uncertainty_score"].fillna(0.0)

    selected = (
        frame[frame["selection_reason"] != ""]
        .sort_values(["priority", "uncertainty_score"], ascending=False)
        .head(max_rows - random_control)
    )
    rng = np.random.default_rng(seed)
    remaining = frame.drop(selected.index)
    if len(remaining) and random_control:
        control = remaining.sample(
            n=min(random_control, len(remaining)), random_state=rng.integers(0, 2**31)
        ).copy()
        control["selection_reason"] = "random_control"
        selected = pd.concat([selected, control])

    for column in ["source", "parent_id"]:
        if column not in selected.columns:
            selected[column] = pd.NA
    for column in ["reviewed_label", "secondary_label", "multi_intent_flag", "review_notes"]:
        selected[column] = ""
    return selected[QUEUE_COLUMNS].reset_index(drop=True)


def write_review_template(labelled: pd.DataFrame, path: Path) -> None:
    """Write the re-audit template for existing labelled rows (labels untouched)."""
    template = pd.DataFrame()
    template["record_id"] = labelled["record_id"]
    template["text"] = labelled["text"]
    template["source"] = labelled["source"]
    template["parent_id"] = labelled["parent_id"]
    template["current_label"] = labelled["corrected_label"]
    template["rule_label"] = [classify_intent(str(t))[0] for t in labelled["text"]]
    template["model_label"] = ""
    template["model_probability"] = ""
    template["selection_reason"] = "full_reaudit"
    for column in ["reviewed_label", "secondary_label", "multi_intent_flag", "review_notes"]:
        template[column] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    template[REVIEW_TEMPLATE_COLUMNS].to_csv(path, index=False)


def main() -> None:
    from src.ml.data import load_labelled_data
    from src.ml.persistence import DEFAULT_ARTIFACT_DIR, load_hierarchical_model

    parser = argparse.ArgumentParser(description="Generate the active-learning review queue")
    parser.add_argument("--input", type=Path, default=Path("data/demo/signals_scored_portfolio.csv"))
    parser.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/annotation/active_learning_queue.csv")
    )
    parser.add_argument("--max-rows", type=int, default=150)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    model, _ = load_hierarchical_model(args.model)
    labelled = load_labelled_data()
    labelled_texts = set(labelled["text"].astype(str).str.strip().str.casefold())
    queue = build_queue(df, model, labelled_texts=labelled_texts, max_rows=args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False)
    write_review_template(labelled, args.output.parent / "review_queue.csv")
    print(f"Queue rows: {len(queue)} -> {args.output}")
    print(queue["selection_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()

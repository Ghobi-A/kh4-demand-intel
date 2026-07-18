"""Unified experiment runner.

    python -m src.ml.benchmark \
        --input reports/audits/intent_audit_corrected.csv \
        --models majority rules logistic linear_svm char_word_svm \
        --seeds 42 43 44 45 46 \
        --output reports/ml/

For each seed: validate data, regenerate the group-aware split, train
each model, calibrate and select the Stage 1 threshold on validation
only, then evaluate flat multiclass, Stage 1, Stage 2, hierarchical
end-to-end, and ranking metrics on the held-out test split.

All outputs from the current audit corpus carry
``evaluation_status = development_preliminary``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import EVALUATION_STATUS_DEVELOPMENT
from src.ml.calibration import ScoreCalibrator, calibration_curve_points
from src.ml.config import ExperimentConfig
from src.ml.data import (
    GENERAL_LABEL,
    dataset_hash,
    load_labelled_data,
    validate_labelled_data,
)
from src.ml.hierarchy import ACTIONABLE, HierarchicalClassifier, to_binary_labels
from src.ml.metrics import binary_metrics, multiclass_metrics
from src.ml.models import SCORE_PROBABILITY, build_model
from src.ml.ranking import ranking_metrics, save_ranking_figures
from src.ml.splits import SplitResult, check_leakage, group_split, save_split_manifests
from src.ml.thresholds import select_threshold

SUMMARY_METRICS = [
    "stage1_f1",
    "stage1_pr_auc",
    "stage1_precision",
    "stage1_recall",
    "stage1_brier",
    "stage1_ece",
    "stage2_macro_f1",
    "flat_macro_f1",
    "flat_weighted_f1",
    "e2e_macro_f1",
    "e2e_weighted_f1",
    "lift_at_10pct",
    "precision_at_10pct",
]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def train_stage1(model_name: str, split: SplitResult, config: ExperimentConfig, seed: int):
    """Train, calibrate (validation only), and threshold a Stage 1 model.

    Returns (model, calibrator, threshold_info, notes).
    """
    notes: list[str] = []
    train_binary = to_binary_labels(split.train["corrected_label"])
    stage1_targets = np.where(train_binary == 1, ACTIONABLE, GENERAL_LABEL)
    if model_name == "rules":
        from src.ml.models import RuleBaselineClassifier

        model = RuleBaselineClassifier(binary=True)
    else:
        model = build_model(model_name, seed=seed)
    model.fit(split.train["text"].tolist(), stage1_targets.tolist())

    val_texts = split.validation["text"].tolist()
    val_binary = to_binary_labels(split.validation["corrected_label"])
    val_scores = model.positive_scores(val_texts, ACTIONABLE)

    calibrator = None
    threshold_info = {"selected_threshold": 0.5, "selection_metric": "default", "validation_score": None}
    if val_scores is not None:
        if model.score_type != SCORE_PROBABILITY or config.calibration_method == "isotonic":
            try:
                calibrator = ScoreCalibrator(method=config.calibration_method).fit(
                    val_scores, val_binary
                )
                val_scores = calibrator.predict_proba(val_scores)
            except ValueError as exc:
                notes.append(f"calibration skipped: {exc}")
                calibrator = None
        threshold_info = select_threshold(
            val_binary,
            val_scores,
            objective=config.threshold.objective,
            min_precision=config.threshold.min_precision,
            min_recall=config.threshold.min_recall,
        )
    else:
        notes.append("model provides no scores; label-based Stage 1 decisions")
    return model, calibrator, threshold_info, notes


def evaluate_model_on_seed(
    model_name: str, split: SplitResult, config: ExperimentConfig, seed: int
) -> tuple[dict, pd.DataFrame, dict]:
    """Evaluate one model for one seed. Returns (metric row, predictions, extras)."""
    test_texts = split.test["text"].tolist()
    test_labels = split.test["corrected_label"].astype(str).to_numpy()
    test_binary = to_binary_labels(test_labels)

    # Flat multiclass benchmark.
    flat_model = build_model(model_name, seed=seed)
    flat_model.fit(split.train["text"].tolist(), split.train["corrected_label"].tolist())
    flat_pred = flat_model.predict(test_texts)
    flat = multiclass_metrics(test_labels, flat_pred)

    # Hierarchical benchmark.
    stage1, calibrator, threshold_info, notes = train_stage1(model_name, split, config, seed)
    stage2 = build_model(model_name, seed=seed)
    hier = HierarchicalClassifier(
        stage1,
        stage2,
        threshold=threshold_info["selected_threshold"],
        calibrator=calibrator,
    )
    train_actionable = split.train[
        to_binary_labels(split.train["corrected_label"]) == 1
    ]
    stage2.fit(train_actionable["text"].tolist(), train_actionable["corrected_label"].tolist())

    stage1_scores = hier.stage1_scores(test_texts)
    stage1_pred = hier.stage1_predict(test_texts)
    has_probability = stage1_scores is not None and (
        calibrator is not None or stage1.score_type == SCORE_PROBABILITY
    )
    stage1_eval = binary_metrics(
        test_binary,
        stage1_pred,
        y_prob=stage1_scores if has_probability else None,
        y_score=stage1_scores,
    )

    # Stage 2 in isolation: true actionable test rows only.
    actionable_mask = test_binary == 1
    stage2_eval = None
    if actionable_mask.any():
        actionable_texts = [t for t, keep in zip(test_texts, actionable_mask) if keep]
        stage2_pred = stage2.predict(actionable_texts)
        stage2_eval = multiclass_metrics(test_labels[actionable_mask], stage2_pred)

    e2e_pred = hier.predict(test_texts)
    e2e = multiclass_metrics(test_labels, e2e_pred)

    rank = {}
    if stage1_scores is not None:
        rank = ranking_metrics(
            test_binary,
            stage1_scores,
            k_values=tuple(config.ranking_k_values),
            percentages=tuple(config.ranking_percentages),
        )

    row = {
        "model": model_name,
        "seed": seed,
        "evaluation_mode": split.evaluation_mode,
        "evaluation_status": config.evaluation_status,
        "selected_threshold": threshold_info["selected_threshold"],
        "threshold_objective": threshold_info["selection_metric"],
        "calibrated": calibrator is not None,
        "stage1_accuracy": stage1_eval["accuracy"],
        "stage1_precision": stage1_eval["precision"],
        "stage1_recall": stage1_eval["recall"],
        "stage1_f1": stage1_eval["f1"],
        "stage1_roc_auc": stage1_eval.get("roc_auc"),
        "stage1_pr_auc": stage1_eval.get("pr_auc"),
        "stage1_brier": stage1_eval.get("brier_score"),
        "stage1_ece": stage1_eval.get("ece"),
        "stage2_macro_f1": stage2_eval["macro_f1"] if stage2_eval else None,
        "stage2_accuracy": stage2_eval["accuracy"] if stage2_eval else None,
        "flat_accuracy": flat["accuracy"],
        "flat_macro_f1": flat["macro_f1"],
        "flat_weighted_f1": flat["weighted_f1"],
        "e2e_accuracy": e2e["accuracy"],
        "e2e_macro_f1": e2e["macro_f1"],
        "e2e_weighted_f1": e2e["weighted_f1"],
        "notes": "; ".join(notes),
        **{k: v for k, v in rank.items()},
    }

    predictions = split.test[["record_id", "text", "source", "parent_id"]].copy()
    predictions["true_label"] = test_labels
    predictions["true_actionable"] = test_binary
    predictions["flat_predicted_label"] = flat_pred
    predictions["e2e_predicted_label"] = e2e_pred
    predictions["stage1_predicted_actionable"] = stage1_pred
    predictions["stage1_score"] = stage1_scores if stage1_scores is not None else np.nan
    predictions["stage1_score_is_probability"] = bool(has_probability)
    predictions["model"] = model_name
    predictions["seed"] = seed

    extras = {
        "stage1_eval": stage1_eval,
        "stage2_eval": stage2_eval,
        "flat_eval": flat,
        "e2e_eval": e2e,
        "hierarchical_model": hier,
        "test_binary": test_binary,
        "stage1_scores": stage1_scores,
        "has_probability": has_probability,
    }
    return row, predictions, extras


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/median per model over seeds for headline metrics."""
    records = []
    for model, group in per_seed.groupby("model"):
        record: dict = {"model": model, "n_seeds": len(group)}
        for metric in SUMMARY_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_median"] = float(values.median())
        records.append(record)
    return pd.DataFrame(records)


def save_calibration_figure(
    test_binary: np.ndarray, probabilities: np.ndarray, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_pred, frac_pos, counts = calibration_curve_points(test_binary, probabilities, n_bins=8)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], "--", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.set_title("Stage 1 calibration (development preliminary)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction actionable")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_benchmark(config: ExperimentConfig) -> dict:
    """Run the full multi-seed benchmark and write all outputs."""
    output_dir = Path(config.output_dir)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_labelled_data(Path(config.input_path))
    validation_report = validate_labelled_data(df)
    usable = df[df["corrected_label"].notna() & (df["text"].astype(str).str.strip() != "")]

    per_seed_rows: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    reference_extras = None
    reference_split = None

    for seed in config.seeds:
        split = group_split(
            usable,
            seed=seed,
            train_frac=config.split.train,
            validation_frac=config.split.validation,
            test_frac=config.split.test,
        )
        leaks = check_leakage(split)
        if leaks:
            raise RuntimeError(f"Split leakage detected for seed {seed}: {leaks}")
        if seed == config.seeds[0]:
            save_split_manifests(split, Path("reports/splits"))
            reference_split = split
        for model_name in config.models:
            row, predictions, extras = evaluate_model_on_seed(model_name, split, config, seed)
            per_seed_rows.append(row)
            all_predictions.append(predictions)
            if seed == config.seeds[0] and model_name == "logistic":
                reference_extras = extras

    per_seed = pd.DataFrame(per_seed_rows)
    summary = summarise(per_seed)
    per_seed.to_csv(output_dir / "results_per_seed.csv", index=False)
    summary.to_csv(output_dir / "results_summary.csv", index=False)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(output_dir / "tables" / "test_predictions.csv", index=False)

    if reference_extras is not None and reference_extras["stage1_scores"] is not None:
        save_ranking_figures(
            reference_extras["test_binary"],
            reference_extras["stage1_scores"],
            output_dir / "figures",
        )
        if reference_extras["has_probability"]:
            save_calibration_figure(
                reference_extras["test_binary"],
                reference_extras["stage1_scores"],
                output_dir / "figures" / "calibration_stage1.png",
            )

    metadata = {
        "run_id": uuid.uuid4().hex[:12],
        "experiment_name": config.experiment_name,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_hash": dataset_hash(df),
        "annotation_version": validation_report["annotation_version"],
        "row_count": validation_report["total_rows"],
        "class_counts": validation_report["class_counts"],
        "seeds": config.seeds,
        "seed_scope": config.seed_scope,
        "split_method": config.split.method,
        "evaluation_mode": reference_split.evaluation_mode if reference_split else None,
        "train_rows": len(reference_split.train) if reference_split else None,
        "validation_rows": len(reference_split.validation) if reference_split else None,
        "test_rows": len(reference_split.test) if reference_split else None,
        "train_groups": int(reference_split.train["parent_id"].dropna().nunique()) if reference_split else None,
        "validation_groups": int(reference_split.validation["parent_id"].dropna().nunique()) if reference_split else None,
        "test_groups": int(reference_split.test["parent_id"].dropna().nunique()) if reference_split else None,
        "models": config.models,
        "threshold_objective": config.threshold.objective,
        "calibration_method": config.calibration_method,
        "evaluation_status": config.evaluation_status,
        "config": config.to_dict(),
        "note": (
            "Preliminary development benchmark on the current 200-row audit "
            "corpus. Not final held-out performance. Repeated-seed spread "
            "describes variability across seeded group splits and model "
            "initialisations, not a population confidence guarantee."
        ),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    from src.ml.reporting import write_model_comparison_report

    write_model_comparison_report(summary, metadata, output_dir / "model_comparison.md")
    return {"per_seed": per_seed, "summary": summary, "metadata": metadata, "predictions": predictions_df}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supervised NLP benchmark")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="YAML experiment config")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    if args.input:
        config.input_path = str(args.input)
    if args.models:
        config.models = args.models
    if args.seeds:
        config.seeds = args.seeds
    if args.output:
        config.output_dir = str(args.output)
    if config.evaluation_status != EVALUATION_STATUS_DEVELOPMENT:
        print(f"evaluation_status = {config.evaluation_status}")

    result = run_benchmark(config)
    print(result["summary"].to_string(index=False))
    print(f"\nOutputs written to {config.output_dir} (status: {config.evaluation_status})")


if __name__ == "__main__":
    main()

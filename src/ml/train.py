"""Train and persist the development hierarchical model.

    python -m src.ml.train --model logistic --seed 42

Trains Stage 1 + Stage 2 on the group-split training data, calibrates
and selects the threshold on validation only, and saves artefacts under
``artifacts/hierarchical`` with full metadata. The artefact is tagged
``development_preliminary`` until the corpus re-audit.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from src.ml import EVALUATION_STATUS_DEVELOPMENT
from src.ml.benchmark import _git_commit, train_stage1
from src.ml.config import ExperimentConfig
from src.ml.data import dataset_hash, load_labelled_data
from src.ml.hierarchy import HierarchicalClassifier, to_binary_labels
from src.ml.models import build_model
from src.ml.persistence import DEFAULT_ARTIFACT_DIR, save_hierarchical_model
from src.ml.splits import check_leakage, group_split


def train_and_save(
    model_name: str = "logistic",
    seed: int = 42,
    input_path: Path | None = None,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> Path:
    config = ExperimentConfig(seeds=[seed], models=[model_name])
    if input_path:
        config.input_path = str(input_path)

    df = load_labelled_data(Path(config.input_path))
    usable = df[df["corrected_label"].notna() & (df["text"].astype(str).str.strip() != "")]
    split = group_split(usable, seed=seed)
    leaks = check_leakage(split)
    if leaks:
        raise RuntimeError(f"Split leakage detected: {leaks}")

    stage1, calibrator, threshold_info, notes = train_stage1(model_name, split, config, seed)
    stage2 = build_model(model_name, seed=seed)
    train_actionable = split.train[to_binary_labels(split.train["corrected_label"]) == 1]
    stage2.fit(train_actionable["text"].tolist(), train_actionable["corrected_label"].tolist())

    model = HierarchicalClassifier(
        stage1, stage2, threshold=threshold_info["selected_threshold"], calibrator=calibrator
    )
    metadata = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": seed,
        "model_name": model_name,
        "dataset_hash": dataset_hash(df),
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
        "evaluation_mode": split.evaluation_mode,
        "evaluation_status": EVALUATION_STATUS_DEVELOPMENT,
        "threshold_selection": threshold_info,
        "notes": notes,
    }
    path = save_hierarchical_model(model, metadata, artifact_dir)
    print(f"Saved {model.name} to {path} (threshold={model.threshold:.3f})")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist the hierarchical model")
    parser.add_argument("--model", default="logistic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    train_and_save(args.model, args.seed, args.input, args.artifacts)


if __name__ == "__main__":
    main()

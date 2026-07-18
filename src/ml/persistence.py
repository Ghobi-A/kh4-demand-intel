"""Model persistence: save/load trained hierarchical artefacts.

Artefacts live under ``artifacts/`` (git-ignored) with stage1/stage2
models, the calibrator, the frozen threshold, class mapping, and
training metadata (git commit, seed, dataset hash, evaluation status).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.ml.calibration import ScoreCalibrator
from src.ml.hierarchy import HierarchicalClassifier

DEFAULT_ARTIFACT_DIR = Path("artifacts/hierarchical")

METADATA_FILE = "metadata.json"


def save_hierarchical_model(
    model: HierarchicalClassifier,
    metadata: dict,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> Path:
    """Persist a trained hierarchical classifier and its metadata."""
    artifact_dir = Path(artifact_dir)
    (artifact_dir / "stage1").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "stage2").mkdir(parents=True, exist_ok=True)

    joblib.dump(model.stage1, artifact_dir / "stage1" / "model.joblib")
    joblib.dump(model.stage2, artifact_dir / "stage2" / "model.joblib")
    if model.calibrator is not None:
        joblib.dump(model.calibrator, artifact_dir / "stage1" / "calibrator.joblib")

    stored = dict(metadata)
    stored.update(
        {
            "threshold": model.threshold,
            "calibrated": model.calibrator is not None,
            "stage1_model_name": model.stage1.name,
            "stage2_model_name": model.stage2.name,
            "stage1_score_type": model.stage1.score_type,
            "stage2_score_type": model.stage2.score_type,
            "stage2_classes": [str(c) for c in model.stage2.classes_],
        }
    )
    (artifact_dir / METADATA_FILE).write_text(json.dumps(stored, indent=2, default=str))
    return artifact_dir


def load_hierarchical_model(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> tuple[HierarchicalClassifier, dict]:
    """Load a persisted hierarchical classifier and its metadata."""
    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / METADATA_FILE).read_text())
    stage1 = joblib.load(artifact_dir / "stage1" / "model.joblib")
    stage2 = joblib.load(artifact_dir / "stage2" / "model.joblib")
    calibrator: ScoreCalibrator | None = None
    calibrator_path = artifact_dir / "stage1" / "calibrator.joblib"
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
    model = HierarchicalClassifier(
        stage1,
        stage2,
        threshold=float(metadata.get("threshold", 0.5)),
        calibrator=calibrator,
    )
    return model, metadata

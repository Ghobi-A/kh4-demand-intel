"""Unified single-text inference.

    from src.ml.inference import predict_text
    result = predict_text("Still waiting for news, Square please give us something")

Returns the shared schema. If trained artefacts are absent, falls back to
the rule baseline and says so — the rule confidence indicator is never
presented as a probability.
"""

from __future__ import annotations

from pathlib import Path

from src.intent import classify_intent
from src.ml.data import is_actionable
from src.ml.persistence import DEFAULT_ARTIFACT_DIR, load_hierarchical_model

_MODEL_CACHE: dict[str, tuple] = {}
_LIVE_MODEL: dict[str, object] = {}


def ensure_live_model(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> tuple[Path | None, dict]:
    """A usable trained model directory for interactive inference.

    Trained artefacts are git-ignored, so a deployed checkout has none and
    would silently fall back to the rule baseline. If no compatible artefact
    exists, the same logistic model ``python -m src.ml.train`` builds is
    trained once per process from the committed labelled corpus (about a
    second) into a temporary directory. Returns ``(None, info)`` when that is
    impossible, and the caller falls back to rules and says so.
    """
    if "dir" in _LIVE_MODEL:
        return _LIVE_MODEL["dir"], dict(_LIVE_MODEL["info"])

    import json
    import tempfile

    from src.ml.probabilities import _is_compatible

    artifact_dir = Path(artifact_dir)
    metadata_path = artifact_dir / "metadata.json"
    if metadata_path.exists() and _is_compatible(json.loads(metadata_path.read_text())):
        directory, source = artifact_dir, "persisted_artifact"
    else:
        try:
            from src.ml.train import train_and_save

            directory = Path(tempfile.mkdtemp(prefix="kh4-live-model-")) / "hierarchical"
            train_and_save(model_name="logistic", seed=42, artifact_dir=directory)
            source = "trained_this_session"
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a rule fallback
            return None, {"source": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    metadata = json.loads((directory / "metadata.json").read_text())
    info = {
        "source": source,
        "model_name": metadata.get("model_name"),
        "dataset_hash": metadata.get("dataset_hash"),
        "seed": metadata.get("seed"),
        "evaluation_status": metadata.get("evaluation_status"),
    }
    _LIVE_MODEL["dir"], _LIVE_MODEL["info"] = directory, info
    return directory, dict(info)


def _load_cached(artifact_dir: Path):
    key = str(artifact_dir)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_hierarchical_model(artifact_dir)
    return _MODEL_CACHE[key]


def predict_text(text: str, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict:
    """Classify one text through the trained two-stage model (or rule fallback)."""
    artifact_dir = Path(artifact_dir)
    if not (artifact_dir / "metadata.json").exists():
        label, indicator = classify_intent(str(text))
        return {
            "is_actionable": is_actionable(label),
            "intent_label": label,
            "rule_confidence_indicator": indicator,
            "model_name": "rules",
            "evaluation_status": "rule_baseline_fallback",
            "note": (
                "No trained model artefacts found; rule baseline used. The "
                "confidence indicator is not a probability."
            ),
        }

    model, metadata = _load_cached(artifact_dir)
    record = model.predict_records([str(text)])[0]
    record["evaluation_status"] = metadata.get("evaluation_status", "unknown")
    if "actionable_score_uncalibrated" in record:
        record["note"] = "Stage 1 score is an uncalibrated margin, not a probability."
    return record

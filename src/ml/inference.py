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

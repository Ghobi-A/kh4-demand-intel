"""Calibrated actionable probabilities for temporal modelling.

The weekly behavioural demand proxy sums P(actionable) over the comments
observed in a period, so those probabilities have to be both calibrated and
honest about provenance:

* Rows that took part in supervised model development get **out-of-fold**
  probabilities from group-aware folds, so no row is scored by a model that
  saw it or any comment from its discussion group.
* Every other row is scored by the persisted hierarchical model.
* The 0.7/0.3 rule-confidence indicator is never used as a probability.

Each scored row carries ``probability_source`` so a reader can tell which of
those paths produced its number.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.benchmark import train_stage1
from src.ml.config import ExperimentConfig
from src.ml.data import TAXONOMY, load_labelled_data
from src.ml.hierarchy import ACTIONABLE, HierarchicalClassifier
from src.ml.models import SCORE_PROBABILITY
from src.ml.persistence import DEFAULT_ARTIFACT_DIR, load_hierarchical_model
from src.ml.splits import SplitResult, _effective_groups, group_split

LOGGER = logging.getLogger(__name__)

SOURCE_OOF = "oof_supervised"
SOURCE_PERSISTED = "persisted_supervised"
SOURCE_EXCLUDED = "excluded_training_row"
SOURCE_UNAVAILABLE = "unavailable"

PROBABILITY_COLUMN = "actionable_probability"
PROVENANCE_COLUMN = "probability_source"


class ProbabilityUnavailableError(RuntimeError):
    """Raised when no calibrated supervised probabilities can be produced."""


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _is_compatible(metadata: dict) -> bool:
    """Check a persisted artefact against the current taxonomy and score type.

    An artefact is usable when its Stage 1 output is a probability — either
    because a calibrator was fitted, or because the estimator emits
    probabilities natively. An uncalibrated margin is not accepted.
    """
    classes = set(metadata.get("stage2_classes", []))
    if not classes or not classes.issubset(set(TAXONOMY)):
        return False
    if metadata.get("calibrated", False):
        return True
    return metadata.get("stage1_score_type") == SCORE_PROBABILITY


def ensure_hierarchical_model(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    seed: int = 42,
    model_name: str = "logistic",
    train_if_missing: bool = True,
) -> tuple[HierarchicalClassifier, dict]:
    """Load the persisted hierarchical model, training it once if absent.

    No new model architecture is introduced: this is the same two-stage model
    the supervised layer already trains, loaded or rebuilt deterministically.
    """
    artifact_dir = Path(artifact_dir)
    metadata_path = artifact_dir / "metadata.json"

    if metadata_path.exists():
        try:
            model, metadata = load_hierarchical_model(artifact_dir)
        except Exception as exc:  # noqa: BLE001 - a stale artefact must not be trusted
            LOGGER.warning("Could not load artefacts at %s: %s", artifact_dir, exc)
        else:
            if _is_compatible(metadata):
                metadata = dict(metadata)
                metadata["artifact_dir"] = str(artifact_dir)
                metadata["artifact_hash"] = _file_hash(artifact_dir / "stage1" / "model.joblib")
                metadata["probability_generation"] = "persisted_artifact"
                return model, metadata
            LOGGER.warning("Artefacts at %s are incompatible; retraining", artifact_dir)

    if not train_if_missing:
        raise ProbabilityUnavailableError(
            f"No compatible model artefacts at {artifact_dir} and training is disabled"
        )

    from src.ml.train import train_and_save

    try:
        train_and_save(model_name=model_name, seed=seed, artifact_dir=artifact_dir)
        model, metadata = load_hierarchical_model(artifact_dir)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear unavailability
        raise ProbabilityUnavailableError(
            f"Could not train a hierarchical model for probabilities: {exc}"
        ) from exc

    metadata = dict(metadata)
    metadata["artifact_dir"] = str(artifact_dir)
    metadata["artifact_hash"] = _file_hash(artifact_dir / "stage1" / "model.joblib")
    metadata["probability_generation"] = "trained_if_missing"
    return model, metadata


def group_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> list[np.ndarray]:
    """Assign whole discussion groups to folds, deterministically by seed."""
    groups, _ = _effective_groups(df)
    unique = sorted(groups.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_folds = max(2, min(n_folds, len(unique)))
    assignment = {group: index % n_folds for index, group in enumerate(unique)}
    fold_ids = groups.map(assignment).to_numpy()
    return [np.where(fold_ids == fold)[0] for fold in range(n_folds)]


def _inner_split(remainder: pd.DataFrame, seed: int) -> SplitResult:
    """Group-aware train/validation split of a fold's remaining rows.

    ``group_split`` always produces three parts; the held-out fold already
    plays the test role here, so its test slice is folded back into training.
    Calibration still sees only groups absent from training.
    """
    base = group_split(remainder, seed=seed)
    train = pd.concat([base.train, base.test], ignore_index=True)
    return SplitResult(
        train=train,
        validation=base.validation,
        test=base.test.iloc[0:0],
        seed=seed,
        evaluation_mode=base.evaluation_mode,
    )


def oof_actionable_probabilities(
    labelled: pd.DataFrame,
    model_name: str = "logistic",
    seed: int = 42,
    n_folds: int = 5,
    by_text: bool = False,
) -> dict[str, float]:
    """Out-of-fold calibrated P(actionable) keyed by ``record_id``.

    Each fold's held-out groups are scored by a model trained and calibrated
    only on the other groups, so a row's probability never reflects a model
    that saw its own discussion.
    """
    usable = labelled[
        labelled["corrected_label"].notna()
        & (labelled["text"].astype(str).str.strip() != "")
    ].reset_index(drop=True)
    if usable.empty:
        return {}

    config = ExperimentConfig(seeds=[seed], models=[model_name])
    probabilities: dict[str, float] = {}

    for held_out_idx in group_folds(usable, n_folds=n_folds, seed=seed):
        if len(held_out_idx) == 0:
            continue
        held_out = usable.iloc[held_out_idx]
        remainder = usable.drop(index=usable.index[held_out_idx]).reset_index(drop=True)
        if remainder.empty or remainder["corrected_label"].nunique() < 2:
            continue
        try:
            inner = _inner_split(remainder, seed)
        except ValueError as exc:
            LOGGER.warning("Skipping OOF fold: %s", exc)
            continue

        model, calibrator, _threshold, _notes = train_stage1(model_name, inner, config, seed)
        if not _yields_probabilities(model, calibrator):
            LOGGER.warning(
                "Fold model produces only uncalibrated scores; its rows stay unscored"
            )
            continue
        raw = model.positive_scores(held_out["text"].astype(str).tolist(), ACTIONABLE)
        if raw is None:
            continue
        calibrated = calibrator.predict_proba(raw) if calibrator is not None else raw
        keys = (
            held_out["text"].map(normalise_text_key)
            if by_text
            else held_out["record_id"].astype(str)
        )
        for key, value in zip(keys, calibrated):
            probabilities[str(key)] = float(value)

    return probabilities


def _yields_probabilities(stage1, calibrator) -> bool:
    """True when Stage 1 output is a probability rather than a raw margin.

    A fitted calibrator makes any score a probability; a model whose own
    ``score_type`` is already a probability (logistic) needs no calibrator.
    This mirrors the rule ``HierarchicalClassifier.predict_records`` uses to
    decide whether it may call its output ``actionable_probability``.
    """
    return calibrator is not None or getattr(stage1, "score_type", None) == SCORE_PROBABILITY


def _persisted_probabilities(model: HierarchicalClassifier, texts: list[str]) -> np.ndarray | None:
    """Calibrated P(actionable) from the persisted model, or None if unavailable."""
    if not _yields_probabilities(model.stage1, model.calibrator):
        return None
    scores = model.stage1_scores(texts)
    if scores is None:
        return None
    return np.asarray(scores, dtype=float)


def normalise_text_key(text) -> str:
    """Deterministic key for matching a signal row to a labelled row by content."""
    return " ".join(str(text).split()).strip().casefold()


def attach_actionable_probabilities(
    signals: pd.DataFrame,
    model: HierarchicalClassifier,
    oof_map: dict[str, float] | None = None,
    id_column: str = "id",
    text_oof_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add ``actionable_probability`` and ``probability_source`` to signal rows.

    Rows matching the labelled corpus take their out-of-fold value. Matching is
    by record id where available and by normalised text otherwise: the portfolio
    extract strips identifiers for privacy, and without a content fallback those
    rows would silently be scored in-sample by a model that trained on them.
    """
    result = signals.copy().reset_index(drop=True)
    oof_map = oof_map or {}
    text_oof_map = text_oof_map or {}

    texts = result["text"].astype(str) if "text" in result.columns else pd.Series([""] * len(result))
    has_text = texts.str.strip() != ""

    ids = (
        result[id_column].astype(str)
        if id_column in result.columns
        else pd.Series([""] * len(result))
    )
    by_id = ids.isin(set(oof_map)) if oof_map else pd.Series(False, index=result.index)

    text_keys = texts.map(normalise_text_key)
    by_text = (
        text_keys.isin(set(text_oof_map)) & ~by_id
        if text_oof_map
        else pd.Series(False, index=result.index)
    )
    in_corpus = by_id | by_text

    probabilities = pd.Series(np.nan, index=result.index, dtype=float)
    provenance = pd.Series(SOURCE_UNAVAILABLE, index=result.index, dtype=object)

    if by_id.any():
        probabilities[by_id] = ids[by_id].map(oof_map).astype(float)
        provenance[by_id] = SOURCE_OOF
    if by_text.any():
        probabilities[by_text] = text_keys[by_text].map(text_oof_map).astype(float)
        provenance[by_text] = SOURCE_OOF

    to_score = has_text & ~in_corpus
    if to_score.any():
        scored = _persisted_probabilities(model, texts[to_score].tolist())
        if scored is not None:
            probabilities.loc[to_score] = scored
            provenance.loc[to_score] = SOURCE_PERSISTED

    result[PROBABILITY_COLUMN] = probabilities
    result[PROVENANCE_COLUMN] = provenance
    return result


def build_probabilities(
    signals: pd.DataFrame,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    labelled_path: Path | None = None,
    model_name: str = "logistic",
    seed: int = 42,
    n_folds: int = 5,
    use_oof: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """End-to-end: ensure a model, compute OOF, score signals, return metadata."""
    model, model_metadata = ensure_hierarchical_model(
        artifact_dir=artifact_dir, seed=seed, model_name=model_name
    )
    if not _yields_probabilities(model.stage1, model.calibrator):
        raise ProbabilityUnavailableError(
            "The trained model produced no calibrator, so no calibrated "
            "probabilities are available. The rule confidence indicator is "
            "not a probability and is never substituted."
        )

    oof_map: dict[str, float] = {}
    text_oof_map: dict[str, float] = {}
    if use_oof:
        labelled = load_labelled_data(labelled_path) if labelled_path else load_labelled_data()
        oof_map = oof_actionable_probabilities(
            labelled, model_name=model_name, seed=seed, n_folds=n_folds
        )
        text_oof_map = oof_actionable_probabilities(
            labelled, model_name=model_name, seed=seed, n_folds=n_folds, by_text=True
        )

    scored = attach_actionable_probabilities(signals, model, oof_map, text_oof_map=text_oof_map)
    counts = scored[PROVENANCE_COLUMN].value_counts().to_dict()
    matched = int(counts.get(SOURCE_OOF, 0))
    if oof_map and not matched:
        LOGGER.warning(
            "No signal row matched the labelled corpus by id or text, so every "
            "probability came from the persisted model. If this dataset contains "
            "audit rows, some probabilities may be in-sample."
        )
    metadata = {
        "model_run": {
            "model_name": model_metadata.get("model_name"),
            "seed": model_metadata.get("seed"),
            "git_commit": model_metadata.get("git_commit"),
            "dataset_hash": model_metadata.get("dataset_hash"),
            "annotation_version": model_metadata.get("annotation_version"),
            "artifact_dir": model_metadata.get("artifact_dir"),
            "artifact_hash": model_metadata.get("artifact_hash"),
            "evaluation_status": model_metadata.get("evaluation_status"),
            "threshold": model_metadata.get("threshold"),
        },
        "probability_generation": model_metadata.get("probability_generation"),
        "oof_folds": n_folds if use_oof else 0,
        "oof_rows": len(oof_map),
        "oof_matched_rows": matched,
        "oof_match_note": (
            "Rows matched to the labelled corpus (by id, or by normalised text "
            "when identifiers were stripped) use out-of-fold probabilities; the "
            "rest use the persisted model."
        ),
        "probability_source_counts": {str(k): int(v) for k, v in counts.items()},
    }
    return scored, metadata


__all__ = [
    "PROBABILITY_COLUMN",
    "PROVENANCE_COLUMN",
    "ProbabilityUnavailableError",
    "SOURCE_EXCLUDED",
    "SOURCE_OOF",
    "SOURCE_PERSISTED",
    "SOURCE_UNAVAILABLE",
    "attach_actionable_probabilities",
    "build_probabilities",
    "ensure_hierarchical_model",
    "group_folds",
    "oof_actionable_probabilities",
]

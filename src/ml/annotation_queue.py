"""Stratified annotation queue and quality controls for corpus expansion.

The labelled corpus is the binding constraint on every supervised result: two
classes have no examples, three have fewer than ten, and 197 of 200 rows are
YouTube. This module prepares the *next human annotation batch*. It never
writes a label.

Selection is stratified with explicit quotas, so the batch targets the gaps
rather than whatever the model finds most uncertain:

1. ``random_control`` is drawn first, uniformly from the whole eligible pool,
   so it stays an unbiased sample for estimating base rates and model error.
2. ``missing_class_candidate``: the rule baseline predicts a class with zero or
   very few labelled rows.
3. ``reddit``: the platform that is barely represented.
4. ``unseen_group``: discussion groups absent from the labelled corpus.
5. ``model_disagreement``: rule and model disagree, or the model's calibrated
   P(actionable) sits near its decision threshold.

Rows already labelled, exact duplicates, and near-duplicates of labelled rows
are excluded. A stratum the pool cannot fill is reported as a shortfall, never
back-filled from another stratum or invented.

Quality controls:

* **Blind annotation.** The annotator file carries no rule or model output, so
  suggestions cannot anchor the label. Model outputs live in a separate
  provenance file keyed by ``queue_id``.
* **Double annotation.** A random ~15% overlap sample is flagged for a second,
  independent annotator, so inter-annotator agreement can finally be measured
  (``src.ml.agreement``).
* **Validation, not promotion.** ``validate`` checks a completed batch
  (annotator present, labels in the taxonomy, no duplicates, no overlap with
  the corpus) and writes a candidate file under ``reports/annotation/``. It
  never modifies ``reports/audits/intent_audit_corrected.csv``; merging is a
  deliberate, reviewed step.

    python -m src.ml.annotation_queue build
    python -m src.ml.annotation_queue validate --completed path/to/completed.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.intent import classify_intent
from src.ml.data import TAXONOMY, load_labelled_data
from src.ml.probabilities import normalise_text_key

LOGGER = logging.getLogger(__name__)

DEFAULT_CANONICAL_POOL = Path("data/processed/signals_scored.csv")
DEFAULT_DEMO_POOL = Path("data/demo/signals_scored_portfolio.csv")
DEFAULT_OUTPUT_DIR = Path("reports/annotation")

STRATUM_RANDOM = "random_control"
STRATUM_MISSING_CLASS = "missing_class_candidate"
STRATUM_REDDIT = "reddit"
STRATUM_UNSEEN_GROUP = "unseen_group"
STRATUM_DISAGREEMENT = "model_disagreement"

#: Selection order. Random control goes first so it is not depleted by the
#: targeted strata and remains a uniform sample of the pool.
DEFAULT_QUOTAS: dict[str, int] = {
    STRATUM_RANDOM: 25,
    STRATUM_MISSING_CLASS: 40,
    STRATUM_REDDIT: 30,
    STRATUM_UNSEEN_GROUP: 25,
    STRATUM_DISAGREEMENT: 30,
}

#: A class with fewer labelled rows than this counts as "missing" for targeting.
RARE_CLASS_THRESHOLD = 10
NEAR_DUPLICATE_THRESHOLD = 0.80
DOUBLE_ANNOTATION_SHARE = 0.15
NEAR_THRESHOLD_BAND = 0.10

ANNOTATOR_COLUMNS = [
    "queue_id",
    "text",
    "source",
    "parent_id",
    "stratum",
    "double_annotate",
    "human_label",
    "secondary_label",
    "multi_intent_flag",
    "annotator_id",
    "annotated_at",
    "review_notes",
]

PROVENANCE_COLUMNS = [
    "queue_id",
    "pool_path",
    "pool_hash",
    "pool_row",
    "source_record_id",
    "stratum",
    "stratum_reason",
    "rule_label",
    "model_label",
    "model_probability",
    "model_threshold",
    "selection_seed",
]


def _text_id(text: str) -> str:
    return hashlib.sha1(normalise_text_key(text).encode("utf-8")).hexdigest()[:12]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _group_key(row_parent, source) -> str | None:
    """Labelled groups are ``yt:<id>``/``rd:<id>``; pool files may hold bare ids."""
    if row_parent is None or (isinstance(row_parent, float) and np.isnan(row_parent)):
        return None
    value = str(row_parent)
    if ":" in value:
        return value
    prefix = {"youtube": "yt", "reddit": "rd"}.get(str(source), None)
    return f"{prefix}:{value}" if prefix else value


def rare_classes(labelled: pd.DataFrame, threshold: int = RARE_CLASS_THRESHOLD) -> list[str]:
    counts = labelled["corrected_label"].value_counts()
    return sorted(label for label in TAXONOMY if counts.get(label, 0) < threshold)


def _near_duplicate_mask(pool_texts: pd.Series, reference_texts: pd.Series) -> np.ndarray:
    """True where a pool text is a near-duplicate of any reference text."""
    if pool_texts.empty or reference_texts.empty:
        return np.zeros(len(pool_texts), dtype=bool)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    vectoriser.fit(pd.concat([pool_texts, reference_texts]).str.casefold())
    pool = vectoriser.transform(pool_texts.str.casefold())
    ref = vectoriser.transform(reference_texts.str.casefold())
    return (cosine_similarity(pool, ref) >= NEAR_DUPLICATE_THRESHOLD).any(axis=1)


def prepare_pool(pool: pd.DataFrame, labelled: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Eligible, de-duplicated candidates plus counts of what was excluded."""
    frame = pool.copy()
    frame["pool_row"] = np.arange(len(frame))
    frame["text"] = frame.get("text", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame = frame[frame["text"].str.strip() != ""]
    excluded = {"empty_text": int(len(pool) - len(frame))}

    frame["text_key"] = frame["text"].map(normalise_text_key)
    labelled_keys = set(labelled["text"].map(normalise_text_key))
    already = frame["text_key"].isin(labelled_keys)
    excluded["already_labelled"] = int(already.sum())
    frame = frame[~already]

    dup = frame["text_key"].duplicated()
    excluded["duplicate_in_pool"] = int(dup.sum())
    frame = frame[~dup]

    near = _near_duplicate_mask(frame["text"], labelled["text"].fillna("").astype(str))
    excluded["near_duplicate_of_labelled"] = int(near.sum())
    frame = frame[~near].reset_index(drop=True)

    frame["queue_id"] = frame["text"].map(_text_id)
    source = frame["source"] if "source" in frame.columns else pd.Series(pd.NA, index=frame.index)
    frame["source"] = source
    parent = frame["parent_id"] if "parent_id" in frame.columns else pd.Series(pd.NA, index=frame.index)
    frame["group_key"] = [_group_key(p, s) for p, s in zip(parent, source)]
    frame["parent_id"] = frame["group_key"]
    record_column = next((c for c in ("id", "record_id") if c in frame.columns), None)
    frame["source_record_id"] = frame[record_column].astype(str) if record_column else pd.NA
    return frame, excluded


def score_pool(frame: pd.DataFrame, model=None) -> pd.DataFrame:
    """Attach rule and (optionally) model outputs. These never become labels."""
    scored = frame.copy()
    texts = scored["text"].tolist()
    scored["rule_label"] = [classify_intent(t)[0] for t in texts]
    scored["model_label"] = pd.NA
    scored["model_probability"] = np.nan
    scored["model_threshold"] = np.nan
    if model is not None and texts:
        scored["model_label"] = model.predict(texts)
        scores = model.stage1_scores(texts)
        if scores is not None:
            scored["model_probability"] = np.asarray(scores, dtype=float)
        scored["model_threshold"] = float(model.threshold)
    return scored


def _stratum_masks(scored: pd.DataFrame, labelled: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """(eligibility mask, human-readable reason) per targeted stratum."""
    rare = set(rare_classes(labelled))
    labelled_groups = set(labelled["parent_id"].dropna().astype(str))

    missing = scored["rule_label"].isin(rare)
    reddit = scored["source"].astype(str).str.lower() == "reddit"
    unseen = scored["group_key"].notna() & ~scored["group_key"].isin(labelled_groups)

    has_model = scored["model_label"].notna()
    rule_model = has_model & (scored["rule_label"] != scored["model_label"])
    near = (scored["model_probability"] - scored["model_threshold"]).abs() < NEAR_THRESHOLD_BAND
    disagreement = rule_model | near.fillna(False)

    return {
        STRATUM_MISSING_CLASS: (missing, "rule predicts rare class " + scored["rule_label"].astype(str)),
        STRATUM_REDDIT: (reddit, pd.Series("source is reddit", index=scored.index)),
        STRATUM_UNSEEN_GROUP: (
            unseen,
            "group " + scored["group_key"].astype(str) + " not in labelled corpus",
        ),
        STRATUM_DISAGREEMENT: (
            disagreement,
            np.where(rule_model, "rule/model label disagreement", "near decision threshold"),
        ),
    }


def select_queue(
    scored: pd.DataFrame,
    labelled: pd.DataFrame,
    quotas: dict[str, int] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Fill each stratum's quota in order; report shortfalls instead of back-filling."""
    quotas = dict(quotas or DEFAULT_QUOTAS)
    rng = np.random.default_rng(seed)
    masks = _stratum_masks(scored, labelled)
    taken = pd.Series(False, index=scored.index)
    parts: list[pd.DataFrame] = []
    report: dict[str, dict] = {}

    for stratum, quota in quotas.items():
        if stratum == STRATUM_RANDOM:
            eligible = ~taken
            reason = pd.Series("uniform random sample of the eligible pool", index=scored.index)
        else:
            mask, reason = masks[stratum]
            reason = pd.Series(reason, index=scored.index)
            eligible = mask & ~taken
        candidates = scored.index[eligible.to_numpy()]
        if stratum == STRATUM_DISAGREEMENT and len(candidates):
            # Most uncertain first; ties broken randomly.
            distance = (
                scored.loc[candidates, "model_probability"] - scored.loc[candidates, "model_threshold"]
            ).abs().fillna(1.0)
            jitter = rng.random(len(candidates)) * 1e-6
            order = np.argsort(distance.to_numpy() + jitter)
            chosen = candidates[order[:quota]]
        else:
            chosen = rng.permutation(candidates)[:quota]
        part = scored.loc[chosen].copy()
        part["stratum"] = stratum
        part["stratum_reason"] = reason.loc[chosen].to_numpy()
        parts.append(part)
        taken.loc[chosen] = True
        report[stratum] = {
            "quota": int(quota),
            "eligible": int(len(candidates)),
            "selected": int(len(chosen)),
            "shortfall": int(max(0, quota - len(chosen))),
        }

    queue = pd.concat(parts, ignore_index=True) if parts else scored.iloc[0:0].copy()
    n_double = int(round(len(queue) * DOUBLE_ANNOTATION_SHARE))
    double = np.zeros(len(queue), dtype=bool)
    if n_double:
        double[rng.choice(len(queue), size=n_double, replace=False)] = True
    queue["double_annotate"] = double
    queue["selection_seed"] = seed
    return queue, report


def load_model_if_available(train_if_missing: bool = True):
    """The persisted hierarchical model, or None (disagreement stratum then rule-only)."""
    from src.ml.probabilities import ProbabilityUnavailableError, ensure_hierarchical_model

    try:
        model, metadata = ensure_hierarchical_model(train_if_missing=train_if_missing)
    except (ProbabilityUnavailableError, Exception) as exc:  # noqa: BLE001
        LOGGER.warning("No model available for the disagreement stratum: %s", exc)
        return None, {}
    return model, metadata


def build_annotation_queue(
    pool_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    quotas: dict[str, int] | None = None,
    seed: int = 42,
    model=None,
    model_metadata: dict | None = None,
) -> dict:
    """Build and write the blind queue, its provenance file and a manifest."""
    if pool_path is None:
        pool_path = DEFAULT_CANONICAL_POOL if DEFAULT_CANONICAL_POOL.exists() else DEFAULT_DEMO_POOL
    pool_path = Path(pool_path)
    pool_kind = "canonical" if pool_path == DEFAULT_CANONICAL_POOL else (
        "demo_extract" if pool_path == DEFAULT_DEMO_POOL else "custom"
    )
    pool = pd.read_csv(pool_path)
    labelled = load_labelled_data()

    frame, excluded = prepare_pool(pool, labelled)
    scored = score_pool(frame, model)
    queue, strata = select_queue(scored, labelled, quotas, seed)
    queue["pool_path"] = str(pool_path)
    queue["pool_hash"] = _file_hash(pool_path)

    for column in ["human_label", "secondary_label", "multi_intent_flag", "annotator_id",
                   "annotated_at", "review_notes"]:
        queue[column] = ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queue[ANNOTATOR_COLUMNS].to_csv(output_dir / "annotation_queue.csv", index=False)
    queue[PROVENANCE_COLUMNS].to_csv(output_dir / "annotation_queue_provenance.csv", index=False)

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pool_path": str(pool_path),
        "pool_kind": pool_kind,
        "pool_hash": _file_hash(pool_path),
        "pool_rows": int(len(pool)),
        "eligible_rows": int(len(frame)),
        "excluded": excluded,
        "rare_classes_targeted": rare_classes(labelled),
        "strata": strata,
        "queue_rows": int(len(queue)),
        "double_annotation_rows": int(queue["double_annotate"].sum()),
        "seed": seed,
        "model": {
            "available": model is not None,
            "model_name": (model_metadata or {}).get("model_name"),
            "dataset_hash": (model_metadata or {}).get("dataset_hash"),
            "threshold": float(model.threshold) if model is not None else None,
        },
        "label_status": "pending_human_review",
        "notes": [
            "No label in this queue is filled in. Rule and model outputs are kept in "
            "annotation_queue_provenance.csv and must not be shown to annotators.",
            "Completed batches are validated with `python -m src.ml.annotation_queue "
            "validate`; they are never merged into the labelled corpus automatically.",
        ],
    }
    if pool_kind == "demo_extract":
        manifest["notes"].append(
            "Built from the 304-row demo extract because the canonical signal set is "
            "absent. This pool is small and YouTube-heavy, so most strata cannot be "
            "filled; rebuild from data/processed/signals_scored.csv for the real batch."
        )
    (output_dir / "annotation_queue_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    return {"queue": queue, "manifest": manifest}


def validate_completed_batch(
    completed: pd.DataFrame,
    provenance: pd.DataFrame,
    labelled: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Check a completed annotation batch; return accepted rows and a QC report.

    Accepted rows keep their queue provenance and are stamped with a new
    annotation version. Nothing is written to the existing corpus.
    """
    frame = completed.copy()
    for column in ["human_label", "annotator_id", "secondary_label"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
        else:
            frame[column] = ""
    problems: dict[str, int] = {}

    labelled_rows = frame["human_label"] != ""
    problems["unlabelled_rows"] = int((~labelled_rows).sum())
    frame = frame[labelled_rows]

    no_annotator = frame["annotator_id"] == ""
    problems["missing_annotator_id"] = int(no_annotator.sum())

    bad_label = ~frame["human_label"].isin(TAXONOMY)
    problems["label_not_in_taxonomy"] = int(bad_label.sum())

    bad_secondary = (frame["secondary_label"] != "") & ~frame["secondary_label"].isin(TAXONOMY)
    problems["secondary_not_in_taxonomy"] = int(bad_secondary.sum())

    unknown_queue = ~frame["queue_id"].isin(set(provenance["queue_id"]))
    problems["unknown_queue_id"] = int(unknown_queue.sum())

    labelled_keys = set(labelled["text"].map(normalise_text_key))
    overlap = frame["text"].map(normalise_text_key).isin(labelled_keys)
    problems["already_in_corpus"] = int(overlap.sum())

    duplicate = frame.duplicated(["queue_id", "annotator_id"], keep=False)
    problems["duplicate_queue_id_per_annotator"] = int(duplicate.sum())

    reject = no_annotator | bad_label | bad_secondary | unknown_queue | overlap | duplicate
    accepted = frame[~reject].merge(
        provenance.drop(columns=["stratum"], errors="ignore"), on="queue_id", how="left"
    )
    accepted["annotation_version"] = "expansion_" + dt.date.today().strftime("%Y%m%d")
    accepted["label_status"] = "human_reviewed_unmerged"

    agreement = None
    multi = accepted.groupby("queue_id")["annotator_id"].nunique()
    shared = multi[multi >= 2].index
    if len(shared):
        from src.ml.agreement import cohens_kappa, raw_agreement

        pairs = (
            accepted[accepted["queue_id"].isin(shared)]
            .sort_values(["queue_id", "annotator_id"])
            .groupby("queue_id")["human_label"]
            .apply(list)
        )
        first = [labels[0] for labels in pairs]
        second = [labels[1] for labels in pairs]
        agreement = {
            "items": len(pairs),
            "raw_agreement": raw_agreement(first, second),
            "cohens_kappa": cohens_kappa(first, second) if len(set(first + second)) > 1 else None,
        }

    report = {
        "rows_received": int(len(completed)),
        "rows_accepted": int(len(accepted)),
        "problems": problems,
        "accepted_label_counts": accepted["human_label"].value_counts().to_dict(),
        "inter_annotator_agreement": agreement,
        "note": (
            "Accepted rows are human-reviewed but NOT merged into the labelled corpus. "
            "Merging requires an explicit, reviewed change to the corpus file."
        ),
    }
    return accepted, report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Annotation queue and QC")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build the stratified, blind annotation queue")
    build.add_argument("--pool", type=Path, default=None)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--no-model", action="store_true", help="Rule-only disagreement stratum")
    validate = sub.add_parser("validate", help="QC a completed annotation batch")
    validate.add_argument("--completed", type=Path, required=True)
    validate.add_argument(
        "--provenance", type=Path, default=DEFAULT_OUTPUT_DIR / "annotation_queue_provenance.csv"
    )
    validate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "validated")
    args = parser.parse_args()

    if args.command == "build":
        model, metadata = (None, {}) if args.no_model else load_model_if_available()
        result = build_annotation_queue(
            pool_path=args.pool, output_dir=args.output_dir, seed=args.seed,
            model=model, model_metadata=metadata,
        )
        manifest = result["manifest"]
        print(f"Pool: {manifest['pool_path']} ({manifest['pool_kind']}), "
              f"{manifest['eligible_rows']} eligible of {manifest['pool_rows']}")
        for stratum, info in manifest["strata"].items():
            print(f"- {stratum}: {info['selected']}/{info['quota']} "
                  f"(eligible {info['eligible']}, shortfall {info['shortfall']})")
        return

    completed = pd.read_csv(args.completed)
    provenance = pd.read_csv(args.provenance)
    accepted, report = validate_completed_batch(completed, provenance, load_labelled_data())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.completed.stem
    accepted.to_csv(args.output_dir / f"{stem}_accepted.csv", index=False)
    (args.output_dir / f"{stem}_qc.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Independent double annotation, disagreement analysis and adjudication.

Implements the workflow in ``docs/ANNOTATION_PROTOCOL.md``. Three steps, each
reading and writing files under ``reports/annotation/``; none of them ever
writes ``reports/audits/intent_audit_corrected.csv``.

1. ``agreement``: two annotators' completed copies of the *same* queue are
   compared. It reports raw agreement and Cohen's kappa for the 8-class label
   and for the binary actionable target, per-class one-vs-rest kappa, the
   confusion matrix, and prevalence-adjusted (PABAK) agreement, with
   item-bootstrap intervals. It also writes an adjudication queue of every
   disagreement. The two labels are shown in random order and without
   annotator IDs, so the adjudicator cannot side with a person.
2. The adjudicator (a third person) fills ``adjudicated_label`` and
   ``rationale``.
3. ``finalise``: agreed items plus adjudicated items become a gold *candidate*
   file with full provenance (``label_source`` = agreed | adjudicated). It
   refuses an adjudicator who was one of the two annotators, adjudications
   without a rationale, and labels outside the taxonomy.

    python -m src.ml.adjudication agreement --a annotator_a.csv --b annotator_b.csv
    python -m src.ml.adjudication finalise --a ... --b ... --adjudicated adjudication_queue.csv

No label is ever generated here. Model or rule outputs are never inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.ml.data import TAXONOMY, is_actionable

DEFAULT_OUTPUT_DIR = Path("reports/annotation/adjudication")
#: Below this kappa a class definition is treated as unreliable and goes back
#: to the guide before more labels are collected (see the protocol).
KAPPA_REVIEW_THRESHOLD = 0.60


def _load_annotations(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"queue_id", "human_label", "annotator_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Annotator file {name} lacks columns {sorted(missing)}")
    out = frame.copy()
    for column in ("queue_id", "annotator_id"):
        out[column] = out[column].fillna("").astype(str).str.strip()
    out["human_label"] = out["human_label"].fillna("").astype(str).str.strip()
    out = out[out["human_label"] != ""]
    if out.empty or out[["queue_id", "annotator_id"]].eq("").any().any():
        raise ValueError(f"{name}: completed rows need nonempty queue_id and annotator_id")
    bad = ~out["human_label"].isin(TAXONOMY)
    if bad.any():
        raise ValueError(
            f"{name}: {int(bad.sum())} labels outside the taxonomy: "
            f"{sorted(out.loc[bad, 'human_label'].unique())}"
        )
    if out["queue_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate queue_id rows")
    annotators = out["annotator_id"].dropna().astype(str).unique()
    if len(annotators) != 1:
        raise ValueError(f"{name}: expected exactly one annotator_id, found {list(annotators)}")
    return out


def pair_annotations(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Items labelled by both annotators, with an independence check."""
    a = _load_annotations(a, "annotator A")
    b = _load_annotations(b, "annotator B")
    if a["annotator_id"].iloc[0] == b["annotator_id"].iloc[0]:
        raise ValueError("Both files carry the same annotator_id; agreement needs two people")
    keep = ["queue_id", "human_label", "annotator_id"]
    extra = [c for c in ("text", "parent_id", "source", "stratum") if c in a.columns]
    paired = a[keep + extra].merge(b[keep], on="queue_id", suffixes=("_a", "_b"))
    if paired.empty:
        raise ValueError("No shared completed queue items between annotators")
    if "text" in a and "text" in b:
        joined = a[["queue_id", "text"]].merge(b[["queue_id", "text"]], on="queue_id")
        if not joined["text_x"].equals(joined["text_y"]):
            raise ValueError("Shared queue IDs refer to different text")
    return paired


def _pabak(x: np.ndarray, y: np.ndarray) -> float:
    """Prevalence-adjusted bias-adjusted kappa for a binary label."""
    return float(2 * np.mean(x == y) - 1)


def _kappa(x, y) -> float:
    if len(set(x) | set(y)) < 2:
        return float("nan")
    return float(cohen_kappa_score(x, y))


def agreement_report(paired: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    a = paired["human_label_a"].to_numpy()
    b = paired["human_label_b"].to_numpy()
    act_a = np.array([is_actionable(x) for x in a])
    act_b = np.array([is_actionable(x) for x in b])

    rng = np.random.default_rng(seed)
    boot = {"kappa": [], "actionable_kappa": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        boot["kappa"].append(_kappa(a[idx], b[idx]))
        boot["actionable_kappa"].append(_kappa(act_a[idx], act_b[idx]))

    def ci(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if len(values) else None

    per_class = []
    for label in TAXONOMY:
        xa, xb = a == label, b == label
        support = int(xa.sum() + xb.sum())
        if not support:
            continue
        k = _kappa(xa, xb)
        per_class.append(
            {
                "class": label,
                "labelled_by_a": int(xa.sum()),
                "labelled_by_b": int(xb.sum()),
                "both": int((xa & xb).sum()),
                "kappa_one_vs_rest": k,
                "needs_guide_review": bool(np.isfinite(k) and k < KAPPA_REVIEW_THRESHOLD),
            }
        )
    labels = [c for c in TAXONOMY if (a == c).any() or (b == c).any()]
    confusion = pd.crosstab(
        pd.Categorical(a, categories=labels), pd.Categorical(b, categories=labels),
        rownames=["annotator_a"], colnames=["annotator_b"], dropna=False,
    )
    disagreements = (
        pd.DataFrame({"a": a, "b": b})[a != b]
        .apply(lambda r: " vs ".join(sorted([r["a"], r["b"]])), axis=1)
        .value_counts()
    )
    return {
        "items": int(len(a)),
        "annotators": [str(paired["annotator_id_a"].iloc[0]), str(paired["annotator_id_b"].iloc[0])],
        "raw_agreement": float(np.mean(a == b)),
        "kappa": _kappa(a, b),
        "kappa_95": ci(boot["kappa"]),
        "actionable_raw_agreement": float(np.mean(act_a == act_b)),
        "actionable_kappa": _kappa(act_a, act_b),
        "actionable_kappa_95": ci(boot["actionable_kappa"]),
        "actionable_pabak": _pabak(act_a, act_b),
        "per_class": per_class,
        "confusion": confusion.to_dict(),
        "most_common_disagreements": disagreements.head(10).to_dict(),
        "disagreement_items": int((a != b).sum()),
        "kappa_review_threshold": KAPPA_REVIEW_THRESHOLD,
    }


def adjudication_queue(paired: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Disagreements with both labels in random order and no annotator identity."""
    rng = np.random.default_rng(seed)
    disputed = paired[paired["human_label_a"] != paired["human_label_b"]].copy()
    flip = rng.random(len(disputed)) < 0.5
    first = np.where(flip, disputed["human_label_b"], disputed["human_label_a"])
    second = np.where(flip, disputed["human_label_a"], disputed["human_label_b"])
    queue = pd.DataFrame(
        {
            "queue_id": disputed["queue_id"].to_numpy(),
            "text": disputed["text"].to_numpy() if "text" in disputed else "",
            "candidate_label_1": first,
            "candidate_label_2": second,
            "adjudicated_label": "",
            "adjudicator_id": "",
            "rationale": "",
        }
    )
    return queue


def finalise(paired: pd.DataFrame, adjudicated: pd.DataFrame, version: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Gold candidate rows: agreed items plus independently adjudicated ones."""
    annotators = {str(paired["annotator_id_a"].iloc[0]), str(paired["annotator_id_b"].iloc[0])}
    version = version or "gold_candidate_" + dt.date.today().strftime("%Y%m%d")

    agreed = paired[paired["human_label_a"] == paired["human_label_b"]].copy()
    agreed["label"] = agreed["human_label_a"]
    agreed["label_source"] = "agreed"
    agreed["adjudicator_id"] = ""
    agreed["rationale"] = ""

    adj = adjudicated.copy()
    if adj["queue_id"].duplicated().any():
        raise ValueError("Duplicate adjudication queue_id rows")
    disputed_ids = set(paired.loc[paired["human_label_a"] != paired["human_label_b"], "queue_id"])
    if not set(adj["queue_id"]).issubset(disputed_ids):
        raise ValueError("Adjudication contains unknown or already agreed queue IDs")
    for column in ("adjudicated_label", "adjudicator_id", "rationale"):
        adj[column] = adj[column].fillna("").astype(str).str.strip()
    problems = {
        "unadjudicated": int((adj["adjudicated_label"] == "").sum()),
        "label_not_in_taxonomy": int(
            ((adj["adjudicated_label"] != "") & ~adj["adjudicated_label"].isin(TAXONOMY)).sum()
        ),
        "missing_rationale": int(((adj["adjudicated_label"] != "") & (adj["rationale"] == "")).sum()),
        "adjudicator_was_an_annotator": int(adj["adjudicator_id"].isin(annotators).sum()),
        "missing_adjudicator_id": int(((adj["adjudicated_label"] != "") & (adj["adjudicator_id"] == "")).sum()),
    }
    if problems["adjudicator_was_an_annotator"]:
        raise ValueError(
            "Adjudication must be independent: an adjudicator_id matches one of the annotators "
            f"{sorted(annotators)}"
        )
    valid = (
        adj["adjudicated_label"].isin(TAXONOMY)
        & (adj["rationale"] != "")
        & (adj["adjudicator_id"] != "")
    )
    disputed = paired.set_index("queue_id")
    resolved = adj[valid & adj["queue_id"].isin(disputed.index)].copy()
    resolved = resolved.join(disputed, on="queue_id", rsuffix="_paired")
    resolved["label"] = resolved["adjudicated_label"]
    resolved["label_source"] = "adjudicated"

    columns = ["queue_id", "label", "label_source", "human_label_a", "human_label_b",
               "annotator_id_a", "annotator_id_b", "adjudicator_id", "rationale"]
    gold = pd.concat([agreed[columns], resolved[columns]], ignore_index=True)
    gold["annotation_version"] = version
    gold["label_status"] = "gold_candidate_unmerged"
    report = {
        "items_paired": int(len(paired)),
        "agreed": int(len(agreed)),
        "adjudicated": int(len(resolved)),
        "still_disputed": int(len(paired) - len(agreed) - len(resolved)),
        "problems": problems,
        "annotation_version": version,
        "note": (
            "Gold candidates are not merged into the labelled corpus. Merging is a reviewed "
            "change with a new annotation_version."
        ),
    }
    return gold, report


def write_agreement_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Inter-annotator agreement",
        "",
        f"- Items annotated by both: {report['items']} (annotators {', '.join(report['annotators'])})",
        f"- 8-class raw agreement {report['raw_agreement']:.2f}; Cohen's κ {report['kappa']:.2f} "
        f"(95% item bootstrap {report['kappa_95']})",
        f"- Actionable vs general: raw {report['actionable_raw_agreement']:.2f}; κ "
        f"{report['actionable_kappa']:.2f} ({report['actionable_kappa_95']}); PABAK "
        f"{report['actionable_pabak']:.2f}",
        f"- Items sent to adjudication: {report['disagreement_items']}",
        "",
        "| Class | A | B | Both | κ (one-vs-rest) | Needs guide review |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["per_class"]:
        k = row["kappa_one_vs_rest"]
        lines.append(
            f"| {row['class']} | {row['labelled_by_a']} | {row['labelled_by_b']} | {row['both']} | "
            f"{'—' if not np.isfinite(k) else f'{k:.2f}'} | {'yes' if row['needs_guide_review'] else 'no'} |"
        )
    lines += ["", "Most frequent disagreements:", ""]
    lines += [f"- {pair}: {count}" for pair, count in report["most_common_disagreements"].items()]
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Double annotation and adjudication")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("agreement", "finalise"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--a", type=Path, required=True, help="Annotator A's completed queue")
        cmd.add_argument("--b", type=Path, required=True, help="Annotator B's completed queue")
        cmd.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        if name == "finalise":
            cmd.add_argument("--adjudicated", type=Path, required=True)
            cmd.add_argument("--version", default=None)
    args = parser.parse_args()
    paired = pair_annotations(pd.read_csv(args.a), pd.read_csv(args.b))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "agreement":
        report = agreement_report(paired)
        (args.output_dir / "agreement.json").write_text(json.dumps(report, indent=2, default=str))
        write_agreement_markdown(report, args.output_dir / "agreement.md")
        adjudication_queue(paired).to_csv(args.output_dir / "adjudication_queue.csv", index=False)
        print(f"κ {report['kappa']:.2f}; {report['disagreement_items']} items to adjudicate "
              f"-> {args.output_dir}")
        return
    gold, report = finalise(paired, pd.read_csv(args.adjudicated), args.version)
    gold.to_csv(args.output_dir / f"{report['annotation_version']}.csv", index=False)
    (args.output_dir / "finalise_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

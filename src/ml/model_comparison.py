"""Leakage-safe model comparison with paired uncertainty and error analysis.

Builds on ``src.ml.group_cv``: every model is scored out of fold on the *same*
group folds (whole discussion groups held out; calibration and threshold fitted
on an inner group-held-out validation split), so models can be compared row
for row. On top of that this module adds:

* **Paired group-bootstrap differences.** Discussion groups are resampled once
  per replicate and every model is scored on the same resample, averaged over
  fold seeds. A model is reported as better than another only when the 95%
  interval of the difference excludes zero; otherwise the report says the data
  cannot separate them.
* **Ranking at fixed review budgets.** Precision@k, Recall@k and Lift@k for the
  binary actionable-vs-general relevance target, which is the formulation a
  review queue actually uses. Ties (the rule baseline has two score levels)
  are broken in expectation, never by row order. Intent-level ranking metrics
  are not reported: there is no single relevance order over eight classes.
* **Inspectable errors.** Every false positive and false negative with its
  text, group, rule label and probability; the errors all supervised models
  share; confusion matrices; and an empty ``error_category`` column for human
  review (categories are never auto-assigned).

    python -m src.ml.model_comparison                     # classical models
    python -m src.ml.model_comparison --with-embedding    # + frozen MiniLM encoder

Status stays ``development_preliminary``: these are cross-validated
development results on the 200-row corpus the models were built around.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix

from src.ml import EVALUATION_STATUS_DEVELOPMENT
from src.ml.data import TAXONOMY, dataset_hash, load_labelled_data
from src.ml.group_cv import run_group_cv
from src.ml.ranking import expected_positives_in_top_k

DEFAULT_OUTPUT_DIR = Path("reports/ml/model_comparison")
DEFAULT_REPORT_PATH = Path("reports/ml/evaluation_report.md")
CLASSICAL_MODELS = ("majority", "rules", "logistic", "linear_svm", "char_word_svm")
REFERENCE_MODEL = "rules"
REVIEW_BUDGETS = (10, 20, 30, 60)

#: Population share of each *rule-predicted* label in the full signal set at the
#: time the audit sample was drawn, read from the committed figure
#: ``reports/figures/intent_distribution.png`` (2026-04-27; one-decimal
#: rounding). The audit sample is stratified on exactly this label: 100 rows
#: where the rule said general_discussion and up to 20 per other rule label
#: (``scripts/sample_intent_audit.py``). These shares let a design-weighted
#: (inverse-probability) estimate undo that stratification approximately.
RULE_LABEL_POPULATION_SHARE = {
    "general_discussion": 0.884,
    "nostalgia_reactivation": 0.065,
    "frustrated_demand": 0.025,
    "new_customer_interest": 0.010,
    "high_intent": 0.010,
    "confusion_barrier": 0.006,
}
DESIGN_SOURCE = "reports/figures/intent_distribution.png (2026-04-27)"
N_BOOTSTRAP = 1000
SUPERVISED = ("logistic", "linear_svm", "char_word_svm", "embedding")


def _ranking_at(y: np.ndarray, p: np.ndarray, k: int) -> dict:
    hits = expected_positives_in_top_k(y, p, k)
    positives = max(int(y.sum()), 1)
    base = float(y.mean()) or 1.0
    return {"precision": hits / k, "recall": hits / positives, "lift": (hits / k) / base}


def design_weights(labelled: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
    """Inverse-probability weights per record for the rule-stratified sample.

    weight = population share of the row's sampling stratum / rows sampled from
    it, normalised to mean 1. Strata are the rule label recorded at sampling
    time (``intent_label`` in the raw audit file), not today's rule output.
    """
    strata = raw.set_index(raw["id"].astype(str))["intent_label"]
    counts = strata.value_counts()
    weights = strata.map(lambda s: RULE_LABEL_POPULATION_SHARE.get(s, np.nan) / counts[s])
    weights = weights / weights.mean()
    return weights.reindex(labelled["record_id"].astype(str))


def _metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict:
    out = {
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    if w is not None:
        out["weighted_pr_auc"] = float(average_precision_score(y, p, sample_weight=w))
        out["weighted_brier"] = float(brier_score_loss(y, p, sample_weight=w))
    for k in REVIEW_BUDGETS:
        if k <= len(y):
            out[f"lift_at_{k}"] = _ranking_at(y, p, k)["lift"]
    return out


def _wide(oof: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Per seed: one row per record, a probability column per model."""
    frames = {}
    for seed, group in oof.groupby("seed"):
        wide = group.pivot_table(index="record_id", columns="model", values="p_actionable")
        columns = ["parent_id", "true_actionable"] + (["weight"] if "weight" in group else [])
        meta = group.drop_duplicates("record_id").set_index("record_id")[columns]
        frames[int(seed)] = wide.join(meta).dropna()
    return frames


def _weights(frame: pd.DataFrame, idx=None):
    if "weight" not in frame:
        return None
    w = frame["weight"].to_numpy(dtype=float)
    return w if idx is None else w[idx]


def paired_bootstrap(oof: pd.DataFrame, models, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Group-bootstrap distributions of each metric per model, seed-averaged."""
    frames = _wide(oof)
    groups = sorted(next(iter(frames.values()))["parent_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    draws: dict[str, dict[str, list[float]]] = {m: {} for m in models}
    index_by_seed = {
        s: {g: np.flatnonzero(f["parent_id"].astype(str).to_numpy() == g) for g in groups}
        for s, f in frames.items()
    }
    for _ in range(n_boot):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        per_model: dict[str, list[dict]] = {m: [] for m in models}
        for s, frame in frames.items():
            idx = np.concatenate([index_by_seed[s][g] for g in chosen])
            y = frame["true_actionable"].to_numpy(dtype=int)[idx]
            if len(np.unique(y)) < 2:
                continue
            w = _weights(frame, idx)
            for m in models:
                per_model[m].append(_metrics(y, frame[m].to_numpy(dtype=float)[idx], w))
        for m in models:
            if not per_model[m]:
                continue
            averaged = pd.DataFrame(per_model[m]).mean().to_dict()
            for key, value in averaged.items():
                draws[m].setdefault(key, []).append(value)
    return {m: {k: np.asarray(v) for k, v in d.items()} for m, d in draws.items()}


def point_estimates(oof: pd.DataFrame, models) -> pd.DataFrame:
    rows = []
    for m in models:
        per_seed = []
        for _, frame in _wide(oof).items():
            y = frame["true_actionable"].to_numpy(dtype=int)
            p = frame[m].to_numpy(dtype=float)
            values = _metrics(y, p, _weights(frame))
            for k in REVIEW_BUDGETS:
                ranking = _ranking_at(y, p, k)
                values[f"precision_at_{k}"] = ranking["precision"]
                values[f"recall_at_{k}"] = ranking["recall"]
            per_seed.append(values)
        rows.append({"model": m, **pd.DataFrame(per_seed).mean().to_dict()})
    return pd.DataFrame(rows).set_index("model")


def compare_pairs(draws: dict, models,
                  metrics=("pr_auc", "lift_at_20", "brier", "weighted_pr_auc")) -> pd.DataFrame:
    """Paired differences (row model minus reference) with 95% intervals."""
    rows = []
    references = [REFERENCE_MODEL] + [m for m in models if m in SUPERVISED]
    seen = set()
    for reference in references:
        for model in models:
            if model == reference or (reference, model) in seen or (model, reference) in seen:
                continue
            if reference != REFERENCE_MODEL and model not in SUPERVISED:
                continue
            seen.add((model, reference))
            for metric in metrics:
                a, b = draws[model].get(metric), draws[reference].get(metric)
                if a is None or b is None or not len(a):
                    continue
                diff = a - b
                low, high = np.quantile(diff, [0.025, 0.975])
                better_is_lower = metric.endswith("brier")
                if low > 0 or high < 0:
                    favoured = model if (high < 0) == better_is_lower else reference
                    verdict = f"{favoured} better (interval excludes 0)"
                else:
                    verdict = "not distinguishable"
                rows.append(
                    {
                        "model": model,
                        "versus": reference,
                        "metric": metric,
                        "mean_difference": float(diff.mean()),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "verdict": verdict,
                    }
                )
    return pd.DataFrame(rows)


def error_table(oof: pd.DataFrame, labelled: pd.DataFrame, seed: int, models) -> pd.DataFrame:
    """Every Stage 1 error on the reference seed, with text and context."""
    ref = oof[oof["seed"] == seed]
    texts = labelled.set_index("record_id")["text"]
    rows = []
    for m in models:
        part = ref[ref["model"] == m].copy()
        part["pred_actionable"] = part["pred_actionable"].astype(int)
        wrong = part[part["pred_actionable"] != part["true_actionable"]].copy()
        wrong["error_type"] = np.where(wrong["true_actionable"] == 1, "false_negative", "false_positive")
        rows.append(wrong)
    errors = pd.concat(rows, ignore_index=True)
    errors["text"] = errors["record_id"].map(texts)
    supervised = [m for m in models if m in SUPERVISED]
    shared = (
        errors[errors["model"].isin(supervised)]
        .groupby(["record_id", "error_type"])["model"]
        .nunique()
    )
    errors["supervised_models_making_this_error"] = [
        int(shared.get((r, e), 0)) for r, e in zip(errors["record_id"], errors["error_type"])
    ]
    errors["words"] = errors["text"].astype(str).str.split().map(len)
    errors["error_category"] = ""  # filled in by a human reviewer, never auto-assigned
    columns = [
        "model", "error_type", "record_id", "parent_id", "source", "sample_strategy",
        "corrected_label", "rule_label", "pred_label", "p_actionable", "threshold", "words",
        "supervised_models_making_this_error", "error_category", "text",
    ]
    return errors[columns].sort_values(["model", "error_type", "p_actionable"], ascending=[True, True, False])


def confusion_tables(oof: pd.DataFrame, seed: int, models) -> dict[str, pd.DataFrame]:
    ref = oof[oof["seed"] == seed]
    labels = [c for c in TAXONOMY if (ref["corrected_label"] == c).any()]
    tables = {}
    for m in models:
        part = ref[ref["model"] == m]
        pred_labels = labels + sorted(set(part["pred_label"].astype(str)) - set(labels))
        matrix = confusion_matrix(part["corrected_label"], part["pred_label"].astype(str), labels=pred_labels)
        tables[m] = pd.DataFrame(matrix, index=pred_labels, columns=pred_labels).loc[labels]
    return tables


def _fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def _ci(draws, model, metric, digits=2) -> str:
    values = draws.get(model, {}).get(metric)
    if values is None or not len(values):
        return "—"
    low, high = np.quantile(values, [0.025, 0.975])
    return f"{low:.{digits}f}–{high:.{digits}f}"


def _md_table(frame: pd.DataFrame) -> list[str]:
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
             "|" + "---|" * len(frame.columns)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return lines


def _excerpt(text: str, n: int = 140) -> str:
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= n else text[: n - 1] + "…"


def write_report(points, draws, pairs, per_class, errors, confusion, meta, path: Path) -> None:
    models = list(points.index)
    base_rate = meta["base_rate"]
    lines = [
        "# Evaluation report: actionable-signal detection",
        "",
        "> **Executed, development-preliminary results.** Every number below was produced "
        "by `python -m src.ml.model_comparison"
        + (" --with-embedding" if "embedding" in models else "")
        + f"` on {meta['generated_at'][:10]} over the {meta['rows']}-row single-annotator "
        f"audit corpus ({meta['groups']} discussion groups, dataset hash `{meta['dataset_hash']}`). "
        "It is cross-validated development evaluation, not final held-out performance: the "
        "taxonomy, rules and model families were designed around these rows.",
        "",
        "## Protocol",
        "",
        f"- {meta['n_folds']}-fold **group** cross-validation (whole YouTube videos / Reddit "
        f"threads held out), repeated over fold seeds {meta['seeds']}; every model sees "
        "identical folds.",
        "- Inside each training fold, a group-held-out validation split fits Platt "
        "calibration and the Stage 1 threshold; the held-out fold is touched only for scoring.",
        "- TF-IDF vocabularies and model weights are fitted per fold (scikit-learn pipelines).",
        f"- Uncertainty: {meta['n_bootstrap']} paired group-bootstrap replicates (groups "
        "resampled once per replicate, all models scored on the same resample, averaged over "
        "fold seeds).",
        f"- Relevance target for ranking metrics: actionable vs general discussion "
        f"(base rate {base_rate:.2f}, {meta['positives']} actionable rows). Ties are broken in "
        "expectation.",
    ]
    if "embedding" in models:
        lines.append(
            f"- `embedding`: frozen `{meta['embedding_model']}` (revision "
            f"`{meta['embedding_revision'][:7]}`) sentence embeddings + class-balanced logistic "
            "head. The encoder is not fine-tuned."
        )
    lines += [
        "",
        "## Headline: Stage 1 ranking and calibration",
        "",
        "Mean over fold seeds; 95% group-bootstrap interval in brackets. PR-AUC of a random "
        f"ranking equals the base rate ({base_rate:.2f}); Lift@k of random review is 1.",
        "",
        "| Model | PR-AUC | Lift@20 (top 10%) | Lift@60 (= #positives) | Brier | ECE |",
        "|---|---|---|---|---|---|",
    ]
    for m in models:
        row = points.loc[m]
        lines.append(
            f"| {m} | {_fmt(row['pr_auc'])} [{_ci(draws, m, 'pr_auc')}] | "
            f"{_fmt(row['lift_at_20'], 2)} [{_ci(draws, m, 'lift_at_20')}] | "
            f"{_fmt(row['lift_at_60'], 2)} [{_ci(draws, m, 'lift_at_60')}] | "
            f"{_fmt(row['brier'])} [{_ci(draws, m, 'brier', 3)}] | {_fmt(meta['ece'].get(m))} |"
        )
    lines += [
        "",
        "## Sampling design and what it does to these numbers",
        "",
        "The audit corpus is **stratified on the rule baseline's own prediction**: 100 rows "
        "where the rule said `general_discussion` and 20 rows for each class the rule "
        "predicted (`scripts/sample_intent_audit.py`). Two consequences:",
        "",
        "- Rule recall on a class is close to 1 *by construction*: a row of that class only "
        "enters the sample without the rule firing if it comes from the 100-row "
        "`general_discussion` stratum. The rule's class-wise F1 below is therefore inflated, "
        "and rules-vs-model comparisons are conditional on this design.",
        f"- Actionable rows are over-represented (sample base rate {base_rate:.2f}).",
        "",
        "**Design-weighted sensitivity analysis (approximate).** Weighting each row by its "
        "stratum's population share over its sample count undoes the stratification, using "
        f"rule-label shares read from `{meta['design_source']}`, rounded to 0.1%. The rule set "
        "may have changed since that chart was made. Estimated population prevalence of "
        f"actionable comments: **{meta['weighted_base_rate']:.2f}**. Effective sample size "
        f"{meta['effective_sample_size']:.0f} of {meta['rows']}.",
        "",
        "| Model | Weighted PR-AUC [95%] | Weighted Brier [95%] |",
        "|---|---|---|",
    ]
    for m in models:
        row = points.loc[m]
        lines.append(
            f"| {m} | {_fmt(row['weighted_pr_auc'])} [{_ci(draws, m, 'weighted_pr_auc')}] | "
            f"{_fmt(row['weighted_brier'])} [{_ci(draws, m, 'weighted_brier', 3)}] |"
        )
    lines += [
        "",
        f"A random ranking's weighted PR-AUC is about {meta['weighted_base_rate']:.2f}. These "
        "weighted figures are the closer guide to performance on unselected comments, but "
        "they rest on rounded chart values and a small effective sample.",
        "",
        "## Which differences are supported",
        "",
        "Paired differences (row model minus comparison model). A difference is called only "
        "when its 95% interval excludes zero; for Brier lower is better.",
        "",
    ]
    show = pairs.copy()
    for col in ("mean_difference", "ci_low", "ci_high"):
        show[col] = show[col].map(lambda v: f"{v:+.3f}")
    lines += _md_table(show[["model", "versus", "metric", "mean_difference", "ci_low", "ci_high", "verdict"]])
    supported = pairs[pairs["verdict"] != "not distinguishable"]
    lines += [
        "",
        f"{len(supported)} of {len(pairs)} paired comparisons are supported at this sample size. "
        "Anything marked *not distinguishable* is not evidence that one model is better.",
        "",
        "## Review-queue ranking at fixed budgets",
        "",
        "If a reviewer reads only the top k of these 200 comments: expected precision, recall "
        "and lift (mean over fold seeds).",
        "",
        "| Model | " + " | ".join(f"P@{k} / R@{k} / Lift@{k}" for k in REVIEW_BUDGETS) + " |",
        "|---|" + "---|" * len(REVIEW_BUDGETS),
    ]
    for m in models:
        row = points.loc[m]
        cells = [
            f"{row[f'precision_at_{k}']:.2f} / {row[f'recall_at_{k}']:.2f} / {row[f'lift_at_{k}']:.2f}"
            for k in REVIEW_BUDGETS
        ]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Class-wise end-to-end results",
        "",
        "Mean over fold seeds. Support is the full corpus count. Two classes have no labels; "
        "classes under ~10 rows move by large steps with one prediction. **The `rules` column "
        "is inflated by the sampling design above** (for example, 17 of 18 "
        "`confusion_barrier` rows were sampled because the rule predicted that class). "
        "`majority` routes every row to Stage 2: its constant Stage 1 score, thresholded for "
        "F1 on validation, labels everything actionable.",
        "",
        "| Class | Support | " + " | ".join(f"{m} F1" for m in models) + " |",
        "|---|---|" + "---|" * len(models),
    ]
    means = per_class.groupby(["class", "model"])[["support", "f1"]].mean()
    for label in TAXONOMY:
        support = int(means.loc[(label, models[0]), "support"]) if (label, models[0]) in means.index else 0
        if support == 0:
            lines.append(f"| {label} | 0 | " + " | ".join("—" for _ in models) + " |")
            continue
        cells = [f"{means.loc[(label, m), 'f1']:.2f}" if (label, m) in means.index else "—" for m in models]
        lines.append(f"| {label} | {support} | " + " | ".join(cells) + " |")

    best = points["pr_auc"].idxmax()
    lines += [
        "",
        f"## Confusion matrix: `{best}` (reference seed {meta['seeds'][0]}, rows = true label)",
        "",
        *_md_table(confusion[best].reset_index().rename(columns={"index": "true \\ predicted"})),
        "",
        "## Error analysis (reference seed)",
        "",
        "Full error table: `reports/ml/model_comparison/errors.csv` (every Stage 1 error with "
        "text, group, rule label, probability and an empty `error_category` column for human "
        "review).",
        "",
        "| Model | False positives | False negatives |",
        "|---|---|---|",
    ]
    counts = errors.groupby(["model", "error_type"]).size().unstack(fill_value=0)
    for m in models:
        if m in counts.index:
            lines.append(
                f"| {m} | {counts.loc[m].get('false_positive', 0)} | {counts.loc[m].get('false_negative', 0)} |"
            )
    supervised = [m for m in models if m in SUPERVISED]
    hard = errors[
        (errors["model"] == best)
        & (errors["supervised_models_making_this_error"] == len(supervised))
    ]
    lines += [
        "",
        f"**Errors every supervised model makes** ({len(hard)} rows): these are where more "
        "labelled data or a label re-audit, not a different classifier, is needed.",
        "",
    ]
    for kind in ("false_positive", "false_negative"):
        subset = errors[(errors["model"] == best) & (errors["error_type"] == kind)]
        subset = subset.sort_values("p_actionable", ascending=(kind == "false_negative")).head(8)
        title = "Most confident false positives" if kind == "false_positive" else "Most confident false negatives"
        lines += [
            f"### {title} (`{best}`)",
            "",
            "| P(actionable) | True | Rule | All supervised wrong | Words | Text |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in subset.iterrows():
            lines.append(
                f"| {r['p_actionable']:.2f} | {r['corrected_label']} | {r['rule_label']} | "
                f"{'yes' if r['supervised_models_making_this_error'] == len(supervised) else 'no'} | "
                f"{r['words']} | {_excerpt(r['text'])} |"
            )
        lines.append("")
    lines += [
        "## What these results do and do not support",
        "",
        "- They describe ranking and classification of *public discussion* by behavioural "
        "intent. They are not purchase, sales or revenue predictions.",
        "- With 23 groups, a single well-represented video can move every metric; the "
        "bootstrap intervals above are the honest width of that uncertainty.",
        "- Labels come from one annotator (first-pass vs final κ 0.30); label noise bounds "
        "every number here. `docs/ANNOTATION_PROTOCOL.md` describes the independent "
        "double-annotation and adjudication needed before a final evaluation.",
        "- Per-group stability, calibration bins and slice-level error rates: "
        "`reports/ml/group_cv_report.md`.",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def run_comparison(
    models=CLASSICAL_MODELS,
    seeds=(42, 43, 44),
    n_folds: int = 5,
    n_boot: int = N_BOOTSTRAP,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = run_group_cv(
        models=models, seeds=seeds, n_folds=n_folds, n_boot=200,
        output_dir=output_dir / "group_cv", report_path=output_dir / "group_cv" / "report.md",
    )
    oof = pd.read_csv(output_dir / "group_cv" / "oof_predictions.csv")
    labelled = load_labelled_data()
    raw = pd.read_csv("reports/audits/intent_audit_corrected.csv")
    weights = design_weights(labelled, raw)
    oof["weight"] = oof["record_id"].astype(str).map(dict(zip(labelled["record_id"].astype(str), weights)))

    points = point_estimates(oof, models)
    draws = paired_bootstrap(oof, models, n_boot=n_boot)
    pairs = compare_pairs(draws, models)
    errors = error_table(oof, labelled, seeds[0], models)
    confusion = confusion_tables(oof, seeds[0], models)

    meta = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_status": EVALUATION_STATUS_DEVELOPMENT,
        "dataset_hash": dataset_hash(labelled),
        "rows": int(len(labelled)),
        "groups": int(labelled["parent_id"].nunique()),
        "positives": int(labelled["actionable"].astype(bool).sum()),
        "base_rate": float(labelled["actionable"].astype(bool).mean()),
        "models": list(models),
        "seeds": list(seeds),
        "n_folds": n_folds,
        "n_bootstrap": n_boot,
        "ece": {m: cv["metadata"]["summaries"][m]["ece_10_bins"] for m in models},
        "design_source": DESIGN_SOURCE,
        "design_shares": RULE_LABEL_POPULATION_SHARE,
        "weighted_base_rate": float(np.average(labelled["actionable"].astype(float), weights=weights)),
        "max_weight": float(weights.max()),
        "effective_sample_size": float(weights.sum() ** 2 / (weights**2).sum()),
    }
    if "embedding" in models:
        from src.ml.embeddings import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_REVISION

        meta["embedding_model"] = DEFAULT_EMBEDDING_MODEL
        meta["embedding_revision"] = DEFAULT_EMBEDDING_REVISION

    points.to_csv(output_dir / "point_estimates.csv")
    pairs.to_csv(output_dir / "paired_differences.csv", index=False)
    errors.to_csv(output_dir / "errors.csv", index=False)
    for m, table in confusion.items():
        table.to_csv(output_dir / f"confusion_{m}.csv")
    (output_dir / "summary.json").write_text(json.dumps(meta, indent=2, default=str))
    write_report(points, draws, pairs, cv["per_class"], errors, confusion, meta, report_path)
    return {"points": points, "pairs": pairs, "errors": errors, "meta": meta}


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe model comparison")
    parser.add_argument("--models", nargs="+", default=list(CLASSICAL_MODELS))
    parser.add_argument("--with-embedding", action="store_true",
                        help="Add the frozen MiniLM encoder baseline (needs .[embeddings])")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    models = list(args.models) + (["embedding"] if args.with_embedding else [])
    result = run_comparison(models=models, seeds=args.seeds, n_boot=args.bootstrap,
                            output_dir=args.output_dir, report_path=args.report)
    cols = ["pr_auc", "lift_at_20", "brier"]
    print(result["points"][cols].round(3).to_string())
    print(result["pairs"][["model", "versus", "metric", "verdict"]].to_string(index=False))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()

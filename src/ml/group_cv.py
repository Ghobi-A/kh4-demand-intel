"""Group-aware cross-validated evaluation with uncertainty and error slices.

The headline benchmark (``src.ml.benchmark``) scores each seed on one 15%
test split: about 29 rows from 4 discussion groups, often with zero or one
example of a class. This report complements it rather than replacing it:

* every labelled row receives an **out-of-fold** prediction from a model that
  never saw its discussion group (whole groups per fold), with calibration
  and the Stage 1 threshold fitted on an inner group-held-out validation split
  of the training folds only;
* metrics are pooled over all rows, so class support is the full corpus count;
* uncertainty comes from a **group bootstrap** (resampling discussion groups,
  the unit of dependence), and from repeating the fold assignment over seeds;
* results are broken down by class, by discussion group, by calibration bin
  and by error slice (source, sampling strategy, text length, rule agreement).

It is still development evaluation on the same 200 rows the models were
designed against. ``evaluation_status`` stays ``development_preliminary``.

    python -m src.ml.group_cv --models logistic char_word_svm --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.intent import classify_intent
from src.ml import EVALUATION_STATUS_DEVELOPMENT
from src.ml.benchmark import train_stage1
from src.ml.config import ExperimentConfig
from src.ml.data import ACTIONABLE_CLASSES, TAXONOMY, dataset_hash, load_labelled_data
from src.ml.hierarchy import HierarchicalClassifier, to_binary_labels
from src.ml.metrics import binary_metrics, expected_calibration_error, multiclass_metrics
from src.ml.models import build_model
from src.ml.probabilities import _inner_split, _yields_probabilities, group_folds
from src.ml.ranking import ranking_metrics

DEFAULT_OUTPUT_DIR = Path("reports/ml/group_cv")
DEFAULT_REPORT_PATH = Path("reports/ml/group_cv_report.md")
DEFAULT_MODELS = ("logistic", "char_word_svm")
DEFAULT_SEEDS = (42, 43, 44)
N_BOOTSTRAP = 1000


def oof_predictions(df: pd.DataFrame, model_name: str, seed: int, n_folds: int = 5) -> pd.DataFrame:
    """Out-of-fold Stage 1 probability and end-to-end label for every row."""
    usable = df.reset_index(drop=True)
    config = ExperimentConfig(seeds=[seed], models=[model_name])
    out = pd.DataFrame(
        {
            "record_id": usable["record_id"],
            "fold": -1,
            "p_actionable": np.nan,
            "threshold": np.nan,
            "pred_actionable": pd.NA,
            "pred_label": pd.NA,
        }
    )
    for fold, held_idx in enumerate(group_folds(usable, n_folds=n_folds, seed=seed)):
        if len(held_idx) == 0:
            continue
        held = usable.iloc[held_idx]
        rest = usable.drop(index=usable.index[held_idx]).reset_index(drop=True)
        try:
            inner = _inner_split(rest, seed)
        except ValueError:
            continue
        stage1, calibrator, threshold, _ = train_stage1(model_name, inner, config, seed)
        if not _yields_probabilities(stage1, calibrator):
            continue
        # Stage 2 learns from every actionable training-fold row.
        train_rows = pd.concat([inner.train, inner.validation], ignore_index=True)
        actionable = train_rows[to_binary_labels(train_rows["corrected_label"]) == 1]
        stage2 = build_model(model_name, seed=seed)
        stage2.fit(actionable["text"].tolist(), actionable["corrected_label"].tolist())
        hier = HierarchicalClassifier(
            stage1, stage2, threshold=threshold["selected_threshold"], calibrator=calibrator
        )
        texts = held["text"].astype(str).tolist()
        scores = hier.stage1_scores(texts)
        out.loc[held_idx, "fold"] = fold
        out.loc[held_idx, "p_actionable"] = np.asarray(scores, dtype=float)
        out.loc[held_idx, "threshold"] = threshold["selected_threshold"]
        out.loc[held_idx, "pred_actionable"] = hier.stage1_predict(texts)
        out.loc[held_idx, "pred_label"] = hier.predict(texts)
    out["model"] = model_name
    out["seed"] = seed
    return out


def _wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Equal-width reliability bins with counts and Wilson intervals."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & ((p < upper) if upper < 1 else (p <= upper))
        n = int(mask.sum())
        observed = float(y[mask].mean()) if n else float("nan")
        lo, hi = _wilson(float(y[mask].sum()), n)
        rows.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "n": n,
                "mean_predicted": float(p[mask].mean()) if n else float("nan"),
                "observed_rate": observed,
                "observed_95_low": lo,
                "observed_95_high": hi,
            }
        )
    return pd.DataFrame(rows)


def _stage1_scalar_metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict:
    metrics = binary_metrics(y, pred, y_prob=p)
    ranking = ranking_metrics(y, p, k_values=(10, 25, 50), percentages=(0.10, 0.20))
    return {
        "pr_auc": metrics.get("pr_auc"),
        "roc_auc": metrics.get("roc_auc"),
        "brier": metrics.get("brier_score"),
        "ece": metrics.get("ece"),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "base_rate": float(y.mean()),
        "precision_at_10pct": ranking.get("precision_at_10pct"),
        "lift_at_10pct": ranking.get("lift_at_10pct"),
        "lift_at_20pct": ranking.get("lift_at_20pct"),
    }


def group_bootstrap(
    frame: pd.DataFrame, n_boot: int = N_BOOTSTRAP, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """95% intervals for Stage 1 metrics, resampling discussion groups."""
    rng = np.random.default_rng(seed)
    groups = frame["parent_id"].astype(str).unique()
    by_group = {g: frame[frame["parent_id"].astype(str) == g] for g in groups}
    draws: dict[str, list[float]] = {"pr_auc": [], "roc_auc": [], "brier": [], "lift_at_10pct": []}
    for _ in range(n_boot):
        sample = pd.concat([by_group[g] for g in rng.choice(groups, size=len(groups))])
        y = sample["true_actionable"].to_numpy(dtype=int)
        p = sample["p_actionable"].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            continue
        draws["pr_auc"].append(average_precision_score(y, p))
        draws["roc_auc"].append(roc_auc_score(y, p))
        draws["brier"].append(brier_score_loss(y, p))
        draws["lift_at_10pct"].append(
            ranking_metrics(y, p, k_values=(), percentages=(0.10,))["lift_at_10pct"]
        )
    return {
        key: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for key, values in draws.items()
        if values
    }


def _length_bucket(text: str) -> str:
    words = len(str(text).split())
    if words <= 8:
        return "short (<=8 words)"
    if words <= 25:
        return "medium (9-25 words)"
    return "long (>25 words)"


def error_slices(frame: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 error rates by source, sampling strategy, length and rule agreement."""
    data = frame.copy()
    data["length_bucket"] = data["text"].map(_length_bucket)
    data["rule_agrees_with_label"] = np.where(
        data["rule_label"] == data["corrected_label"], "rule correct", "rule wrong"
    )
    rows = []
    for dimension in ["source", "sample_strategy", "length_bucket", "rule_agrees_with_label"]:
        for value, group in data.groupby(dimension):
            y = group["true_actionable"].to_numpy(dtype=int)
            pred = group["pred_actionable"].to_numpy(dtype=int)
            p = group["p_actionable"].to_numpy(dtype=float)
            positives = int(y.sum())
            negatives = int(len(y) - positives)
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "n": int(len(y)),
                    "actionable": positives,
                    "recall": float((pred[y == 1] == 1).mean()) if positives else np.nan,
                    "false_positive_rate": float((pred[y == 0] == 1).mean()) if negatives else np.nan,
                    "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
                    "e2e_accuracy": float((group["pred_label"] == group["corrected_label"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def per_group_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, group in frame.groupby("parent_id"):
        y = group["true_actionable"].to_numpy(dtype=int)
        rows.append(
            {
                "parent_id": group_id,
                "n": int(len(group)),
                "actionable": int(y.sum()),
                "observed_rate": float(y.mean()),
                "mean_p_actionable": float(group["p_actionable"].mean()),
                "stage1_accuracy": float((group["pred_actionable"].astype(int) == y).mean()),
                "e2e_accuracy": float((group["pred_label"] == group["corrected_label"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def run_group_cv(
    models=DEFAULT_MODELS,
    seeds=DEFAULT_SEEDS,
    n_folds: int = 5,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    n_boot: int = N_BOOTSTRAP,
    link_near_duplicates: bool = False,
) -> dict:
    df = load_labelled_data()
    duplicate_audit = None
    if link_near_duplicates:
        from src.ml.leakage import link_near_duplicate_groups

        df, duplicate_audit = link_near_duplicate_groups(df)
    df = df[df["corrected_label"].notna()].reset_index(drop=True)
    df["true_actionable"] = to_binary_labels(df["corrected_label"])
    df["rule_label"] = [classify_intent(str(t))[0] for t in df["text"]]

    all_oof, seed_rows, per_class_rows = [], [], []
    summaries: dict[str, dict] = {}
    for model_name in models:
        reference = None
        for seed in seeds:
            oof = oof_predictions(df, model_name, seed, n_folds)
            merged = df.merge(oof, on="record_id")
            scored = merged[merged["p_actionable"].notna()].copy()
            scored["pred_actionable"] = scored["pred_actionable"].astype(int)
            y = scored["true_actionable"].to_numpy(dtype=int)
            p = scored["p_actionable"].to_numpy(dtype=float)
            row = {"model": model_name, "seed": seed, "rows_scored": int(len(scored))}
            row.update(_stage1_scalar_metrics(y, p, scored["pred_actionable"].to_numpy()))
            e2e = multiclass_metrics(
                scored["corrected_label"], scored["pred_label"].astype(str), labels=list(TAXONOMY)
            )
            observed = [c for c in TAXONOMY if (scored["corrected_label"] == c).any()]
            row["e2e_macro_f1_observed_classes"] = float(
                np.mean([e2e["per_class"][c]["f1"] for c in observed])
            )
            row["e2e_accuracy"] = e2e["accuracy"]
            seed_rows.append(row)
            for label, values in e2e["per_class"].items():
                per_class_rows.append({"model": model_name, "seed": seed, "class": label, **values})
            all_oof.append(scored.assign(model=model_name, seed=seed))
            if reference is None:
                reference = scored

        intervals = group_bootstrap(reference, n_boot=n_boot) if reference is not None else {}
        summaries[model_name] = {
            "reference_seed": seeds[0],
            "group_bootstrap_95": intervals,
            "calibration": calibration_table(
                reference["true_actionable"].to_numpy(dtype=int),
                reference["p_actionable"].to_numpy(dtype=float),
            ).to_dict(orient="records"),
            "ece_10_bins": expected_calibration_error(
                reference["true_actionable"].to_numpy(dtype=int),
                reference["p_actionable"].to_numpy(dtype=float),
            ),
        }

    seed_table = pd.DataFrame(seed_rows)
    per_class = pd.DataFrame(per_class_rows)
    oof_all = pd.concat(all_oof, ignore_index=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_table.to_csv(output_dir / "per_seed.csv", index=False)
    per_class.to_csv(output_dir / "per_class.csv", index=False)
    keep = ["record_id", "parent_id", "source", "sample_strategy", "corrected_label",
            "true_actionable", "rule_label", "model", "seed", "fold", "threshold",
            "p_actionable", "pred_actionable", "pred_label"]
    oof_all[keep].to_csv(output_dir / "oof_predictions.csv", index=False)

    slices, groups = {}, {}
    for model_name in models:
        reference = oof_all[(oof_all["model"] == model_name) & (oof_all["seed"] == seeds[0])]
        slices[model_name] = error_slices(reference)
        groups[model_name] = per_group_table(reference)
        slices[model_name].assign(model=model_name).to_csv(
            output_dir / f"error_slices_{model_name}.csv", index=False
        )
        groups[model_name].assign(model=model_name).to_csv(
            output_dir / f"per_group_{model_name}.csv", index=False
        )

    metadata = {
        "evaluation_status": EVALUATION_STATUS_DEVELOPMENT,
        "dataset_hash": dataset_hash(df),
        "rows": int(len(df)),
        "groups": int(df["parent_id"].nunique()),
        "models": list(models),
        "seeds": list(seeds),
        "n_folds": n_folds,
        "n_bootstrap": n_boot,
        "near_duplicate_linking": duplicate_audit,
        "design": (
            "Whole discussion groups per outer fold; calibration and Stage 1 threshold "
            "fitted on an inner group-held-out validation split of the training folds; "
            "group bootstrap over the reference seed's pooled OOF predictions."
        ),
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, default=str))
    write_group_cv_report(seed_table, per_class, slices, groups, metadata, report_path)
    return {"per_seed": seed_table, "per_class": per_class, "metadata": metadata}


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def write_group_cv_report(seed_table, per_class, slices, groups, metadata, path: Path) -> None:
    lines = [
        "# Group-aware cross-validated evaluation",
        "",
        "> **Preliminary development evaluation.** Pooled out-of-fold predictions over "
        f"the {metadata['rows']}-row audit corpus ({metadata['groups']} discussion groups). "
        "Not final held-out performance: the models, taxonomy and thresholds were "
        "developed against these same rows. Generated by `python -m src.ml.group_cv`.",
        "",
        f"- Dataset hash `{metadata['dataset_hash']}`; {metadata['n_folds']} group folds; "
        f"seeds {metadata['seeds']}; {metadata['n_bootstrap']} group-bootstrap resamples",
        f"- Design: {metadata['design']}",
        "",
        "## Stage 1 (actionable vs general), mean over seeds",
        "",
        "| Model | PR-AUC | ROC-AUC | Brier | ECE | F1 | Lift@10% | e2e macro-F1 (observed classes) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, group in seed_table.groupby("model"):
        def m(col):
            return f"{group[col].mean():.3f} ± {group[col].std(ddof=1) if len(group) > 1 else 0:.3f}"
        lines.append(
            f"| {model} | {m('pr_auc')} | {m('roc_auc')} | {m('brier')} | {m('ece')} | "
            f"{m('f1')} | {m('lift_at_10pct')} | {m('e2e_macro_f1_observed_classes')} |"
        )
    base_rate = seed_table["base_rate"].iloc[0]
    lines += [
        "",
        f"± is the spread over fold assignments (seeds). Base rate of actionable rows: "
        f"{base_rate:.2f}, which is the PR-AUC of a random ranking.",
        "",
        "## Uncertainty: group bootstrap (reference seed)",
        "",
        "Resampling discussion groups, the unit of dependence. With 23 groups, these "
        "intervals are wide; that width is the honest finding.",
        "",
        "| Model | PR-AUC 95% | ROC-AUC 95% | Brier 95% | Lift@10% 95% |",
        "|---|---|---|---|---|",
    ]
    for model, summary in metadata["summaries"].items():
        ci = summary["group_bootstrap_95"]

        def c(key):
            low, high = ci.get(key, (float("nan"), float("nan")))
            return f"{_fmt(low)}–{_fmt(high)}"
        lines.append(f"| {model} | {c('pr_auc')} | {c('roc_auc')} | {c('brier')} | {c('lift_at_10pct')} |")

    lines += [
        "",
        "## Calibration (reference seed)",
        "",
        "Observed actionable rate per predicted-probability bin, with Wilson 95% "
        "intervals. Sparse bins are reported, not hidden.",
        "",
    ]
    for model, summary in metadata["summaries"].items():
        lines += [
            f"**{model}** (10-bin ECE {summary['ece_10_bins']:.3f})",
            "",
            "| Bin | n | Mean predicted | Observed | 95% interval |",
            "|---|---|---|---|---|",
        ]
        for row in summary["calibration"]:
            lines.append(
                f"| {row['bin']} | {row['n']} | {_fmt(row['mean_predicted'], 2)} | "
                f"{_fmt(row['observed_rate'], 2)} | {_fmt(row['observed_95_low'], 2)}–"
                f"{_fmt(row['observed_95_high'], 2)} |"
            )
        lines.append("")

    lines += [
        "## Per-class end-to-end results (mean over seeds)",
        "",
        "Support is the full corpus count, because every row is scored out-of-fold. "
        "Classes with fewer than ~10 rows have F1 values that move by large steps "
        "with a single prediction.",
        "",
        "| Model | Class | Support | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    means = per_class.groupby(["model", "class"])[["support", "precision", "recall", "f1"]].mean()
    for (model, label), row in means.iterrows():
        support = int(row["support"])
        if support == 0:
            lines.append(f"| {model} | {label} | 0 | — | — | — (no labelled rows) |")
            continue
        lines.append(
            f"| {model} | {label} | {support} | {row['precision']:.2f} | "
            f"{row['recall']:.2f} | {row['f1']:.2f} |"
        )

    lines += ["", "## Error slices (reference seed)", ""]
    for model, table in slices.items():
        lines += [
            f"**{model}**",
            "",
            "| Dimension | Value | n | Actionable | Recall | FPR | Brier | e2e accuracy |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, row in table.iterrows():
            lines.append(
                f"| {row['dimension']} | {row['value']} | {row['n']} | {row['actionable']} | "
                f"{_fmt(row['recall'], 2)} | {_fmt(row['false_positive_rate'], 2)} | "
                f"{_fmt(row['brier'], 3)} | {_fmt(row['e2e_accuracy'], 2)} |"
            )
        lines.append("")

    lines += [
        "## Per-group stability (reference seed)",
        "",
        "Each group is scored by a model that never saw it. Large gaps between "
        "observed rate and mean predicted probability show group-level shift.",
        "",
    ]
    for model, table in groups.items():
        lines += [
            f"**{model}**",
            "",
            "| Group | n | Actionable | Observed rate | Mean P(actionable) | Stage 1 acc | e2e acc |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, row in table.iterrows():
            lines.append(
                f"| {row['parent_id']} | {row['n']} | {row['actionable']} | "
                f"{row['observed_rate']:.2f} | {row['mean_p_actionable']:.2f} | "
                f"{row['stage1_accuracy']:.2f} | {row['e2e_accuracy']:.2f} |"
            )
        lines.append("")

    lines += [
        "## Reading notes",
        "",
        "- Two taxonomy classes have no labelled rows; their metrics are undefined, "
        "and the e2e macro-F1 here averages only classes present in the corpus.",
        f"- {len(ACTIONABLE_CLASSES)} actionable classes share {int(base_rate * metadata['rows'])} "
        "actionable rows, so Stage 2 is learned from very few examples per class.",
        "- 197/200 rows are YouTube; the Reddit slice is three rows and says nothing "
        "about Reddit performance.",
        "- Labels come from one annotator; a first-pass/final kappa of about 0.30 "
        "(`reports/data_quality/dataset_audit.md`) means label noise is likely a "
        "ceiling on every metric here.",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-aware cross-validated evaluation")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    result = run_group_cv(
        models=args.models, seeds=args.seeds, n_folds=args.folds,
        output_dir=args.output_dir, report_path=args.report, n_boot=args.bootstrap,
    )
    cols = ["model", "seed", "rows_scored", "pr_auc", "brier", "ece", "f1", "lift_at_10pct",
            "e2e_macro_f1_observed_classes"]
    print(result["per_seed"][cols].round(3).to_string(index=False))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()

"""Markdown report generation from benchmark result files.

Metrics are always rendered from result CSVs, never hand-written.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PRELIMINARY_BANNER = (
    "> **Preliminary development benchmark.** These numbers come from the "
    "current 200-row audit corpus, which is still being re-audited and "
    "expanded. They are not final held-out performance and must not be "
    "quoted as production or CV headline metrics.\n"
)


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.3f}"


def write_model_comparison_report(summary: pd.DataFrame, metadata: dict, path: Path) -> None:
    """Render the model-comparison Markdown report from the summary table."""
    lines = [
        "# Model comparison",
        "",
        PRELIMINARY_BANNER,
        f"- Run: `{metadata.get('run_id')}` at {metadata.get('timestamp')}",
        f"- Git commit: `{metadata.get('git_commit')}`",
        f"- Dataset hash: `{metadata.get('dataset_hash')}` "
        f"({metadata.get('row_count')} rows, annotation `{metadata.get('annotation_version')}`)",
        f"- Seeds: {metadata.get('seeds')} (seed scope: {metadata.get('seed_scope')})",
        f"- Evaluation mode: `{metadata.get('evaluation_mode')}`",
        f"- Evaluation status: `{metadata.get('evaluation_status')}`",
        f"- Split rows (seed {metadata.get('seeds', ['?'])[0]}): "
        f"train {metadata.get('train_rows')} / validation {metadata.get('validation_rows')} / "
        f"test {metadata.get('test_rows')}",
        f"- Split groups: train {metadata.get('train_groups')} / "
        f"validation {metadata.get('validation_groups')} / test {metadata.get('test_groups')}",
        "",
        "Values are mean ± standard deviation over seeds (each seed regenerates "
        "the group split and retrains the model).",
        "",
        "| Model | Stage 1 F1 | Stage 1 PR-AUC | Stage 2 macro-F1 | Flat macro-F1 | End-to-end macro-F1 | Lift@10% |",
        "|---|---|---|---|---|---|---|",
    ]

    def cell(row, metric):
        mean = row.get(f"{metric}_mean")
        std = row.get(f"{metric}_std")
        if mean is None or pd.isna(mean):
            return "—"
        if std is None or pd.isna(std):
            return _fmt(mean)
        return f"{mean:.3f} ± {std:.3f}"

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {cell(row, 'stage1_f1')} | {cell(row, 'stage1_pr_auc')} | "
            f"{cell(row, 'stage2_macro_f1')} | {cell(row, 'flat_macro_f1')} | "
            f"{cell(row, 'e2e_macro_f1')} | {cell(row, 'lift_at_10pct')} |"
        )

    lines += [
        "",
        "## Reading notes",
        "",
        "- Stage 1 = actionable vs general_discussion; PR-AUC and F1 are "
        "prioritised over accuracy because actionable signals are the minority.",
        "- Stage 2 macro-F1 is measured on true-actionable test rows only.",
        "- End-to-end macro-F1 routes every test row through Stage 1 then Stage 2.",
        "- The rule baseline emits a fixed confidence indicator, not a "
        "probability, so probability-dependent metrics are blank for it.",
        "- Lift@10% = how many times more actionable signals a reviewer finds in "
        "the top 10% model-ranked comments than under random review.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def load_summary(output_dir: Path) -> pd.DataFrame | None:
    path = Path(output_dir) / "results_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

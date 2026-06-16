"""Build portfolio-ready dashboard datasets from locally scored signals.

This script intentionally depends on the local processed dataset and does not
fabricate replacement rows when that dataset is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.score import aggregate_video_scores, score_dataframe

INPUT_PATH = Path("data/processed/signals_scored.csv")
AUDIT_PATH = Path("reports/audits/intent_audit_corrected.csv")
SIGNALS_OUTPUT_PATH = Path("data/demo/signals_scored_portfolio.csv")
VIDEO_OUTPUT_PATH = Path("data/demo/video_scores_portfolio.csv")

IDENTIFYING_COLUMNS = {
    "author",
    "id",
    "permalink",
    "scraped_at",
    "username",
    "user",
    "channel_id",
}
INTENT_CLASSES = [
    "high_intent",
    "nostalgia_reactivation",
    "new_customer_interest",
    "frustrated_demand",
    "content_drought_fatigue",
    "confusion_barrier",
    "expectation_decay",
    "general_discussion",
]
SAMPLE_PER_CLASS = 60
RANDOM_STATE = 42


def _fail_if_missing_input() -> None:
    if INPUT_PATH.exists():
        return

    raise FileNotFoundError(
        f"Required local input not found: {INPUT_PATH}.\n"
        "Build the processed scored dataset locally before creating the "
        "portfolio extract. This script will not fabricate portfolio data when "
        "the processed file is missing."
    )


def _strip_identifying_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = sorted(IDENTIFYING_COLUMNS.intersection(df.columns))
    return df.drop(columns=columns_to_drop)


def _sample_by_intent(df: pd.DataFrame) -> pd.DataFrame:
    if "intent_label" not in df.columns:
        raise ValueError("Expected column 'intent_label' in scored signals.")

    sampled = (
        df.groupby("intent_label", group_keys=False, dropna=False)
        .apply(lambda group: group.sample(n=min(len(group), SAMPLE_PER_CLASS), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    return sampled


def _audit_rows_for_missing_classes(missing_classes: set[str]) -> pd.DataFrame:
    if not missing_classes:
        return pd.DataFrame()
    if not AUDIT_PATH.exists():
        print(
            "Audit patch skipped because "
            f"{AUDIT_PATH} does not exist. Missing classes remain: {sorted(missing_classes)}"
        )
        return pd.DataFrame()

    audit_df = pd.read_csv(AUDIT_PATH)
    required_columns = {"corrected_label", "sentiment_label", "engagement"}
    missing_columns = required_columns - set(audit_df.columns)
    if missing_columns:
        raise ValueError(
            f"Expected columns {sorted(required_columns)} in {AUDIT_PATH}; "
            f"missing {sorted(missing_columns)}"
        )

    patched = audit_df[audit_df["corrected_label"].isin(missing_classes)].copy()
    if patched.empty:
        return patched

    patched["intent_label"] = patched["corrected_label"]
    if "parent_id" not in patched.columns:
        patched["parent_id"] = "intent_audit_" + patched["intent_label"].astype(str)
    if "source" not in patched.columns:
        patched["source"] = "intent_audit"

    patched = _strip_identifying_columns(patched)
    patched = score_dataframe(patched)
    return patched


def build_portfolio_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build row-level and parent-level portfolio CSVs."""
    _fail_if_missing_input()

    scored_df = pd.read_csv(INPUT_PATH)
    sampled_df = _sample_by_intent(scored_df)

    represented_classes = set(sampled_df["intent_label"].dropna().astype(str))
    missing_classes = set(INTENT_CLASSES) - represented_classes
    patched_df = _audit_rows_for_missing_classes(missing_classes)

    if not patched_df.empty:
        sampled_df = pd.concat([sampled_df, patched_df], ignore_index=True, sort=False)
        sampled_df = _sample_by_intent(sampled_df)

    portfolio_df = _strip_identifying_columns(sampled_df)
    portfolio_df = score_dataframe(portfolio_df)
    video_scores_df = aggregate_video_scores(portfolio_df)

    SIGNALS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VIDEO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    portfolio_df.to_csv(SIGNALS_OUTPUT_PATH, index=False)
    video_scores_df.to_csv(VIDEO_OUTPUT_PATH, index=False)

    intent_classes = sorted(portfolio_df["intent_label"].dropna().astype(str).unique().tolist())
    print(f"Portfolio signal rows: {len(portfolio_df):,}")
    print(f"Intent classes: {intent_classes}")
    print(f"Unique parent_id count: {portfolio_df['parent_id'].nunique(dropna=False):,}")
    print(f"Signals output: {SIGNALS_OUTPUT_PATH}")
    print(f"Video scores output: {VIDEO_OUTPUT_PATH}")

    return portfolio_df, video_scores_df


def main() -> None:
    build_portfolio_dataset()


if __name__ == "__main__":
    main()

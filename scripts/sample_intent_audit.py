"""Sample stratified rows for manual intent audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/processed/signals_intent.csv")
DEFAULT_OUTPUT = Path("reports/audits/intent_audit_sample.csv")
OUTPUT_COLUMNS = [
    "id",
    "source",
    "record_type",
    "text",
    "sentiment_label",
    "intent_label",
    "engagement",
    "permalink",
    "sample_type",
    "manual_label",
    "notes",
]


def _sample_group(
    frame: pd.DataFrame,
    n: int,
    seed: int,
    label: str,
    sample_type: str,
) -> pd.DataFrame:
    available = len(frame)
    if available < n:
        print(
            f"Warning: requested {n} rows for {sample_type} label '{label}' "
            f"but only found {available}; using all available rows."
        )
        sampled = frame.copy()
    else:
        sampled = frame.sample(n=n, random_state=seed)

    sampled = sampled.copy()
    sampled["sample_type"] = sample_type
    return sampled


def build_audit_sample(
    input_path: Path,
    output_path: Path,
    recall_n: int,
    precision_n: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    recall_df = _sample_group(
        frame=df[df["intent_label"] == "general_discussion"],
        n=recall_n,
        seed=seed,
        label="general_discussion",
        sample_type="recall",
    )

    precision_parts: list[pd.DataFrame] = []
    other_labels = sorted(label for label in df["intent_label"].dropna().unique() if label != "general_discussion")
    for label in other_labels:
        precision_parts.append(
            _sample_group(
                frame=df[df["intent_label"] == label],
                n=precision_n,
                seed=seed,
                label=label,
                sample_type="precision",
            )
        )

    sampled_df = pd.concat([recall_df, *precision_parts], ignore_index=True)

    sampled_df["manual_label"] = ""
    sampled_df["notes"] = ""

    missing_columns = [col for col in OUTPUT_COLUMNS if col not in sampled_df.columns]
    for column in missing_columns:
        sampled_df[column] = ""

    sampled_df = sampled_df[OUTPUT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False)
    return sampled_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create intent audit sample CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recall-n", type=int, default=100)
    parser.add_argument("--precision-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_audit_sample(
        input_path=args.input,
        output_path=args.output,
        recall_n=args.recall_n,
        precision_n=args.precision_n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

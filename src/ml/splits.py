"""Group-aware train/validation/test splitting with leakage checks.

Random row-level splitting leaks discussion context: comments from the
same YouTube video or Reddit thread share vocabulary, topic, and author
overlap. Where a real discussion-group ID exists (derived from platform
permalinks), whole groups are assigned to exactly one split.

Rows without a recoverable group ID are given per-row fallback groups and
the resulting split is marked ``development_only`` — final evaluation
requires recovered group metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SPLIT_DIR = Path("reports/splits")

EVALUATION_MODE_GROUP = "group_holdout"
EVALUATION_MODE_TEMPORAL = "temporal_holdout"
EVALUATION_MODE_DEV_ONLY = "development_only"


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    seed: int
    evaluation_mode: str

    def frames(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def _effective_groups(df: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Return group keys per row and whether any fallback groups were needed."""
    groups = df["parent_id"].astype("string")
    missing = groups.isna()
    if missing.any():
        fallback = "row:" + df.loc[missing, "record_id"].astype(str)
        groups = groups.copy()
        groups[missing] = fallback
    return groups, bool(missing.any())


def group_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SplitResult:
    """Split rows into train/validation/test with group isolation.

    Groups are shuffled deterministically by seed and assigned greedily to
    the split furthest below its target row count, so no ``parent_id``
    appears in more than one split.
    """
    if abs(train_frac + validation_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    groups, used_fallback = _effective_groups(df)
    unique_groups = sorted(groups.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    group_sizes = groups.value_counts()
    total = len(df)
    targets = {
        "train": train_frac * total,
        "validation": validation_frac * total,
        "test": test_frac * total,
    }
    assigned: dict[str, str] = {}
    filled = {"train": 0, "validation": 0, "test": 0}

    for group in unique_groups:
        deficits = {
            name: (filled[name] / targets[name]) if targets[name] else 1.0
            for name in targets
        }
        chosen = min(deficits, key=lambda name: (deficits[name], name))
        assigned[group] = chosen
        filled[chosen] += int(group_sizes[group])

    membership = groups.map(assigned)
    result = SplitResult(
        train=df[membership == "train"].reset_index(drop=True),
        validation=df[membership == "validation"].reset_index(drop=True),
        test=df[membership == "test"].reset_index(drop=True),
        seed=seed,
        evaluation_mode=EVALUATION_MODE_DEV_ONLY if used_fallback else EVALUATION_MODE_GROUP,
    )
    for name, frame in result.frames().items():
        if frame.empty:
            raise ValueError(f"Split '{name}' is empty; dataset too small for these fractions")
    return result


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
) -> SplitResult:
    """Train on earlier data, test on later data, by ``created_at``.

    Secondary to group holdout; requires timestamps on every row.
    """
    if "created_at" not in df.columns or df["created_at"].isna().any():
        raise ValueError(
            "Temporal split requires a complete created_at column; "
            "current audit data lacks timestamps"
        )
    ordered = df.sort_values("created_at").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + validation_frac))
    return SplitResult(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        validation=ordered.iloc[train_end:val_end].reset_index(drop=True),
        test=ordered.iloc[val_end:].reset_index(drop=True),
        seed=0,
        evaluation_mode=EVALUATION_MODE_TEMPORAL,
    )


def check_leakage(split: SplitResult) -> list[str]:
    """Return leakage problems: shared groups or duplicate texts across splits."""
    problems: list[str] = []
    frames = split.frames()
    names = list(frames)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            groups_a = set(frames[first]["parent_id"].dropna())
            groups_b = set(frames[second]["parent_id"].dropna())
            shared = groups_a & groups_b
            if shared:
                problems.append(
                    f"Groups shared between {first} and {second}: {sorted(shared)[:5]}"
                )
            texts_a = set(frames[first]["text"].astype(str).str.strip().str.casefold())
            texts_b = set(frames[second]["text"].astype(str).str.strip().str.casefold())
            overlap = texts_a & texts_b
            if overlap:
                problems.append(
                    f"{len(overlap)} duplicate texts shared between {first} and {second}"
                )
    return problems


def save_split_manifests(split: SplitResult, output_dir: Path = DEFAULT_SPLIT_DIR) -> None:
    """Save one manifest CSV per split plus a summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in split.frames().items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(frame),
                "groups": frame["parent_id"].dropna().nunique(),
                "seed": split.seed,
                "evaluation_mode": split.evaluation_mode,
            }
            for name, frame in split.frames().items()
        ]
    )
    summary.to_csv(output_dir / "split_summary.csv", index=False)

"""Inter-annotator agreement utilities.

Ready for use once a second annotator reviews the corpus. No synthetic
second annotations are generated anywhere in this project.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def raw_agreement(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Fraction of items on which two annotators agree."""
    a, b = pd.Series(list(labels_a)), pd.Series(list(labels_b))
    if len(a) != len(b):
        raise ValueError("Annotator label lists must be the same length")
    if len(a) == 0:
        raise ValueError("Cannot compute agreement on zero items")
    return float((a.to_numpy() == b.to_numpy()).mean())


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Cohen's kappa for two annotators."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Annotator label lists must be the same length")
    return float(cohen_kappa_score(list(labels_a), list(labels_b)))


def disagreement_table(labels_a: Sequence[str], labels_b: Sequence[str]) -> pd.DataFrame:
    """Class-level cross-tabulation of annotator disagreements."""
    a, b = pd.Series(list(labels_a), name="annotator_a"), pd.Series(list(labels_b), name="annotator_b")
    table = pd.crosstab(a, b)
    return table

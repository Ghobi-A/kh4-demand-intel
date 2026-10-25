"""Link groups connected by near-duplicate text before splitting, without labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def link_near_duplicate_groups(frame: pd.DataFrame, threshold: float = 0.80) -> tuple[pd.DataFrame, dict]:
    """Use connected components; threshold is a declared sensitivity assumption."""
    if not 0 < threshold <= 1:
        raise ValueError("Similarity threshold must be in (0, 1]")
    if frame["parent_id"].isna().any():
        raise ValueError("Near-duplicate group linking requires complete discussion groups")
    data = frame.copy()
    groups = data["parent_id"].astype(str).to_numpy()
    parent = {g: g for g in groups}

    def root(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(
        data["text"].fillna("").astype(str).str.casefold()
    )
    pairs = []
    for start in range(0, len(data), 256):
        similarity = cosine_similarity(vectors[start:start + 256], vectors)
        for local, j in zip(*np.where(similarity >= threshold)):
            i = start + local
            if j <= i or groups[i] == groups[j]:
                continue
            a, b = sorted([root(groups[i]), root(groups[j])])
            parent[b] = a
            pairs.append({"record_a": str(data.iloc[i]["record_id"]),
                          "record_b": str(data.iloc[j]["record_id"]),
                          "similarity": float(similarity[local, j])})
    data["parent_id"] = [root(g) for g in groups]
    return data, {"threshold": threshold, "original_groups": len(set(groups)),
                  "effective_groups": int(data["parent_id"].nunique()), "cross_group_pairs": pairs}

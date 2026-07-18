"""Ranking metrics for actionable-signal prioritisation.

Answers the review-efficiency question: if a human reviews only the top-K
comments ranked by model score, how many actionable signals do they find
compared with random review? Only measured lift is reported — no business
impact is inferred.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _ranked_labels(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return np.asarray(y_true, dtype=int)[order]


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    ranked = _ranked_labels(y_true, scores)
    k = min(k, len(ranked))
    if k == 0:
        return 0.0
    return float(ranked[:k].mean())


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    ranked = _ranked_labels(y_true, scores)
    total_positive = int(ranked.sum())
    if total_positive == 0:
        return 0.0
    k = min(k, len(ranked))
    return float(ranked[:k].sum() / total_positive)


def lift_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision@K divided by the base rate (random-review precision)."""
    base_rate = float(np.asarray(y_true, dtype=int).mean())
    if base_rate == 0:
        return 0.0
    return precision_at_k(y_true, scores, k) / base_rate


def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_values: tuple[int, ...] = (10, 25, 50, 100),
    percentages: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict:
    """Precision@K / Recall@K / Lift@K at absolute and percentage cut-offs."""
    n = len(y_true)
    metrics: dict[str, float] = {}
    for k in k_values:
        if k > n:
            continue
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
        metrics[f"recall_at_{k}"] = recall_at_k(y_true, scores, k)
        metrics[f"lift_at_{k}"] = lift_at_k(y_true, scores, k)
    for pct in percentages:
        k = max(1, int(round(n * pct)))
        tag = f"{int(pct * 100)}pct"
        metrics[f"precision_at_{tag}"] = precision_at_k(y_true, scores, k)
        metrics[f"recall_at_{tag}"] = recall_at_k(y_true, scores, k)
        metrics[f"lift_at_{tag}"] = lift_at_k(y_true, scores, k)
    return metrics


def gains_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative gains: fraction reviewed vs fraction of positives found."""
    ranked = _ranked_labels(y_true, scores)
    total = max(int(ranked.sum()), 1)
    fractions = np.arange(1, len(ranked) + 1) / len(ranked)
    gains = np.cumsum(ranked) / total
    return fractions, gains


def save_ranking_figures(
    y_true: np.ndarray, scores: np.ndarray, output_dir: Path
) -> list[Path]:
    """Save cumulative-gains, lift, and precision/recall@K curves as PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    fractions, gains = gains_curve(y_true, scores)
    base_rate = max(float(np.asarray(y_true, dtype=int).mean()), 1e-9)
    ranked = _ranked_labels(y_true, scores)
    cum_precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    cum_recall = gains

    curves = [
        (
            "cumulative_gains.png",
            "Cumulative gains",
            "Fraction of comments reviewed",
            "Fraction of actionable signals found",
            [(fractions, gains, "model"), (fractions, fractions, "random")],
        ),
        (
            "lift_curve.png",
            "Lift curve",
            "Fraction of comments reviewed",
            "Lift vs random review",
            [(fractions, cum_precision / base_rate, "model"), (fractions, np.ones_like(fractions), "random")],
        ),
        (
            "precision_at_k.png",
            "Precision@K",
            "K (top-ranked comments reviewed)",
            "Precision",
            [(np.arange(1, len(ranked) + 1), cum_precision, "model")],
        ),
        (
            "recall_at_k.png",
            "Recall@K",
            "K (top-ranked comments reviewed)",
            "Recall",
            [(np.arange(1, len(ranked) + 1), cum_recall, "model")],
        ),
    ]
    for filename, title, xlabel, ylabel, series in curves:
        fig, ax = plt.subplots(figsize=(6, 4))
        for x, y, label in series:
            ax.plot(x, y, label=label)
        ax.set_title(f"{title} (development preliminary)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written

"""How much a matched-origin model ranking depends on individual origins.

The Bayesian model is backtested on only a few recent origins, so the
like-for-like comparison against the baselines rests on a handful of
forecasts. A ranking decided by one lucky or unlucky origin is not evidence
that one model is better. This module makes that visible:

* per-origin WAPE for every model;
* leave-one-origin-out WAPE and the winner each time;
* how much of the gap between the leader and the reference (Bayesian) model,
  or the runner-up when the reference leads, comes from the single most
  influential origin.

It never changes which model is selected. It only qualifies how strongly the
selection can be read.

    python -m src.forecasting.sensitivity --report-dir reports/demo_forecasting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Below this many forecasts a ranking is reported as fragile whatever it shows.
MIN_FORECASTS_FOR_RANKING = 12


def _wape(group: pd.DataFrame) -> float:
    denominator = float(np.abs(group["actual"]).sum())
    if denominator <= 0:
        return float("nan")
    return float(np.abs(group["actual"] - group["forecast"]).sum() / denominator)


def matched_subset(backtest: pd.DataFrame, reference_model: str = "bayesian") -> pd.DataFrame:
    """Rows on the origins where ``reference_model`` was scored, every model."""
    origins = set(backtest.loc[backtest["model"] == reference_model, "origin"])
    if not origins:
        return backtest.iloc[0:0]
    subset = backtest[backtest["origin"].isin(origins)]
    # Only models scored on every one of those origins are comparable.
    counts = subset.groupby("model")["origin"].nunique()
    keep = set(counts[counts == len(origins)].index)
    return subset[subset["model"].isin(keep)].reset_index(drop=True)


def per_origin_wape(subset: pd.DataFrame) -> pd.DataFrame:
    """WAPE and absolute error for every (origin, model) pair."""
    rows = []
    for (origin, model), group in subset.groupby(["origin", "model"]):
        rows.append(
            {
                "origin": int(origin),
                "model": str(model),
                "n": int(len(group)),
                "actual_sum": float(np.abs(group["actual"]).sum()),
                "abs_error_sum": float(np.abs(group["actual"] - group["forecast"]).sum()),
                "wape": _wape(group),
            }
        )
    return pd.DataFrame(rows)


def leave_one_origin_out(subset: pd.DataFrame) -> pd.DataFrame:
    """WAPE per model with each origin dropped in turn, and that run's leader."""
    rows = []
    for dropped in sorted(subset["origin"].unique()):
        rest = subset[subset["origin"] != dropped]
        if rest.empty:
            continue
        scores = {str(model): _wape(group) for model, group in rest.groupby("model")}
        finite = {k: v for k, v in scores.items() if np.isfinite(v)}
        leader = min(finite, key=finite.get) if finite else None
        for model, value in scores.items():
            rows.append(
                {
                    "dropped_origin": int(dropped),
                    "model": model,
                    "wape": value,
                    "leader": leader,
                }
            )
    return pd.DataFrame(rows)


def matched_origin_sensitivity(backtest: pd.DataFrame, reference_model: str = "bayesian") -> dict:
    """Summarise how fragile the matched-origin ranking is."""
    subset = matched_subset(backtest, reference_model)
    if subset.empty:
        return {"available": False, "reason": f"No {reference_model} backtest rows."}

    overall = {str(m): _wape(g) for m, g in subset.groupby("model")}
    ranked = sorted((v, k) for k, v in overall.items() if np.isfinite(v))
    leader = ranked[0][1] if ranked else None
    runner_up = ranked[1][1] if len(ranked) > 1 else None

    origin_table = per_origin_wape(subset)
    loo = leave_one_origin_out(subset)
    loo_leaders = (
        loo.drop_duplicates("dropped_origin").set_index("dropped_origin")["leader"].to_dict()
        if not loo.empty
        else {}
    )

    gap_share = None
    dominant_origin = None
    challenger = reference_model if leader != reference_model and reference_model in overall else runner_up
    if leader and challenger:
        errors = origin_table.pivot(index="origin", columns="model", values="abs_error_sum")
        per_origin_gap = errors[challenger] - errors[leader]
        total_gap = float(per_origin_gap.sum())
        if total_gap > 0:
            dominant_origin = int(per_origin_gap.idxmax())
            gap_share = float(per_origin_gap.max() / total_gap)

    n_forecasts = int(subset.groupby("model").size().min())
    n_origins = int(subset["origin"].nunique())
    stable = bool(loo_leaders) and all(value == leader for value in loo_leaders.values())
    fragile = (n_forecasts < MIN_FORECASTS_FOR_RANKING) or not stable or (
        gap_share is not None and gap_share > 0.5
    )

    reference_wape = overall.get(reference_model)
    verdict_parts = [
        f"{n_forecasts} forecasts over {n_origins} origins per model.",
        f"Lowest matched WAPE: {leader} ({overall.get(leader, float('nan')):.3f}); "
        f"{reference_model} {reference_wape:.3f}." if leader and reference_wape is not None else "",
    ]
    if dominant_origin is not None and gap_share is not None:
        without = loo[loo["dropped_origin"] == dominant_origin].set_index("model")["wape"]
        remaining = (
            f" Without it, WAPE is {leader} {without.get(leader, float('nan')):.3f} "
            f"vs {challenger} {without.get(challenger, float('nan')):.3f}."
        )
        if gap_share >= 1.0:
            verdict_parts.append(
                f"The whole absolute-error gap between {leader} and {challenger} "
                f"comes from origin {dominant_origin}.{remaining}"
            )
        else:
            verdict_parts.append(
                f"Origin {dominant_origin} alone accounts for {gap_share:.0%} of the "
                f"absolute-error gap between {leader} and {challenger}.{remaining}"
            )
    if loo_leaders:
        verdict_parts.append(
            "Leader with each origin left out: "
            + ", ".join(f"drop {o} -> {m}" for o, m in loo_leaders.items())
            + "."
        )
    verdict_parts.append(
        "The ranking is fragile and should be read as anecdotal, not as evidence "
        "that one model is better."
        if fragile
        else "The ranking survives dropping any single origin."
    )

    return {
        "available": True,
        "reference_model": reference_model,
        "n_forecasts_per_model": n_forecasts,
        "n_origins": n_origins,
        "matched_wape": overall,
        "leader": leader,
        "runner_up": runner_up,
        "gap_compared_against": challenger,
        "leave_one_origin_out_leaders": {str(k): v for k, v in loo_leaders.items()},
        "leader_stable_under_leave_one_out": stable,
        "dominant_origin": dominant_origin,
        "dominant_origin_gap_share": gap_share,
        "fragile": bool(fragile),
        "verdict": " ".join(part for part in verdict_parts if part),
        "_per_origin": origin_table,
        "_leave_one_out": loo,
    }


def write_sensitivity(backtest: pd.DataFrame, report_dir: Path) -> dict:
    """Write per-origin and leave-one-out tables plus a JSON summary per target."""
    report_dir = Path(report_dir)
    summaries: dict = {}
    per_origin_frames, loo_frames = [], []
    for target, group in backtest.groupby("target"):
        result = matched_origin_sensitivity(group)
        if result.get("available"):
            per_origin_frames.append(result["_per_origin"].assign(target=target))
            loo_frames.append(result["_leave_one_out"].assign(target=target))
        summaries[str(target)] = {k: v for k, v in result.items() if not k.startswith("_")}
    if per_origin_frames:
        pd.concat(per_origin_frames, ignore_index=True).to_csv(
            report_dir / "matched_origin_per_origin.csv", index=False
        )
        pd.concat(loo_frames, ignore_index=True).to_csv(
            report_dir / "matched_origin_leave_one_out.csv", index=False
        )
    (report_dir / "matched_origin_sensitivity.json").write_text(
        json.dumps(summaries, indent=2, default=str)
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-origin ranking sensitivity")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/demo_forecasting"))
    args = parser.parse_args()
    backtest = pd.read_csv(args.report_dir / "backtest_results.csv")
    summaries = write_sensitivity(backtest, args.report_dir)
    for target, summary in summaries.items():
        print(f"{target}: {summary.get('verdict', summary.get('reason'))}")


if __name__ == "__main__":
    main()


__all__ = [
    "MIN_FORECASTS_FOR_RANKING",
    "leave_one_origin_out",
    "matched_origin_sensitivity",
    "matched_subset",
    "per_origin_wape",
    "write_sensitivity",
]

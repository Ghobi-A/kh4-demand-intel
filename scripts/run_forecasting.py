"""Run the behavioural-demand forecasting experiment.

Canonical run over the full local dataset:

    python scripts/build_temporal_dataset.py --input data/processed/signals_scored.csv
    python scripts/run_forecasting.py

Smoke run over the stratified sample (NOT a substantive result):

    python scripts/build_temporal_dataset.py --demo
    python scripts/run_forecasting.py --demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.forecasting.data import DEFAULT_TARGETS  # noqa: E402
from src.forecasting.experiment import ForecastConfig, run_experiment  # noqa: E402
from src.temporal import DEFAULT_OUTPUT_PATH, DEMO_OUTPUT_PATH  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast behavioural demand proxies")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--min-train", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--forecast-horizon", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--no-bayesian", action="store_true")
    parser.add_argument(
        "--save-posterior",
        action="store_true",
        help="Write posterior samples locally (git-ignored) for scenario analysis",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Smoke run over the stratified sample; writes to reports/demo_forecasting/",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    input_path = args.input or (DEMO_OUTPUT_PATH if args.demo else DEFAULT_OUTPUT_PATH)

    config = ForecastConfig(
        targets=tuple(args.targets),
        min_train=args.min_train,
        horizon=args.horizon,
        step=args.step,
        forecast_horizon=args.forecast_horizon,
        seed=args.seed,
        max_origins=args.max_origins,
        include_bayesian=not args.no_bayesian,
        bayesian_draws=args.draws,
        bayesian_tune=args.tune,
        bayesian_chains=args.chains,
    )
    outcome = run_experiment(
        input_path=input_path,
        report_dir=args.report_dir,
        config=config,
        demo=args.demo,
        save_posterior=args.save_posterior,
    )

    print(f"Report directory: {outcome['report_dir']}")
    for target, result in outcome["per_target"].items():
        if result.get("usable"):
            print(f"- {target}: selected {result['selected_model']} over {len(result['origins'])} origins")
        else:
            print(f"- {target}: not modelled ({result.get('reason')})")


if __name__ == "__main__":
    main()

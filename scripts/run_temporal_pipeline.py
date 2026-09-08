"""One command to regenerate every temporal, forecasting and MMM artefact.

Canonical run over the full local dataset:

    python scripts/run_temporal_pipeline.py --input data/processed/signals_scored.csv

Smoke run over the committed stratified sample (NOT a substantive result):

    python scripts/run_temporal_pipeline.py --demo

The synthetic marketing lab is independent of the real data and always runs
unless --skip-mmm is passed.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger(__name__)


def _run(command: list[str]) -> None:
    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate all temporal and marketing artefacts")
    parser.add_argument("--input", type=Path, default=None, help="Scored signals CSV")
    parser.add_argument("--demo", action="store_true", help="Use the committed stratified sample")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--skip-mmm", action="store_true")
    parser.add_argument("--skip-forecasting", action="store_true")
    parser.add_argument(
        "--save-posterior",
        action="store_true",
        help="Also write git-ignored posterior samples so scenarios can run",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    python = sys.executable

    build = [python, "scripts/build_temporal_dataset.py"]
    if args.demo:
        build.append("--demo")
    if args.input:
        build += ["--input", str(args.input)]
    _run(build)

    if not args.skip_forecasting:
        forecast = [
            python, "scripts/run_forecasting.py",
            "--draws", str(args.draws),
            "--tune", str(args.tune),
            "--chains", str(args.chains),
        ]
        if args.demo:
            forecast.append("--demo")
        if args.save_posterior:
            forecast.append("--save-posterior")
        _run(forecast)

    if not args.skip_mmm:
        mmm = [
            python, "scripts/run_mmm.py",
            "--draws", str(args.draws),
            "--tune", str(args.tune),
            "--chains", str(args.chains),
        ]
        if args.save_posterior:
            mmm.append("--save-posterior")
        _run(mmm)

    print("\nPipeline complete.")
    if args.demo:
        print(
            "Outputs are demo-stamped smoke results from the stratified sample. "
            "They are not substantive forecasting results."
        )


if __name__ == "__main__":
    main()

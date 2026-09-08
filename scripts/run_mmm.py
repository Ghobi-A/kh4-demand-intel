"""Run the synthetic marketing science lab end to end.

SYNTHETIC MARKETING SCIENCE DEMONSTRATION. This is not Square Enix data.

    python scripts/run_mmm.py
    python scripts/run_mmm.py --correlated-channels   # harder identifiability
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mmm import SYNTHETIC_NOTICE  # noqa: E402
from src.mmm.evaluation import evaluate_recovery, identifiability_notes  # noqa: E402
from src.mmm.model import MMMSamplerConfig, fit_mmm  # noqa: E402
from src.mmm.reporting import write_mmm_report  # noqa: E402
from src.mmm.scenarios import budget_reallocation, simulate_scenario  # noqa: E402
from src.mmm.synthetic import SyntheticMMMConfig, generate_synthetic_mmm  # noqa: E402

DEFAULT_REPORT_DIR = Path("reports/mmm")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic Bayesian MMM demonstration")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--weeks", type=int, default=104)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument(
        "--correlated-channels",
        action="store_true",
        help="Harder scenario: channel spends move together, so contributions are less identifiable",
    )
    parser.add_argument("--save-posterior", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(SYNTHETIC_NOTICE)

    config = SyntheticMMMConfig(
        seed=args.seed, n_weeks=args.weeks, correlated_channels=args.correlated_channels
    )
    data, truth = generate_synthetic_mmm(config)

    fit = fit_mmm(
        data,
        config=MMMSamplerConfig(
            draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed
        ),
    )

    recovery = evaluate_recovery(fit, truth)
    notes = identifiability_notes(fit, truth, recovery)
    scenario_result = simulate_scenario(
        fit, data, budget_reallocation("social_spend", "video_spend", 0.20)
    )

    fit.contributions.to_csv(report_dir / "contributions.csv", index=False)
    fit.response_curves.to_csv(report_dir / "response_curves.csv", index=False)
    recovery.to_csv(report_dir / "recovery.csv", index=False)
    (report_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "notice": SYNTHETIC_NOTICE,
                "diagnostics": fit.diagnostics,
                "posterior_predictive": fit.posterior_predictive,
                "sampler": fit.sampler,
            },
            indent=2,
            default=str,
        )
    )
    (report_dir / "scenario_example.json").write_text(
        json.dumps(scenario_result, indent=2, default=str)
    )

    metadata = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "notice": SYNTHETIC_NOTICE,
        "data_type": "synthetic",
        "is_square_enix_data": False,
        "generator_seed": config.seed,
        "ground_truth": truth,
        "sampler": fit.sampler,
        "diagnostics_summary": {
            key: value for key, value in fit.diagnostics.items() if key != "parameters"
        },
        "recovery_checks_passed": int(recovery["passed"].sum()),
        "recovery_checks_total": int(len(recovery)),
        "identifiability_notes": notes,
    }
    (report_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    if args.save_posterior:
        path = report_dir / "posterior_samples.npz"
        np.savez_compressed(
            path,
            **{
                key: value
                for key, value in fit.posterior.items()
                if isinstance(value, np.ndarray)
            },
        )
        print(f"Wrote posterior samples to {path}")

    write_mmm_report(fit, truth, recovery, notes, scenario_result, metadata, report_dir)

    print(f"Recovery checks passed: {metadata['recovery_checks_passed']}/{metadata['recovery_checks_total']}")
    print(f"Max R-hat: {fit.diagnostics.get('max_r_hat'):.4f}, divergences: {fit.diagnostics.get('divergences')}")
    print(f"Scenario difference: {scenario_result['difference_mean']:,.0f} "
          f"({scenario_result['difference_lower']:,.0f} to {scenario_result['difference_upper']:,.0f})")
    print(f"Report directory: {report_dir}")


if __name__ == "__main__":
    main()

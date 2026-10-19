"""Check whether a full-data forecasting evaluation can run, and say what is missing.

The committed forecasting results are a demo smoke run over the 304-row
class-stratified extract. A measured result needs the full canonical scored
signal set, which is not committed (it holds raw comment text and is
regenerated from the platform APIs). This script never fabricates or
substitutes data: it inspects what exists, classifies the state, and prints
the exact commands for the next step.

    python scripts/check_full_data_readiness.py

Writes ``reports/data_quality/full_data_readiness.{json,md}`` and exits 0 when
the canonical run can proceed (or has already run), 2 when it is blocked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.temporal import (  # noqa: E402
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    _youtube_collection_meta,
    is_known_demo_extract,
)
from src.timestamps import parse_timestamps  # noqa: E402

FORECAST_REPORT_DIR = Path("reports/forecasting")
DEFAULT_OUTPUT_DIR = Path("reports/data_quality")

REQUIRED_SIGNAL_COLUMNS = ["text", "timestamp", "source", "intent_label"]

STATUS_BLOCKED = "blocked_missing_input"
STATUS_INVALID = "blocked_invalid_input"
STATUS_READY = "ready_to_run"
STATUS_DONE = "full_data_results_present"

COMMANDS = [
    "cp data/raw/youtube/video_seed_list.example.csv data/raw/youtube/video_seed_list.csv",
    "python -m src.scraper_youtube --seed-list --max-results 100 --fetch-metadata",
    "python -m src.scraper_reddit  # optional; see README section 17",
    "python scripts/run_pipeline.py  # preprocess -> sentiment -> intent -> score",
    "python scripts/check_full_data_readiness.py",
    "python scripts/build_temporal_dataset.py --input data/processed/signals_scored.csv",
    "python scripts/run_forecasting.py --chains 4 --save-posterior",
    "python -m src.forecasting.sensitivity --report-dir reports/forecasting",
]

MISSING_INPUT_REQUIREMENTS = [
    f"`{DEFAULT_INPUT_PATH}`: the full scored signal set produced by "
    "`scripts/run_pipeline.py` (~4,831 rows when last built locally). Required "
    f"columns: {', '.join(f'`{c}`' for c in REQUIRED_SIGNAL_COLUMNS)}.",
    "Creation timestamps from the source API (YouTube `publishedAt`, Reddit "
    "`created_utc`) in `timestamp`; rows without one are excluded, never imputed.",
    "`YOUTUBE_API_KEY` in `.env` to regenerate YouTube comments "
    "(`data/raw/` is git-ignored).",
    "`reports/tables/youtube_fetch_manifest.csv` from the scraper, so the build "
    "can tell whether any video hit the `--max-results` cap (seed-capped "
    "collections are stamped `volume_representative = false`).",
]


def inspect_signals(path: Path) -> dict:
    """Profile the canonical scored signal file, if present."""
    if not path.exists():
        return {"present": False, "path": str(path)}
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_SIGNAL_COLUMNS if c not in frame.columns]
    info: dict = {
        "present": True,
        "path": str(path),
        "rows": int(len(frame)),
        "missing_columns": missing,
        "is_demo_extract": bool(is_known_demo_extract(path, frame)),
    }
    if "source" in frame.columns:
        info["source_distribution"] = frame["source"].value_counts().to_dict()
    if "timestamp" in frame.columns:
        _, report = parse_timestamps(frame["timestamp"])
        info["timestamps"] = {
            "valid": int(report.valid),
            "missing": int(report.missing),
            "invalid": int(report.invalid),
        }
    return info


def assess(
    signals_path: Path = DEFAULT_INPUT_PATH,
    weekly_path: Path = DEFAULT_OUTPUT_PATH,
    forecast_dir: Path = FORECAST_REPORT_DIR,
) -> dict:
    signals = inspect_signals(signals_path)
    weekly_meta_path = weekly_path.with_suffix(".meta.json")
    weekly_meta = json.loads(weekly_meta_path.read_text()) if weekly_meta_path.exists() else None
    forecast_meta_path = forecast_dir / "run_metadata.json"
    forecast_meta = (
        json.loads(forecast_meta_path.read_text()) if forecast_meta_path.exists() else None
    )
    collection = _youtube_collection_meta()

    problems: list[str] = []
    if not signals["present"]:
        status = STATUS_BLOCKED
        problems.append(f"{signals_path} does not exist in this checkout.")
    else:
        if signals["missing_columns"]:
            problems.append(f"Missing required columns: {signals['missing_columns']}")
        if signals["is_demo_extract"]:
            problems.append(
                "The file is a committed demo extract, not the canonical dataset."
            )
        if signals.get("timestamps", {}).get("valid", 0) == 0:
            problems.append("No row has a valid creation timestamp.")
        status = STATUS_INVALID if problems else STATUS_READY

    if (
        status == STATUS_READY
        and forecast_meta
        and not forecast_meta.get("sample_only")
        and forecast_meta.get("evaluation_status") in {"full_dataset", "seed_capped_dataset"}
    ):
        status = STATUS_DONE

    if not collection.get("manifest_found"):
        problems.append(
            "No YouTube fetch manifest, so volume representativeness cannot be "
            "verified; a full run would be stamped full_dataset on trust."
        )

    return {
        "status": status,
        "signals": signals,
        "weekly_table": {
            "present": weekly_path.exists(),
            "evaluation_status": (weekly_meta or {}).get("evaluation_status"),
            "volume_representative": (weekly_meta or {}).get("volume_representative"),
        },
        "full_forecast_results": {
            "present": forecast_meta is not None,
            "evaluation_status": (forecast_meta or {}).get("evaluation_status"),
        },
        "youtube_collection": collection,
        "problems": problems,
        "missing_input_requirements": MISSING_INPUT_REQUIREMENTS
        if status in {STATUS_BLOCKED, STATUS_INVALID}
        else [],
        "commands": COMMANDS,
        "note": (
            "No forecasting result is reported until the canonical run completes. "
            "The committed reports/demo_forecasting/ results are a demo smoke run "
            "over a class-stratified sample and must not be quoted as measured."
        ),
    }


def write_report(result: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_data_readiness.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    lines = [
        "# Full-data forecasting readiness",
        "",
        f"**Status: `{result['status']}`**",
        "",
        result["note"],
        "",
        "## What exists",
        "",
        f"- Canonical signals (`{result['signals']['path']}`): "
        + (
            f"present, {result['signals']['rows']:,} rows"
            if result["signals"]["present"]
            else "**absent**"
        ),
        f"- Weekly table: {'present' if result['weekly_table']['present'] else 'absent'}"
        f" (status `{result['weekly_table']['evaluation_status']}`)",
        "- Full forecasting results: "
        + ("present" if result["full_forecast_results"]["present"] else "absent"),
        "- YouTube fetch manifest: "
        + ("present" if result["youtube_collection"].get("manifest_found") else "absent"),
        "",
    ]
    if result["problems"]:
        lines += ["## Problems", ""] + [f"- {p}" for p in result["problems"]] + [""]
    if result["missing_input_requirements"]:
        lines += ["## Missing inputs", ""]
        lines += [f"- {item}" for item in result["missing_input_requirements"]] + [""]
    lines += ["## Commands", "", "```bash", *result["commands"], "```", ""]
    (output_dir / "full_data_readiness.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--signals", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = assess(signals_path=args.signals)
    write_report(result, args.output_dir)
    print(f"Status: {result['status']}")
    for problem in result["problems"]:
        print(f"- {problem}")
    print(f"Report: {args.output_dir / 'full_data_readiness.md'}")
    sys.exit(0 if result["status"] in {STATUS_READY, STATUS_DONE} else 2)


if __name__ == "__main__":
    main()

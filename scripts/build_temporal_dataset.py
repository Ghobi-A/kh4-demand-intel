"""Build the weekly demand-signal table from scored KH4 signals.

Canonical run (full local dataset):

    python scripts/build_temporal_dataset.py --input data/processed/signals_scored.csv

Smoke run (stratified sample; NOT a substantive result):

    python scripts/build_temporal_dataset.py --demo
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.temporal import main  # noqa: E402

if __name__ == "__main__":
    main()

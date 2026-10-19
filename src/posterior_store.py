"""Small, run-stamped posterior samples that can be committed and deployed.

Full posterior samples (``posterior_samples*.npz``) are large and git-ignored,
so a deployed dashboard or API has nothing to run scenarios against. Every
forecasting and MMM run therefore also writes a *thinned* copy: at most
``MAX_DRAWS`` draws, stored as float32, stamped with the run ID that produced
it. A reader passes the run ID from that run's ``run_metadata.json``; a
mismatched stamp means the thinned file belongs to a different run, and it is
refused rather than silently paired with the wrong report.

Thinning keeps every Nth draw, which preserves the posterior's shape; with 1,000
draws, scenario means and 90% intervals move by Monte Carlo error only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_DRAWS = 1000
RUN_ID_KEY = "__run_id__"
TOTAL_DRAWS_KEY = "__source_draws__"


class StalePosteriorError(RuntimeError):
    """Raised when thinned samples were produced by a different run."""


def thin_draws(arrays: dict[str, np.ndarray], max_draws: int = MAX_DRAWS) -> tuple[dict, int]:
    """Keep evenly spaced draws along axis 0 of every array that has them.

    Arrays whose first axis is not the draw axis (for example a per-channel
    scale vector) are kept whole.
    """
    lengths = [np.asarray(v).shape[0] for v in arrays.values() if np.ndim(v) >= 1]
    total = max(lengths) if lengths else 0
    step = max(1, int(np.ceil(total / max_draws))) if total else 1
    thinned = {}
    for key, value in arrays.items():
        value = np.asarray(value)
        if value.ndim >= 1 and value.shape[0] == total and total > 1:
            value = value[::step]
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32)
        thinned[key] = value
    return thinned, total


def save_thinned(path: Path, arrays: dict[str, np.ndarray], run_id: str,
                 max_draws: int = MAX_DRAWS) -> Path:
    thinned, total = thin_draws(arrays, max_draws)
    thinned[RUN_ID_KEY] = np.array(str(run_id))
    thinned[TOTAL_DRAWS_KEY] = np.array(int(total))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **thinned)
    return path


def load_thinned(path: Path, expected_run_id: str | None) -> dict[str, np.ndarray] | None:
    """Arrays from a thinned file, or None if absent; raises if it is stale."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as payload:
        stamp = str(payload[RUN_ID_KEY]) if RUN_ID_KEY in payload.files else None
        if expected_run_id is None or stamp != str(expected_run_id):
            raise StalePosteriorError(
                f"{path} was produced by run {stamp!r}, but the report it would be "
                f"paired with is run {expected_run_id!r}."
            )
        return {
            key: payload[key].astype(np.float64)
            if np.issubdtype(payload[key].dtype, np.floating) else payload[key]
            for key in payload.files
            if key not in (RUN_ID_KEY, TOTAL_DRAWS_KEY)
        }


__all__ = ["MAX_DRAWS", "StalePosteriorError", "load_thinned", "save_thinned", "thin_draws"]

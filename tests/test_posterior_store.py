"""Tests for run-stamped thinned posterior samples and their use in scenarios."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import services
from src.posterior_store import StalePosteriorError, load_thinned, save_thinned, thin_draws


def test_thinning_keeps_evenly_spaced_draws_and_whole_non_draw_arrays() -> None:
    arrays = {"beta": np.arange(4000 * 3, dtype=float).reshape(4000, 3), "spend_scale": np.ones(3)}
    thinned, total = thin_draws(arrays, max_draws=1000)
    assert total == 4000
    assert thinned["beta"].shape == (1000, 3)
    assert thinned["beta"].dtype == np.float32
    assert thinned["spend_scale"].shape == (3,)
    np.testing.assert_allclose(thinned["beta"][1], arrays["beta"][4])


def test_thinned_samples_load_only_for_their_own_run(tmp_path) -> None:
    path = save_thinned(tmp_path / "p.npz", {"draws": np.ones((10, 4))}, run_id="abc123")
    assert load_thinned(path, "abc123")["draws"].shape == (10, 4)
    with pytest.raises(StalePosteriorError):
        load_thinned(path, "different")
    assert load_thinned(tmp_path / "missing.npz", "abc123") is None


def test_behavioural_scenario_runs_from_thinned_samples(tmp_path) -> None:
    (tmp_path / "run_metadata.json").write_text(json.dumps({"run_id": "r1", "sample_only": True}))
    rng = np.random.default_rng(0)
    save_thinned(
        tmp_path / "posterior_thinned_actionable_probability_mass.npz",
        {"draws": rng.gamma(2.0, 1.0, size=(400, 4))},
        run_id="r1",
    )
    result = services.behavioural_scenario(level_shift=1.2, report_dir=tmp_path)
    assert result["sample_only"] is True
    assert result["total_difference_mean"] > 0


def test_stale_thinned_samples_are_refused_with_a_clear_error(tmp_path) -> None:
    (tmp_path / "run_metadata.json").write_text(json.dumps({"run_id": "new-run"}))
    save_thinned(
        tmp_path / "posterior_thinned_actionable_probability_mass.npz",
        {"draws": np.ones((5, 4))},
        run_id="old-run",
    )
    with pytest.raises(services.ArtefactMissingError, match="stale"):
        services.behavioural_scenario(report_dir=tmp_path)


def test_committed_thinned_posteriors_match_their_reports() -> None:
    """Generated-artefact consistency: committed samples belong to committed runs."""
    for report_dir, name in [
        (services.DEMO_FORECAST_REPORT_DIR, "posterior_thinned_actionable_probability_mass.npz"),
        (services.MMM_REPORT_DIR, services.MMM_THINNED_POSTERIOR),
    ]:
        run_id = json.loads((report_dir / "run_metadata.json").read_text())["run_id"]
        assert load_thinned(report_dir / name, run_id), report_dir

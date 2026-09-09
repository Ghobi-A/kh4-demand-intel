"""Regression tests for posterior predictive checking.

These exist because a real defect shipped here: the summary was being fed
posterior draws of the *mean* rather than predictive replicates, which made a
well-calibrated model look badly calibrated (coverage ~0.45 against a 0.90
nominal level).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bayes_diagnostics import posterior_predictive_summary, summarise_trace


def _mean_and_replicate_draws(seed: int = 0, n: int = 300, sigma: float = 3.0):
    rng = np.random.default_rng(seed)
    observed = 10.0 + rng.normal(0.0, sigma, n)
    mean_draws = 10.0 + rng.normal(0.0, sigma / np.sqrt(n), (400, n))
    replicates = mean_draws + rng.normal(0.0, sigma, (400, n))
    return observed, mean_draws, replicates


def test_replicates_recover_the_nominal_coverage() -> None:
    observed, _mean_draws, replicates = _mean_and_replicate_draws()

    summary = posterior_predictive_summary(observed, replicates, interval_level=0.9)

    assert summary["ppc_interval_coverage"] == pytest.approx(0.9, abs=0.06)
    assert summary["nominal_interval_level"] == 0.9


def test_summarising_the_mean_instead_of_replicates_understates_coverage() -> None:
    """The exact defect that produced the old ~0.45 coverage figure."""
    observed, mean_draws, replicates = _mean_and_replicate_draws()

    mean_coverage = posterior_predictive_summary(observed, mean_draws)["ppc_interval_coverage"]
    replicate_coverage = posterior_predictive_summary(observed, replicates)[
        "ppc_interval_coverage"
    ]

    assert mean_coverage < 0.5
    assert replicate_coverage > 0.8


def test_predicted_spread_is_comparable_with_observed_spread() -> None:
    observed, mean_draws, replicates = _mean_and_replicate_draws()

    from_replicates = posterior_predictive_summary(observed, replicates)
    from_means = posterior_predictive_summary(observed, mean_draws)

    assert from_replicates["predicted_std"] == pytest.approx(
        from_replicates["observed_std"], rel=0.25
    )
    # Draws of the mean are far tighter than the data, which is the tell.
    assert from_means["predicted_std"] < from_means["observed_std"] / 5


def test_interval_level_is_honoured() -> None:
    observed, _m, replicates = _mean_and_replicate_draws()

    narrow = posterior_predictive_summary(observed, replicates, interval_level=0.5)
    wide = posterior_predictive_summary(observed, replicates, interval_level=0.99)

    assert narrow["ppc_interval_coverage"] < wide["ppc_interval_coverage"]
    assert narrow["ppc_interval_coverage"] == pytest.approx(0.5, abs=0.08)
    assert wide["ppc_interval_coverage"] == pytest.approx(0.99, abs=0.03)


def test_zero_share_is_measured_from_the_replicates() -> None:
    """A hurdle model must be able to reproduce empty periods."""
    rng = np.random.default_rng(1)
    observed = np.where(rng.random(200) < 0.6, 0.0, rng.gamma(3.0, 2.0, 200))
    replicates = np.where(
        rng.random((300, 200)) < 0.6, 0.0, rng.gamma(3.0, 2.0, (300, 200))
    )

    summary = posterior_predictive_summary(observed, replicates)

    assert summary["observed_zero_share"] == pytest.approx(0.6, abs=0.1)
    assert summary["predicted_zero_share"] == pytest.approx(
        summary["observed_zero_share"], abs=0.1
    )


def test_mean_only_draws_can_never_reproduce_a_zero_share() -> None:
    """Why the old check reported a 0% predicted zero share against 64% observed."""
    rng = np.random.default_rng(2)
    observed = np.where(rng.random(200) < 0.6, 0.0, rng.gamma(3.0, 2.0, 200))
    positive_means = np.full((300, 200), 6.0)

    summary = posterior_predictive_summary(observed, positive_means)

    assert summary["observed_zero_share"] > 0.5
    assert summary["predicted_zero_share"] == 0.0


def test_empty_samples_return_an_empty_summary() -> None:
    assert posterior_predictive_summary(np.array([1.0, 2.0]), np.empty((0, 2))) == {}


def test_diagnostic_thresholds_flag_bad_fits(monkeypatch) -> None:
    """A fit that has not converged must be reported as such, not smoothed over."""
    import sys
    import types

    import pandas as pd

    summary_frame = pd.DataFrame(
        {"r_hat": [1.05], "ess_bulk": [50.0], "ess_tail": [60.0]}, index=["intercept"]
    )
    monkeypatch.setitem(
        sys.modules, "arviz", types.SimpleNamespace(summary=lambda *a, **k: summary_frame)
    )

    trace = types.SimpleNamespace(
        sample_stats={"diverging": types.SimpleNamespace(values=np.array([[1, 0], [0, 1]]))}
    )

    result = summarise_trace(trace)

    assert result["converged"] is False
    assert result["divergences"] == 2
    assert result["max_r_hat"] == pytest.approx(1.05)
    assert any("R-hat" in warning for warning in result["warnings"])
    assert any("ESS" in warning for warning in result["warnings"])


def test_a_clean_fit_reports_converged(monkeypatch) -> None:
    import sys
    import types

    import pandas as pd

    summary_frame = pd.DataFrame(
        {"r_hat": [1.001], "ess_bulk": [1500.0], "ess_tail": [1400.0]}, index=["intercept"]
    )
    monkeypatch.setitem(
        sys.modules, "arviz", types.SimpleNamespace(summary=lambda *a, **k: summary_frame)
    )
    trace = types.SimpleNamespace(
        sample_stats={"diverging": types.SimpleNamespace(values=np.zeros((2, 10)))}
    )

    result = summarise_trace(trace)

    assert result["converged"] is True
    assert result["divergences"] == 0
    assert result["warnings"] == []

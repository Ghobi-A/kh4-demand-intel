"""Tests for geometric adstock and Hill saturation."""

from __future__ import annotations

import numpy as np
import pytest

from src.mmm.transforms import (
    adstock_channels,
    geometric_adstock,
    hill_saturation,
    media_response,
)


def test_adstock_impulse_decays_geometrically() -> None:
    impulse = np.zeros(5)
    impulse[0] = 1.0

    result = geometric_adstock(impulse, decay=0.5)

    assert result.tolist() == pytest.approx([1.0, 0.5, 0.25, 0.125, 0.0625])


def test_zero_decay_leaves_spend_untouched() -> None:
    assert geometric_adstock([1.0, 2.0, 3.0], decay=0.0).tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_higher_decay_carries_more_into_later_periods() -> None:
    impulse = np.array([1.0, 0.0, 0.0, 0.0])

    slow = geometric_adstock(impulse, decay=0.2)
    fast = geometric_adstock(impulse, decay=0.8)

    assert fast[-1] > slow[-1]


def test_adstock_never_reduces_the_period_it_is_applied_to() -> None:
    spend = np.array([3.0, 1.0, 4.0, 0.0])

    result = geometric_adstock(spend, decay=0.5)

    assert (result >= spend).all()


def test_normalised_adstock_preserves_the_spend_scale() -> None:
    spend = np.full(60, 10.0)

    result = geometric_adstock(spend, decay=0.6, normalise=True)

    # A steady spend level converges back to that level.
    assert result[-1] == pytest.approx(10.0, rel=1e-6)


def test_adstock_is_deterministic() -> None:
    spend = np.array([1.0, 5.0, 2.0])

    assert geometric_adstock(spend, 0.4).tolist() == geometric_adstock(spend, 0.4).tolist()


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5, float("nan")])
def test_invalid_decay_is_rejected(decay: float) -> None:
    with pytest.raises(ValueError):
        geometric_adstock([1.0, 2.0], decay=decay)


def test_invalid_spend_is_rejected() -> None:
    with pytest.raises(ValueError):
        geometric_adstock([-1.0, 2.0], decay=0.5)
    with pytest.raises(ValueError):
        geometric_adstock([1.0, np.nan], decay=0.5)
    with pytest.raises(ValueError):
        geometric_adstock([[1.0, 2.0]], decay=0.5)


def test_channel_specific_decays_are_applied_per_column() -> None:
    matrix = np.zeros((4, 2))
    matrix[0, :] = 1.0

    result = adstock_channels(matrix, decays=[0.0, 0.5])

    assert result[:, 0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert result[:, 1].tolist() == pytest.approx([1.0, 0.5, 0.25, 0.125])


def test_channel_decay_count_must_match() -> None:
    with pytest.raises(ValueError):
        adstock_channels(np.zeros((4, 3)), decays=[0.5, 0.5])


def test_saturation_is_monotonic_in_spend() -> None:
    spend = np.linspace(0.0, 50.0, 40)

    response = hill_saturation(spend, alpha=1.5, theta=10.0)

    assert (np.diff(response) >= 0).all()


def test_saturation_is_zero_at_zero_spend() -> None:
    assert hill_saturation([0.0], alpha=2.0, theta=5.0)[0] == 0.0


def test_saturation_approaches_one_at_high_spend() -> None:
    assert hill_saturation([1e9], alpha=2.0, theta=5.0)[0] == pytest.approx(1.0, abs=1e-6)


def test_half_saturation_point_returns_one_half() -> None:
    assert hill_saturation([7.0], alpha=1.7, theta=7.0)[0] == pytest.approx(0.5)


def test_marginal_return_falls_as_spend_grows() -> None:
    """The economic claim: the tenth impression is worth less than the first."""
    spend = np.linspace(0.1, 100.0, 200)
    increments = np.diff(hill_saturation(spend, alpha=1.4, theta=20.0))

    assert increments[-1] < increments[len(increments) // 2]


@pytest.mark.parametrize(
    ("alpha", "theta"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0), (float("nan"), 1.0)]
)
def test_invalid_saturation_parameters_are_rejected(alpha: float, theta: float) -> None:
    with pytest.raises(ValueError):
        hill_saturation([1.0], alpha=alpha, theta=theta)


def test_negative_saturation_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        hill_saturation([-1.0], alpha=1.0, theta=1.0)


def test_media_response_chains_adstock_then_saturation() -> None:
    spend = np.array([10.0, 0.0, 0.0, 0.0])

    response = media_response(spend, decay=0.5, alpha=1.0, theta=5.0)

    assert (response >= 0).all() and (response <= 1).all()
    # Carryover means later periods still respond despite zero spend.
    assert response[1] > 0


def test_pytensor_adstock_matches_the_numpy_implementation() -> None:
    """The estimated transform must be the same one the generator used."""
    pytest.importorskip("pytensor")
    import pytensor.tensor as pt

    from src.mmm.model import _pt_adstock

    spend = np.array([5.0, 0.0, 0.0, 3.0, 1.0, 0.0, 2.0, 0.0])
    for decay in [0.0, 0.3, 0.6, 0.9]:
        symbolic = _pt_adstock(
            pt, pt.as_tensor_variable(spend), pt.as_tensor_variable(decay), len(spend)
        ).eval()
        assert symbolic == pytest.approx(geometric_adstock(spend, decay, normalise=True))

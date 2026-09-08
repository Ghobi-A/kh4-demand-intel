"""Media transforms: geometric adstock and Hill saturation.

Two facts about media response that a plain linear regression cannot express:

* **Adstock (carryover).** Advertising seen this week still influences
  behaviour next week. Geometric adstock carries a decaying fraction forward:

      A_t = X_t + decay * A_(t-1)

* **Saturation (diminishing returns).** The tenth impression is worth less
  than the first. Hill saturation bends the response towards a ceiling:

      response(x) = x^alpha / (x^alpha + theta^alpha)

Both are implemented in NumPy here for the data generator and for testing.
``src.mmm.model`` mirrors them in PyTensor so the same shapes are estimable.
"""

from __future__ import annotations

import numpy as np


def geometric_adstock(spend, decay: float, normalise: bool = False) -> np.ndarray:
    """Carry a decaying fraction of past spend into later periods.

    Args:
        spend: Per-period spend for one channel.
        decay: Fraction carried forward each period, in [0, 1).
        normalise: Divide by ``1 / (1 - decay)`` so total adstock preserves the
            spend scale, which keeps coefficients comparable across channels.
    """
    if not np.isscalar(decay):
        raise TypeError("decay must be a scalar; use adstock_channels for per-channel decays")
    decay = float(decay)
    if not np.isfinite(decay) or decay < 0.0 or decay >= 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")

    values = np.asarray(spend, dtype=float)
    if values.ndim != 1:
        raise ValueError("spend must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("spend contains non-finite values")
    if (values < 0).any():
        raise ValueError("spend cannot be negative")

    result = np.empty_like(values)
    carry = 0.0
    for index, value in enumerate(values):
        carry = value + decay * carry
        result[index] = carry
    if normalise:
        result = result * (1.0 - decay)
    return result


def adstock_channels(spend_matrix, decays, normalise: bool = False) -> np.ndarray:
    """Apply a channel-specific decay to each column of a spend matrix."""
    matrix = np.asarray(spend_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("spend_matrix must be two-dimensional (periods x channels)")
    decay_array = np.atleast_1d(np.asarray(decays, dtype=float))
    if decay_array.size != matrix.shape[1]:
        raise ValueError(
            f"Expected one decay per channel: {matrix.shape[1]} channels, "
            f"{decay_array.size} decays"
        )
    return np.column_stack(
        [
            geometric_adstock(matrix[:, index], decay_array[index], normalise=normalise)
            for index in range(matrix.shape[1])
        ]
    )


def hill_saturation(x, alpha: float = 1.0, theta: float = 1.0) -> np.ndarray:
    """Diminishing returns: response rises to a ceiling of 1 as spend grows.

    ``theta`` is the half-saturation point (where response is 0.5) and
    ``alpha`` controls how sharply the curve bends.
    """
    alpha = float(alpha)
    theta = float(theta)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")

    values = np.asarray(x, dtype=float)
    if (values < 0).any():
        raise ValueError("saturation input cannot be negative")

    powered = np.power(values, alpha)
    return powered / (powered + theta**alpha)


def media_response(
    spend,
    decay: float,
    alpha: float,
    theta: float,
    normalise_adstock: bool = True,
) -> np.ndarray:
    """The full spend -> adstock -> saturation path for one channel."""
    return hill_saturation(
        geometric_adstock(spend, decay, normalise=normalise_adstock), alpha=alpha, theta=theta
    )


__all__ = ["adstock_channels", "geometric_adstock", "hill_saturation", "media_response"]

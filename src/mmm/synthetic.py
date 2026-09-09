"""Reproducible synthetic marketing dataset with known ground truth.

SYNTHETIC MARKETING SCIENCE DEMONSTRATION. This is not Square Enix data.

Kingdom Hearts IV is unreleased and no media spend, pricing, promotion or
sales data exists for it. Rather than invent numbers and present them as
observed, this module simulates a weekly marketing dataset from an explicit
data-generating process whose parameters are recorded alongside the output.
Knowing the truth is what makes the model checkable: ``src.mmm.evaluation``
asks whether the fitted model recovers the process that produced the data.

Scope is fixed at three media channels plus price, promotion, an event and
seasonality — enough to demonstrate adstock, saturation, contribution,
identifiability and budget reallocation, and no more.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.mmm import DATA_TYPE, SYNTHETIC_NOTICE
from src.mmm.transforms import media_response

DEFAULT_DATA_PATH = Path("data/synthetic/mmm_weekly_synthetic.csv")
DEFAULT_TRUTH_PATH = Path("data/synthetic/mmm_ground_truth.json")

CHANNELS = ("video_spend", "paid_search_spend", "social_spend")


@dataclass
class ChannelSpec:
    """True media parameters for one simulated channel."""

    name: str
    base_spend: float
    spend_volatility: float
    adstock_decay: float
    saturation_alpha: float
    saturation_theta: float
    beta: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SyntheticMMMConfig:
    """The full data-generating process, recorded with every output."""

    seed: int = 20260101
    n_weeks: int = 104
    baseline_sales: float = 5000.0
    trend_per_week: float = 6.0
    seasonality_amplitude: float = 400.0
    seasonality_period: float = 52.0

    base_price: float = 59.99
    price_volatility: float = 3.0
    #: Sales change per unit of price; negative, because demand falls as price rises.
    price_coefficient: float = -55.0

    promotion_probability: float = 0.18
    promotion_lift: float = 900.0

    event_weeks: tuple = (20, 58, 88)
    event_lift: float = 1500.0
    event_duration: int = 2

    noise_sd: float = 250.0

    #: Correlate channel spends to create a harder identifiability scenario.
    correlated_channels: bool = False
    correlation_strength: float = 0.85

    channels: tuple = field(
        default_factory=lambda: (
            ChannelSpec("video_spend", 22000.0, 6000.0, 0.60, 1.4, 26000.0, 2600.0),
            ChannelSpec("paid_search_spend", 12000.0, 3000.0, 0.25, 1.1, 11000.0, 1500.0),
            ChannelSpec("social_spend", 8000.0, 2500.0, 0.45, 1.2, 9000.0, 900.0),
        )
    )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["channels"] = [channel.to_dict() for channel in self.channels]
        payload["event_weeks"] = list(self.event_weeks)
        return payload


def _spend_paths(config: SyntheticMMMConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Simulate weekly spend per channel, optionally correlated."""
    weeks = np.arange(config.n_weeks)
    shared = rng.normal(0.0, 1.0, config.n_weeks)
    paths: dict[str, np.ndarray] = {}
    for channel in config.channels:
        own = rng.normal(0.0, 1.0, config.n_weeks)
        if config.correlated_channels:
            # A common budget driver makes channels move together, which is what
            # makes real media data hard to identify.
            shock = config.correlation_strength * shared + (
                1 - config.correlation_strength
            ) * own
        else:
            shock = own
        # A mild campaign cycle keeps spend from being pure noise.
        cycle = 1.0 + 0.15 * np.sin(2 * np.pi * weeks / 26.0)
        values = channel.base_spend * cycle + channel.spend_volatility * shock
        paths[channel.name] = np.clip(values, 0.0, None)
    return paths


def generate_synthetic_mmm(
    config: SyntheticMMMConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate the weekly synthetic dataset and its ground-truth record."""
    config = config or SyntheticMMMConfig()
    if config.n_weeks < 12:
        raise ValueError("Need at least 12 weeks for a usable synthetic dataset")
    rng = np.random.default_rng(config.seed)
    weeks = np.arange(config.n_weeks)

    spends = _spend_paths(config, rng)

    contributions: dict[str, np.ndarray] = {}
    for channel in config.channels:
        response = media_response(
            spends[channel.name],
            decay=channel.adstock_decay,
            alpha=channel.saturation_alpha,
            theta=channel.saturation_theta,
        )
        contributions[channel.name] = channel.beta * response

    price = np.clip(
        config.base_price + rng.normal(0.0, config.price_volatility, config.n_weeks), 20.0, None
    )
    price_effect = config.price_coefficient * (price - config.base_price)

    promotion = (rng.random(config.n_weeks) < config.promotion_probability).astype(int)
    promotion_effect = config.promotion_lift * promotion

    event = np.zeros(config.n_weeks)
    for start in config.event_weeks:
        event[start : start + config.event_duration] = 1.0
    event_effect = config.event_lift * event

    seasonal = config.seasonality_amplitude * np.sin(
        2 * np.pi * weeks / config.seasonality_period
    )
    trend = config.trend_per_week * weeks
    noise = rng.normal(0.0, config.noise_sd, config.n_weeks)

    media_total = np.sum(list(contributions.values()), axis=0)
    simulated_sales = (
        config.baseline_sales
        + trend
        + seasonal
        + media_total
        + price_effect
        + promotion_effect
        + event_effect
        + noise
    )

    frame = pd.DataFrame({"week": weeks})
    for channel in config.channels:
        frame[channel.name] = spends[channel.name]
    frame["price"] = price
    frame["promotion"] = promotion
    frame["event"] = event
    frame["seasonal_component"] = seasonal
    frame["trend_component"] = trend
    frame["noise"] = noise
    frame["simulated_sales"] = simulated_sales
    frame["data_type"] = DATA_TYPE

    truth = {
        "notice": SYNTHETIC_NOTICE,
        "data_type": DATA_TYPE,
        "is_square_enix_data": False,
        "statement": "This is not Square Enix data.",
        "config": config.to_dict(),
        "true_contributions": {
            name: float(np.sum(values)) for name, values in contributions.items()
        },
        "true_contribution_share": {
            name: float(np.sum(values) / np.sum(simulated_sales))
            for name, values in contributions.items()
        },
        "true_contribution_ranking": [
            name
            for name, _ in sorted(
                ((n, float(np.sum(v))) for n, v in contributions.items()),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        "expected_price_direction": "negative",
        "expected_promotion_direction": "positive",
        "expected_event_direction": "positive",
    }
    return frame, truth


def write_synthetic_dataset(
    config: SyntheticMMMConfig | None = None,
    data_path: Path = DEFAULT_DATA_PATH,
    truth_path: Path = DEFAULT_TRUTH_PATH,
) -> tuple[Path, Path]:
    """Write the synthetic dataset and its ground truth to disk."""
    frame, truth = generate_synthetic_mmm(config)
    data_path = Path(data_path)
    truth_path = Path(truth_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_path, index=False)
    truth_path.write_text(json.dumps(truth, indent=2, default=str))
    return data_path, truth_path


def load_synthetic_dataset(
    data_path: Path = DEFAULT_DATA_PATH, truth_path: Path = DEFAULT_TRUTH_PATH
) -> tuple[pd.DataFrame, dict]:
    """Load the committed synthetic dataset, regenerating it if absent."""
    data_path = Path(data_path)
    truth_path = Path(truth_path)
    if not data_path.exists() or not truth_path.exists():
        write_synthetic_dataset(data_path=data_path, truth_path=truth_path)
    frame = pd.read_csv(data_path)
    truth = json.loads(truth_path.read_text())
    return frame, truth


def main() -> None:
    data_path, truth_path = write_synthetic_dataset()
    print(SYNTHETIC_NOTICE)
    print(f"Wrote synthetic dataset to {data_path}")
    print(f"Wrote ground-truth parameters to {truth_path}")


if __name__ == "__main__":
    main()

"""Experiment configuration (dataclass-based, YAML-loadable)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SplitConfig:
    method: str = "group"
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15


@dataclass
class ThresholdConfig:
    objective: str = "f1"
    min_precision: float | None = None
    min_recall: float | None = None


@dataclass
class ExperimentConfig:
    experiment_name: str = "tfidf_group_holdout"
    input_path: str = "reports/audits/intent_audit_corrected.csv"
    output_dir: str = "reports/ml"
    models: list[str] = field(
        default_factory=lambda: ["majority", "rules", "logistic", "linear_svm", "char_word_svm"]
    )
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    split: SplitConfig = field(default_factory=SplitConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    calibration_method: str = "sigmoid"
    ranking_percentages: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])
    ranking_k_values: list[int] = field(default_factory=lambda: [10, 25, 50, 100])
    evaluation_status: str = "development_preliminary"

    # Seeds control BOTH the group-split shuffle and model initialisation:
    # each seed regenerates the split and retrains every model.
    seed_scope: str = "split_and_initialisation"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
        split = SplitConfig(**raw.pop("split", {}))
        threshold = ThresholdConfig(**raw.pop("threshold", {}))
        known = {f for f in cls.__dataclass_fields__}
        return cls(split=split, threshold=threshold, **{k: v for k, v in raw.items() if k in known and k not in {"split", "threshold"}})

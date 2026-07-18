"""Optional transformer fine-tuning interface (requires the ``transformers`` extra).

Ready for use once the labelled corpus is expanded; results on the
current 200-row development corpus would be preliminary at best, so
transformer training is never run in CI and never mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.ml.models import SCORE_PROBABILITY, BaseTextClassifier

DEFAULT_TRANSFORMER_MODEL = "distilbert-base-uncased"


@dataclass
class TransformerTrainingConfig:
    model_name: str = DEFAULT_TRANSFORMER_MODEL
    max_sequence_length: int = 128
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    seed: int = 42
    early_stopping_patience: int = 2
    output_dir: Path = field(default_factory=lambda: Path("artifacts/transformer"))


def _require_transformers():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Transformer support requires torch and transformers. "
            "Install with: pip install .[transformers]"
        ) from exc


class TransformerClassifier(BaseTextClassifier):
    """Fine-tunable transformer with the unified classifier interface."""

    name = "transformer"
    score_type = SCORE_PROBABILITY

    def __init__(self, config: TransformerTrainingConfig | None = None):
        _require_transformers()
        self.config = config or TransformerTrainingConfig()
        self._model = None
        self._tokenizer = None
        self._label_list: list[str] = []

    def fit(self, texts, labels, validation_texts=None, validation_labels=None):
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )

        config = self.config
        self._label_list = sorted(set(labels))
        label_to_id = {label: i for i, label in enumerate(self._label_list)}
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=len(self._label_list)
        )

        def make_dataset(text_list, label_list):
            encodings = self._tokenizer(
                list(text_list),
                truncation=True,
                max_length=config.max_sequence_length,
                padding=True,
            )

            class _Dataset(torch.utils.data.Dataset):
                def __len__(self):
                    return len(label_list)

                def __getitem__(self, idx):
                    item = {k: torch.tensor(v[idx]) for k, v in encodings.items()}
                    item["labels"] = torch.tensor(label_to_id[label_list[idx]])
                    return item

            return _Dataset()

        has_validation = validation_texts is not None and validation_labels is not None
        args = TrainingArguments(
            output_dir=str(config.output_dir / "checkpoints"),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
            eval_strategy="epoch" if has_validation else "no",
            save_strategy="epoch" if has_validation else "no",
            load_best_model_at_end=has_validation,
            metric_for_best_model="eval_loss",
            report_to=[],
        )
        callbacks = (
            [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
            if has_validation
            else []
        )
        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=make_dataset(texts, list(labels)),
            eval_dataset=make_dataset(validation_texts, list(validation_labels))
            if has_validation
            else None,
            callbacks=callbacks,
        )
        trainer.train()
        return self

    def predict_scores(self, texts):
        import torch

        if self._model is None:
            raise RuntimeError("Transformer has not been trained")
        encodings = self._tokenizer(
            list(texts),
            truncation=True,
            max_length=self.config.max_sequence_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**encodings).logits
        return torch.softmax(logits, dim=-1).numpy()

    def predict(self, texts):
        scores = self.predict_scores(texts)
        return np.array([self._label_list[i] for i in scores.argmax(axis=1)])

    @property
    def classes_(self):
        return np.array(self._label_list)

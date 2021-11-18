from typing import Dict

import torch
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from src.common.model_training.pl_models.base_gpt2 import GPT2LMHead
from src.common.model_training.single_token_metrics import AccuracyMRR


class FLCCGPT2(GPT2LMHead):
    def __init__(self, config: DictConfig, actual_vocab_size: int):
        super().__init__(config, actual_vocab_size)

    def _get_metrics(self) -> MetricCollection:
        metrics = dict()
        for holdout in ["train", "val", "test"]:
            metrics[holdout] = AccuracyMRR(
                ignore_index=self._config.model.labels_pad,
                top_k=5,
                shift=True,
            )
        return MetricCollection(metrics)

    def _update_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor, holdout: str
    ) -> Dict[str, torch.Tensor]:
        return self._metrics[holdout](logits, labels).items()

    def _compute_metrics(self, holdout: str) -> Dict[str, torch.Tensor]:
        return self._metrics[holdout].compute().items()

from typing import Optional, Dict

import torch
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from flccpsisrc.common.model_training.pl_models.base_gpt2 import GPT2LMHead
from flccpsisrc.common.model_training.single_token_metrics import AccuracyMRR


class PSIGPT2(GPT2LMHead):
    def __init__(self, config: DictConfig, actual_vocab_size: int):
        super().__init__(config, actual_vocab_size)

    def _get_metrics(self) -> MetricCollection:
        metrics = dict()
        for holdout in ["train", "val", "test"]:
            for node_type in ["overall"]:
                metrics[f"{holdout}/{node_type}"] = AccuracyMRR(
                    ignore_index=self._config.model.labels_pad,
                    top_k=5,
                    shift=True,
                )
        return MetricCollection(metrics)

    def _compute_metrics(self, holdout: str) -> Dict[str, torch.Tensor]:
        res = dict()
        for node_type in ["overall"]:
            res.update(
                {f"{holdout}/{node_type}_{k}": v for k, v in self._metrics[f"{holdout}/{node_type}"].compute().items()}
            )
        return res

    def _update_metrics(self, logits: torch.Tensor, labels: torch.Tensor, holdout: str) -> Dict[str, torch.Tensor]:
        res = dict()
        res.update(self._update_metrics_with_mask(logits, labels, holdout, "overall", mask=None))
        return res

    def _update_metrics_with_mask(
        self, logits: torch.Tensor, labels: torch.Tensor, holdout: str, node_type: str, mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if mask is not None:
            labels = labels.clone()
            labels[mask] = self._config.model.labels_pad
        return {
            f"{holdout}/{node_type}_{k}": v for k, v in self._metrics[f"{holdout}/{node_type}"](logits, labels).items()
        }

from typing import Optional, Dict

import torch
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from src.common.model_training.pl_datamodule import PSIDataModule
from src.common.model_training.pl_models.base_gpt2 import GPT2LMHead
from src.common.model_training.single_token_metrics import AccuracyMRR


class PSIGPT2(GPT2LMHead):
    def __init__(self, config: DictConfig, actual_vocab_size: int):
        super().__init__(config, actual_vocab_size)

    def _get_metrics(self) -> MetricCollection:
        metrics = dict()
        for holdout in ["train", "val", "test"]:
            for node_type in ["overall", "bpeleaf", "staticleaf", "nonleaf"]:
                metrics[f"{holdout}/{node_type}"] = AccuracyMRR(
                    ignore_index=self._config.model.labels_pad,
                    top_k=5,
                    shift=True,
                )
        return MetricCollection(metrics)

    def _aggregate_single_token_metrics(self, holdout: str) -> Dict[str, torch.Tensor]:
        res = dict()
        for node_type in ["overall", "bpeleaf", "staticleaf", "nonleaf"]:
            res.update(
                {f"{holdout}/{node_type}_{k}": v for k, v in self._metrics[f"{holdout}/{node_type}"].compute().items()}
            )
        return res

    def _update_metrics(self, logits: torch.Tensor, labels: torch.Tensor, holdout: str) -> Dict[str, torch.Tensor]:
        res = dict()
        datamodule: PSIDataModule = self.trainer.datamodule
        arbitrary_mask, static_leaf_mask, non_leaf_mask = datamodule.psi_facade.tokenizer.classify_ids(labels)

        res.update(self._update_metrics_with_mask(logits, labels, holdout, "overall", mask=None))
        res.update(self._update_metrics_with_mask(logits, labels, holdout, "nonleaf", mask=non_leaf_mask))
        res.update(self._update_metrics_with_mask(logits, labels, holdout, "staticleaf", mask=static_leaf_mask))
        res.update(self._update_metrics_with_mask(logits, labels, holdout, "bpeleaf", mask=arbitrary_mask))

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

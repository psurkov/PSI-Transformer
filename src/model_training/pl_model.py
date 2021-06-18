from typing import Optional, Tuple, List, Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torchmetrics import MetricCollection
from transformers import GPT2Config, GPT2LMHeadModel, AdamW

from src.model_training.pl_datamodule import PSIDataModule
from src.model_training.single_token_metrics import AccuracyMRR
from src.utils import get_linear_schedule_with_warmup


class PSIBasedModel(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self._config = config

        if self._config.model.type == "gpt-2":
            model_config = GPT2Config(
                vocab_size=self._config.tokenizer.vocab_size,
                n_positions=self._config.model.context_length,
                n_ctx=self._config.model.context_length,
                n_embd=self._config.model.hidden_size,
                n_layer=self._config.model.n_layers,
                n_head=self._config.model.hidden_size // 64,
            )
            self._model = GPT2LMHeadModel(config=model_config)
        else:
            raise ValueError(f"Unsupported model type: {self._config.model.type}")

        metrics = dict()
        for holdout in ["train", "val", "test"]:
            for node_type in ["overall", "bpeleaf", "staticleaf", "nonleaf"]:
                metrics[f"{holdout}/{node_type}"] = AccuracyMRR(
                    ignore_index=self._config.model.labels_pad,
                    top_k=5,
                    shift=True,
                )
        self._metrics = MetricCollection(metrics)

    def forward(
        self,
        inputs: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        return_past: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ):
        # Maybe can be generalized to PretrainedModel
        if isinstance(self._model, GPT2LMHeadModel):
            assert len(inputs.size()) == 2
            if labels is not None:
                assert inputs.size() == labels.size()
            return self._model(
                input_ids=inputs,
                past_key_values=past_key_values,
                use_cache=return_past,
                labels=labels,
                return_dict=False,
            )
        else:
            raise ValueError(f"Unsupported model type: {type(self._model)}")

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        inputs, labels = batch
        loss, logits = self.forward(inputs, labels)
        return {"loss": loss, "logits": logits, "labels": labels}

    def training_step_end(self, batch_parts_outputs: Dict[str, torch.Tensor]):
        losses, logits, labels = (
            batch_parts_outputs["loss"],
            batch_parts_outputs["logits"],
            batch_parts_outputs["labels"],
        )
        self.log_dict(
            self._calc_single_token_metrics(logits.detach(), labels, "train"), on_step=True, prog_bar=True, logger=True
        )
        loss = losses.mean()

        self.log("train_loss", loss.detach(), on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        inputs, labels = batch
        [logits] = self.forward(inputs)
        return {"logits": logits, "labels": labels}

    def validation_step_end(self, batch_parts_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits, labels = batch_parts_outputs["logits"], batch_parts_outputs["labels"]
        return self._calc_single_token_metrics(logits, labels, "val")

    def validation_epoch_end(self, outs: List[Dict[str, torch.Tensor]]):
        self.log_dict(
            self._aggregate_single_token_metrics("val"),
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        inputs, labels = batch
        [logits] = self.forward(inputs)
        return {"logits": logits, "labels": labels}

    def test_step_end(self, batch_parts_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits, labels = batch_parts_outputs["logits"], batch_parts_outputs["labels"]
        return self._calc_single_token_metrics(logits, labels, "test")

    def test_epoch_end(self, outs: List[Dict[str, torch.Tensor]]):
        self.log_dict(
            self._aggregate_single_token_metrics("test"),
            on_step=False,
            on_epoch=True,
        )

    def _update_metrics_with_mask(
        self, logits: torch.Tensor, labels: torch.Tensor, holdout: str, node_type: str, mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if mask is not None:
            labels = labels.clone()
            labels[mask] = self._config.model.labels_pad
        return self._metrics[f"{holdout}/{node_type}"](logits, labels)

    def _calc_single_token_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor, holdout: str
    ) -> Dict[str, torch.Tensor]:
        res = dict()
        datamodule: PSIDataModule = self.trainer.datamodule
        arbitrary_mask, static_leaf_mask, non_leaf_mask = datamodule.psi_facade.tokenizer.classify_ids(labels)

        res.update(self._update_metrics_with_mask(logits, labels, holdout, "overall", mask=None))
        res.update(self._update_metrics_with_mask(logits, labels, holdout, "nonleaf", mask=non_leaf_mask))
        res.update(self._update_metrics_with_mask(logits, labels, holdout, "staticleaf", mask=static_leaf_mask))
        res.update(self._update_metrics_with_mask(logits, labels, holdout, "bpeleaf", mask=arbitrary_mask))

        return res

    def _aggregate_single_token_metrics(self, holdout: str) -> Dict[str, torch.Tensor]:
        res = dict()
        for node_type in ["overall", "bpeleaf", "staticleaf", "nonleaf"]:
            res.update(self._metrics[f"{holdout}/{node_type}"].compute())
        return res

    def configure_optimizers(self):
        total_batch_size = (
            self._config.training.batch_size
            * self._config.training.grad_accumulation_steps
            * self._config.training.world_size
        )
        lr = self._config.training.base_lr * total_batch_size

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self._config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            eps=self._config.training.adam_eps,
        )

        total_tokens_per_step = total_batch_size * self._config.model.context_length
        warmup_steps = self._config.training.warmup_tokens // total_tokens_per_step

        num_batches_epoch = len(self.trainer.datamodule.train_dataloader())
        num_steps_epoch = num_batches_epoch // self._config.training.grad_accumulation_steps
        total_steps = num_steps_epoch * self._config.training.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

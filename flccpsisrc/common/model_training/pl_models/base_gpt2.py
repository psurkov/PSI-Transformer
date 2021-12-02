import abc
from typing import Optional, Tuple, List, Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torchmetrics import MetricCollection
from transformers import GPT2Config, GPT2LMHeadModel, AdamW

from flccpsisrc.common.utils import get_linear_schedule_with_warmup


class GPT2LMHead(pl.LightningModule, abc.ABC):
    def __init__(self, config: DictConfig, actual_vocab_size: int) -> None:
        super().__init__()
        self._config = config

        if self._config.model.type == "gpt-2":
            model_config = GPT2Config(
                vocab_size=actual_vocab_size,
                n_positions=self._config.model.context_length,
                n_ctx=self._config.model.context_length,
                n_embd=self._config.model.hidden_size,
                n_layer=self._config.model.n_layers,
                n_head=self._config.model.hidden_size // 64,
            )
            self._model = GPT2LMHeadModel(config=model_config)
        else:
            raise ValueError(f"Unsupported model type: {self._config.model.type}")

        self._metrics = self._get_metrics()

    @property
    def model(self) -> GPT2LMHeadModel:
        return self._model

    @abc.abstractmethod
    def _get_metrics(self) -> MetricCollection:
        pass

    @abc.abstractmethod
    def _update_metrics(self, logits: torch.Tensor, labels: torch.Tensor, holdout: str) -> Dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def _compute_metrics(self, holdout: str) -> Dict[str, torch.Tensor]:
        pass

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
            self._update_metrics(logits.detach(), labels, "train"), on_step=True, prog_bar=False, logger=True
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
        return self._update_metrics(logits, labels, "val")

    def validation_epoch_end(self, outs: List[Dict[str, torch.Tensor]]):
        self.log_dict(
            self._compute_metrics("val"),
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        inputs, labels = batch
        [logits] = self.forward(inputs)
        return {"logits": logits, "labels": labels}

    def test_step_end(self, batch_parts_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits, labels = batch_parts_outputs["logits"], batch_parts_outputs["labels"]
        return self._update_metrics(logits, labels, "test")

    def test_epoch_end(self, outs: List[Dict[str, torch.Tensor]]):
        self.log_dict(
            self._compute_metrics("test"),
            on_step=False,
            on_epoch=True,
        )

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

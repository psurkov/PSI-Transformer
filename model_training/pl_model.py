from typing import Optional, Tuple, List, Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers_lightning.schedulers import LinearSchedulerWithWarmup

from model_training.single_token_metrics import accuracy_mrr


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

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inputs, labels = batch
        loss, _ = self.forward(inputs, labels)
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, logger=True)
        return loss

    def _calc_single_token_metrics(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, labels = batch
        [logits] = self.forward(inputs)
        return accuracy_mrr(logits, labels, ignore_index=self._config.model.labels_pad)

    @staticmethod
    def _aggregate_single_token_metrics(
        outs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], prefix: str
    ) -> Dict[str, float]:
        acc1_sums, acc5_sums, mrr5_sums, total_sums = zip(*outs)
        total = sum(total_sums)
        acc1 = sum(acc1_sums) / total
        acc5 = sum(acc5_sums) / total
        mrr5 = sum(mrr5_sums) / total
        return {f"{prefix}_acc_top1": acc1, f"{prefix}_acc_top5": acc5, f"{prefix}_MRR_top5": mrr5}

    def validation_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._calc_single_token_metrics(batch)

    def validation_epoch_end(self, outs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.log_dict(
            PSIBasedModel._aggregate_single_token_metrics(outs, "val"),
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._calc_single_token_metrics(batch)

    def test_epoch_end(self, outs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.log_dict(
            PSIBasedModel._aggregate_single_token_metrics(outs, "test"),
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

        scheduler = LinearSchedulerWithWarmup(optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

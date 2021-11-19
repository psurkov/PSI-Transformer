import os
from datetime import datetime

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.common.model_training.pl_datamodule import PSIDataModule
from src.common.model_training.pl_models.psi_gpt2 import PSIGPT2
from src.common.utils import run_with_config
from src.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade


def train(config: DictConfig) -> None:
    config.training.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config.training.world_size = config.training.n_gpus if config.training.n_gpus else 1

    lr_logger = LearningRateMonitor()

    model_checkpoints_dir = "model_checkpoints"
    model_checkppint_dir_path = os.path.join(
        config.save_path, model_checkpoints_dir, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    os.makedirs(model_checkppint_dir_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{step}",
        dirpath=model_checkppint_dir_path,
        save_top_k=config.training.save_top_k,
        save_last=True,
        verbose=True,
        monitor="val/overall_MRR@5",
        mode="max",
    )
    if config.training.resume_from_checkpoint is not None:
        checkpoint_path = (
            config.training.resume_from_checkpoint
            if os.path.exists(config.training.resume_from_checkpoint)
            else os.path.join(config.save_path, model_checkpoints_dir, config.training.resume_from_checkpoint)
        )
    else:
        checkpoint_path = None

    cloud_logger = WandbLogger(project="PSI-Transformer", log_model=True, save_dir=config.save_path)

    facade = PSIDatapointFacade(config)
    assert facade.is_trained
    model = PSIGPT2(config, facade.tokenizer.vocab_size)
    datamodule = PSIDataModule(config)

    pl.seed_everything(config.training.seed)

    trainer = pl.Trainer(
        # fast_dev_run=True,
        # limit_val_batches=0.05,
        # limit_train_batches=0.05,
        max_epochs=config.training.epochs,
        gpus=(config.training.n_gpus if config.training.n_gpus else None),
        auto_select_gpus=(True if config.training.n_gpus > 0 else False),
        distributed_backend=("ddp" if config.training.n_gpus > 1 else None),
        precision=(16 if config.training.fp16 and config.training.n_gpus > 0 else 32),
        amp_level=config.training.fp16_opt_level,
        accumulate_grad_batches=config.training.grad_accumulation_steps,
        gradient_clip_val=config.training.max_grad_norm,
        callbacks=[lr_logger, checkpoint_callback],
        checkpoint_callback=True,
        logger=cloud_logger,
        resume_from_checkpoint=checkpoint_path,
        val_check_interval=config.training.val_check_interval,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    run_with_config(train)

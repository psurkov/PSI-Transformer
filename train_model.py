import os
from datetime import datetime

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from model_training.pl_datamodule import PSIDataModule
from model_training.pl_model import PSIBasedModel


@hydra.main(config_name="config.yaml")
def train(config: DictConfig) -> None:
    config.training.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config.training.world_size = config.training.n_gpus if config.training.n_gpus else 1

    lr_logger = LearningRateMonitor()
    model_checkppint_dir_path = os.path.join(config.save_path, f"model_checkpoints_{datetime.now()}")
    os.mkdir(model_checkppint_dir_path)
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{step}-{val_overall_MRR@5:.2f}",
        dirpath=model_checkppint_dir_path,
        save_top_k=config.training.save_top_k,
        save_last=True,
        verbose=True,
        monitor="val_overall_MRR@5",
        mode="min",
    )
    stopping_callback = EarlyStopping(monitor="val_overall_MRR@5", min_delta=1e-2, patience=5, verbose=True, mode="max")

    cloud_logger = WandbLogger(project="PSI-Transformer", log_model=True)

    model = PSIBasedModel(config)
    datamodule = PSIDataModule(config)

    pl.seed_everything(config.training.seed)

    trainer = pl.Trainer(
        # fast_dev_run=True,
        # overfit_pct=0.1,
        max_epochs=config.training.epochs,
        gpus=(config.training.n_gpus if config.training.n_gpus else None),
        auto_select_gpus=(True if config.training.n_gpus > 0 else False),
        distributed_backend=("ddp" if config.training.n_gpus > 1 else None),
        precision=(16 if config.training.fp16 and config.training.n_gpus > 0 else 32),
        amp_level=config.training.fp16_opt_level,
        accumulate_grad_batches=config.training.grad_accumulation_steps,
        gradient_clip_val=config.training.max_grad_norm,
        callbacks=[lr_logger, checkpoint_callback, stopping_callback],
        checkpoint_callback=True,
        logger=cloud_logger,
        # resume_from_checkpoint=config.model.name_or_path,
        val_check_interval=config.training.val_check_interval,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()

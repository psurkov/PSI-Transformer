from typing import Optional, List, Tuple, Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.common.model_training.datasets.psi_dataset import PSIDataset
from src.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade


class PSIDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self._config = config
        self._psi_facade: PSIDatapointFacade = PSIDatapointFacade(config)
        assert self._psi_facade.is_trained

    @property
    def psi_facade(self) -> PSIDatapointFacade:
        return self._psi_facade

    def setup(self, stage: Optional[str] = None):
        pass

    def prepare_data(self):
        psi_facade = PSIDatapointFacade(self._config)
        if not psi_facade.is_trained:
            psi_facade.train()

    def _collate_fn(self, tensors: List[Tuple[torch.Tensor, torch.Tensor]]):
        inputs, labels = zip(*tensors)
        return (
            torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0),
            torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self._config.model.labels_pad),
        )

    def train_dataloader(self) -> DataLoader:  # type: ignore
        dataset = PSIDataset(self._config, "train", self._config.training.local_rank, self._config.training.world_size)
        return DataLoader(
            dataset,
            self._config.training.batch_size,
            num_workers=self._config.training.num_dataset_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore
        dataset = PSIDataset(self._config, "val", self._config.training.local_rank, self._config.training.world_size)
        return DataLoader(
            dataset,
            self._config.training.batch_size,
            num_workers=self._config.training.num_dataset_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore
        dataset = PSIDataset(self._config, "test", self._config.training.local_rank, self._config.training.world_size)
        return DataLoader(
            dataset,
            self._config.training.batch_size,
            num_workers=self._config.training.num_dataset_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            drop_last=True,
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:  # type: ignore
        pass

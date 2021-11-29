import json
from typing import Iterator, Tuple

import torch
from omegaconf import DictConfig

from flccpsisrc.common.model_training.datasets.base_dataset import ParallelIterableDataset
from flccpsisrc.flcc.data_processing.tokenizer import FLCCBPE


class FLCCDataset(ParallelIterableDataset):
    def __init__(
        self,
        config: DictConfig,
        holdout: str,
        rank: int,
        world_size: int,
    ):
        self._holdout = holdout
        if holdout == "train":
            self._data_path = config.source_data.train
            need_shuffle = True
        elif holdout == "val":
            self._data_path = config.source_data.val
            need_shuffle = False
        elif holdout == "test":
            self._data_path = config.source_data.test
            need_shuffle = False
        else:
            raise ValueError(f"Invalid holdout value {holdout}")

        super().__init__(
            pad_id=0,
            labels_pad_id=config.model.labels_pad,
            example_length=config.model.context_length,
            pad_overlap=config.dataset.pad_overlapped,
            overlap_slicing=config.dataset.overlap_slicing,
            bucket_size=config.dataset.shuffle_bucket if need_shuffle else None,
            rank=rank,
            world_size=world_size,
        )

        self._example_len = config.model.context_length
        self._tokenizer: FLCCBPE = FLCCBPE.from_pretrained(config.save_path)

    def __len__(self) -> int:
        pass

    def _example_iterator(self, rank: int, world_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        with open(self._data_path) as f:
            for i, line in enumerate(f):
                if i % world_size == rank:
                    content = json.loads(line)
                    ids = self._tokenizer.encode(content)
                    yield from self._slice_examples(ids)

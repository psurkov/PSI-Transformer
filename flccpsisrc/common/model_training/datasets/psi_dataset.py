import math
from typing import Iterator, Tuple

import torch
from omegaconf import DictConfig

from flccpsisrc.common.model_training.datasets.base_dataset import ParallelIterableDataset
from flccpsisrc.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade


class PSIDataset(ParallelIterableDataset):
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

        self._psi_facade = PSIDatapointFacade(config, diff_warning=False)
        self._example_len = config.model.context_length

    def __len__(self) -> int:
        return (
            sum(
                int(math.ceil(size / ((1 - self._overlap_slicing) * self._example_len)))
                for size in self._psi_facade.get_tokenized_sizes(self._holdout)
            )
            // self._world_size
        )

    def _example_iterator(self, rank: int, world_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        with open(self._data_path, "r") as f:
            for i, line in enumerate(f):
                if i % world_size == rank:
                    tree = self._psi_facade.json_dict_to_split_tree(line, to_filter=True)
                    if tree is not None:
                        yield from self._slice_examples(
                            self._psi_facade.encode_split_tree_to_ids(tree)
                        )

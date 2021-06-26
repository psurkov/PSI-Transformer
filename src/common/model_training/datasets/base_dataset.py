from abc import ABC, abstractmethod
from random import Random
from typing import Iterator, Tuple, Optional, List

import torch
from torch.utils.data import IterableDataset, get_worker_info


class ParallelIterableDataset(ABC, IterableDataset):
    def __init__(
        self,
        pad_id: int,
        labels_pad_id: int,
        example_length: int,
        overlap_slicing: float,
        pad_overlap: bool,
        bucket_size: Optional[int],
        rank: int,
        world_size: int,
    ):
        assert 0 <= rank < world_size
        self._rank = rank
        self._world_size = world_size

        self._pad_id = pad_id
        self._labels_pad_id = labels_pad_id
        self._example_length = example_length
        assert 0 <= overlap_slicing <= 1
        self._overlap_slicing = overlap_slicing
        self._pad_overlap = pad_overlap
        self._bucket_size = bucket_size
        self._rng = Random(42)

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of examples per rank"""
        pass

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        # Assuming that num_workers are the same in the whole world
        world_size = self._world_size * num_workers
        rank = self._rank * num_workers + worker_id

        desired_length = len(self) // num_workers
        examples_counter = 0
        bucket = []
        for example in self._example_iterator(rank, world_size):
            examples_counter += 1

            if self._bucket_size:
                bucket.append(example)
                if len(bucket) >= self._bucket_size:
                    self._rng.shuffle(bucket)
                    yield from bucket
                    bucket = []
            else:
                yield example

            if examples_counter == desired_length:
                break

        if bucket:
            self._rng.shuffle(bucket)
            yield from bucket

        while examples_counter < desired_length:
            examples_counter += 1
            yield self._empty_batch

    @abstractmethod
    def _example_iterator(self, rank: int, world_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        pass

    @property
    def _empty_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([self._pad_id], dtype=torch.long), torch.tensor([self._labels_pad_id], dtype=torch.long)

    def _slice_examples(self, ids: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        examples = []
        for start in range(0, len(ids), int((1 - self._overlap_slicing) * self._example_length)):
            inp = torch.tensor(ids[start : start + self._example_length], dtype=torch.long)
            labels = inp.clone()
            if self._pad_overlap and start:
                labels[: int(self._overlap_slicing * self._example_length)] = self._example_length
            examples.append((inp, labels))
        return examples

    def __getitem__(self, item):
        raise NotImplementedError("Iterable dataset does not support indexing")

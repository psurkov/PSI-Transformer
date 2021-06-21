import logging
from typing import List, Tuple

import torch

from src.model_evaluation.beam_search.model_wrapper import ModelWrapper
from src.model_evaluation.beam_search.search import HypothesesGroups, BeamSearch
from src.psi_datapoint.stateful.tokenizer import TreeBuilder


class SequenceGenerator:
    def __init__(
        self,
        *,
        model_wrapper: ModelWrapper,
        eos_ids: List[int],
        beam_size: int,
        len_norm_base: float,
        len_norm_pow: float,
    ):
        self._model_wrapper = model_wrapper
        self._eos_ids = eos_ids
        self._beam_size = beam_size
        self._len_norm_base = len_norm_base
        self._len_norm_pow = len_norm_pow

    def search_sequence(
        self, num_iterations: int, tree_builder: TreeBuilder, **kwargs
    ) -> Tuple[HypothesesGroups, HypothesesGroups]:
        search = BeamSearch(
            self._eos_ids,
            self._model_wrapper.vocab_size,
            self._beam_size,
            tree_builder,
            self._len_norm_base,
            self._len_norm_pow,
        )

        log_probs, prefix = self._model_wrapper.init_state(**kwargs)

        assert tuple(log_probs.size()) == (
            1,
            self._model_wrapper.vocab_size,
        ), f"log_probs must have shape (1, vocab_size), but {log_probs.size()} was given"

        # expand batch
        log_probs = log_probs.repeat_interleave(search.batch_size, dim=0)
        self._model_wrapper.sort_state(torch.zeros(search.batch_size, dtype=torch.long))

        for _ in range(num_iterations - 1):
            selected_inds = search.step(log_probs)
            self._model_wrapper.sort_state(selected_inds)

            data = search.last_predictions
            log_probs = self._model_wrapper.get_log_probs(data)
        search.step(log_probs)

        self._model_wrapper.reset_state()

        return search.terminated_hypotheses, search.current_hypotheses

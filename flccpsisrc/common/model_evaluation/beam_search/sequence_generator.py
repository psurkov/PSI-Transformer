from typing import List, Tuple

import torch

from flccpsisrc.common.model_evaluation.beam_search.model_wrapper import ModelWrapper
from flccpsisrc.common.model_evaluation.beam_search.search import Hypothesis, BeamSearch


class SequenceGenerator:
    def __init__(self, model_wrapper: ModelWrapper, num_iterations: int, beam_size: int):
        self._model_wrapper = model_wrapper
        self._num_iterations = num_iterations
        self._beam_size = beam_size

    def search_sequence(self, **kwargs) -> Tuple[List[Hypothesis], List[Hypothesis]]:
        self._model_wrapper.reset_state()
        log_probs, tree_builder = self._model_wrapper.init_state(**kwargs)

        assert tuple(log_probs.size()) == (
            1,
            self._model_wrapper.vocab_size,
        ), f"log_probs must have shape (1, {self._model_wrapper.vocab_size}), but {log_probs.size()} was given"

        search = BeamSearch(
            self._model_wrapper.vocab_size,
            self._beam_size,
            tree_builder,
        )

        # expand batch
        log_probs = log_probs.repeat_interleave(search.batch_size, dim=0)
        self._model_wrapper.sort_state(torch.zeros(search.batch_size, dtype=torch.long))

        for i in range(self._num_iterations - 1):
            selected_inds = search.step(log_probs)
            if selected_inds is None:
                print(f"Beam search early stopped at {i}th log_probs")
                return search.terminated_hypotheses, search.current_hypotheses
            self._model_wrapper.sort_state(selected_inds)

            data = search.last_predictions
            log_probs = self._model_wrapper.get_log_probs(data)
        search.step(log_probs)

        self._model_wrapper.reset_state()

        return search.terminated_hypotheses, search.current_hypotheses

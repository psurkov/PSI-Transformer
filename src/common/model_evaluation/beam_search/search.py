from dataclasses import dataclass
from typing import List, Callable

import torch

from src.psi_datapoint.stateful.tokenizer import TreeBuilder
from src.psi_datapoint.tree_structures.LineBreaker import LineBreaker


@dataclass
class Hypothesis:
    ids: torch.LongTensor
    score: float


HypothesesGroups = List[List[Hypothesis]]


class Search:
    """
    Class for search algorithms
    Basically user needs to feed log_probs and perform a step several times
    Results can be found in hypotheses"""

    def __init__(
        self,
        vocab_size: int,
        search_size: int,
        len_norm_base: float = 0.0,
        len_norm_pow: float = 0.0,
    ):
        self._search_size = search_size
        self._vocab_size = vocab_size
        self._len_norm_base = len_norm_base
        self._len_norm_pow = len_norm_pow

        self._is_initialized = False

    def step(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Take a single search step.

        Args:
            log_probs: (batch_size, vocab_size)
                the model's log-probabilities over the vocabulary at the current step

        Return:
            chosen_hypotheses: (batch_size,)
                indices of the chosen hypotheses in range [0, batch_size)
                it should be used for sorting your model's hidden state
        """
        if not self._is_initialized:
            self._init_state(log_probs)
            self._is_initialized = True
        self._step_check(log_probs)
        log_probs = self._preprocess_log_probs(log_probs)
        chosen_hypotheses = self._step(log_probs)
        self._postprocess_state()

        return chosen_hypotheses

    @property
    def terminated_hypotheses(self) -> HypothesesGroups:
        """List of lists of tuples of terminated hypotheses and theirs scores"""
        raise NotImplementedError

    @property
    def current_hypotheses(self) -> HypothesesGroups:
        """List of lists of tuples of current hypotheses and theirs scores"""
        raise NotImplementedError

    @property
    def last_predictions(self) -> torch.Tensor:
        """Tensor of last tokens of the current hypotheses with shape (batch_size,).
        Supposed usage: making a batch for a model"""
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        """Current batch size"""
        raise NotImplementedError

    def _init_state(self, log_probs: torch.Tensor) -> None:
        raise NotImplementedError

    def _step_check(self, log_probs: torch.Tensor) -> None:
        assert log_probs.size() == (
            self.batch_size,
            self._vocab_size,
        ), f"log_probs must have shape {(self.batch_size, self._vocab_size)}, but {log_probs.size()} was given"

        assert all(
            eos < self._vocab_size for eos in self._eos_ids
        ), f"EOS ids must be less than vocab_size, but EOS ids: {self._eos_ids} and vocab_size: {self._vocab_size}"

    def _preprocess_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _step(self, log_probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _postprocess_state(self) -> None:
        raise NotImplementedError


class BeamSearch(Search):
    """Beam search algorithm with normalized by length scores"""

    def __init__(
        self,
        vocab_size: int,
        beam_size: int,
        tree_builder: TreeBuilder,
        line_breaker: LineBreaker,
        len_norm_base: float = 0.0,
        len_norm_pow: float = 0.0,
    ):
        super().__init__(vocab_size, beam_size, len_norm_base, len_norm_pow)

        self._length = 1
        self._terminated_hypotheses = []

        self._scores = None
        self._hypotheses = None
        self._chosen_hypotheses = None
        self._tree_builders = [tree_builder]
        self._line_break = [line_breaker]

    @property
    def terminated_hypotheses(self) -> HypothesesGroups:
        """List of lists of tuples of terminated hypotheses and theirs scores"""
        ans = sorted(self._terminated_hypotheses, key=lambda hyp: hyp.score, reverse=True)
        return [ans]

    @property
    def current_hypotheses(self) -> HypothesesGroups:
        """List of lists of tuples of terminated hypotheses and theirs scores"""
        ans = sorted(
            Hypothesis(hyp.tolist(), score.item())
            for hyp, score in zip(self._hypotheses, torch.exp(self._get_normalized_scores()))
        )
        return [ans]

    @property
    def last_predictions(self) -> torch.Tensor:
        """Tensor of last tokens of the current hypotheses with shape (batch_size,).
        Supposed usage: making a batch for a model"""
        assert (
            self._hypotheses is not None and self._hypotheses.size(1) > 0
        ), f"Can't get last predictions if no steps have been performed"
        return self._hypotheses[:, -1]

    @property
    def batch_size(self) -> int:
        """Current batch size"""
        if self._scores is None:
            return 1
        return self._scores.size(0)

    def _init_state(self, log_probs: torch.Tensor):
        self._eos_tensor = self._eos_tensor.to(log_probs.device)
        # Initial state
        assert self._scores is None and self._hypotheses is None
        self._scores = torch.zeros(1, dtype=log_probs.dtype, device=log_probs.device)
        self._hypotheses = torch.empty(1, 0, dtype=torch.long, device=log_probs.device)

    def _preprocess_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        row_mask = torch.empty(log_probs.size(1), dtype=torch.bool, device=log_probs.device)
        for row_id, tree_builder in enumerate(self._tree_builders):
            possible_ids = tree_builder.get_next_possible_ids()
            row_mask[:] = 1
            row_mask[possible_ids] = 0
            log_probs[row_id, row_mask] = float("-inf")
        return torch.nn.functional.log_softmax(log_probs, dim=-1)

    def _step(self, log_probs: torch.Tensor) -> torch.Tensor:
        log_probs.add_(self._scores.unsqueeze(1))
        log_probs = torch.flatten(log_probs)
        sample_scores, samples = torch.topk(
            log_probs,
            # Take more to ensure that we will keep search_size not terminated
            min((1 + len(self._eos_ids)) * self._search_size, log_probs.size(0)),
            sorted=False,
        )
        # inf_mask = sample_scores != float("-inf")
        # samples = samples[inf_mask]
        # sample_scores = sample_scores[inf_mask]

        sort_mask = torch.floor_divide(samples, self._vocab_size)
        samples.fmod_(self._vocab_size)

        self._update_state(samples, sample_scores, sort_mask)
        self._length += 1

        return self._chosen_hypotheses

    def _postprocess_state(self) -> None:
        # Sort tree builders according to chosen hypotheses
        self._tree_builders = [self._tree_builders[i].copy() for i in self._chosen_hypotheses.tolist()]

        for new_id, tree_builder in zip(self.last_predictions.tolist(), self._tree_builders):
            tree_builder.add_id(new_id)

    def _update_state(self, samples: torch.Tensor, sample_scores: torch.Tensor, sort_mask: torch.Tensor):
        self._chosen_hypotheses = torch.arange(self.batch_size)

        self._apply_slice_to_state(sort_mask)

        self._scores += sample_scores
        self._hypotheses = torch.cat((self._hypotheses, samples.unsqueeze(1)), dim=1)
        self._stash_terminated(samples)

    def _stash_terminated(self, samples: torch.Tensor):
        terminated = self._is_sample_terminated(samples)

        scores = torch.exp(self._get_normalized_scores())
        for terminated_hypothesis, score in zip(
            self._hypotheses[terminated][: self._search_size],
            scores[terminated][: self._search_size],
        ):
            assert len(terminated_hypothesis) == int(self._length)
            hypothesis = Hypothesis(terminated_hypothesis.tolist(), score.item())
            self._terminated_hypotheses.append(hypothesis)

        # And throw out all terminated
        self._apply_slice_to_state(~terminated)
        # Select first beam_size
        _, top_bs_mask = torch.topk(self._scores, min(self._search_size, self._scores.size(0)), sorted=False)
        self._apply_slice_to_state(top_bs_mask)

    def _is_sample_terminated(self, samples: torch.Tensor):
        result = samples == self._eos_tensor.expand(self._eos_tensor.size(0), samples.size(0))
        return result.sum(dim=0, dtype=torch.bool)

    def _apply_slice_to_state(self, tensor_slice):
        self._scores = self._scores[tensor_slice]
        self._hypotheses = self._hypotheses[tensor_slice]
        self._chosen_hypotheses = self._chosen_hypotheses[tensor_slice]

    def _get_normalized_scores(self) -> torch.Tensor:
        norm_factor = ((self._len_norm_base + self._length) / (self._len_norm_base + 1)) ** self._len_norm_pow
        return self._scores / norm_factor

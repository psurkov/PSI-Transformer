import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from flccpsisrc.psi.psi_datapoint.tree_structures.split_tree_builder import SplitTreeBuilder


@dataclass
class Hypothesis:
    ids: List[int]
    score: float
    is_terminated: bool
    _split_tree_builder: SplitTreeBuilder

    def get_normalized_score(self, len_norm_base: float = 5.0, len_norm_pow: float = 0.7) -> float:
        hyp_length = len(self.ids)
        norm_factor = ((len_norm_base + hyp_length) / (len_norm_base + 1)) ** len_norm_pow
        return math.exp(self.score / norm_factor)

    @property
    def text(self) -> str:
        return self._split_tree_builder.decode_generated_ids(self.ids)


class BeamSearch:
    """Beam search algorithm with normalized by length scores"""

    def __init__(
            self,
            vocab_size: int,
            beam_size: int,
            start_split_tree_builder: SplitTreeBuilder,
    ):
        self._vocab_size = vocab_size
        self._beam_size = beam_size

        self._length = 1
        self._terminated_hypotheses = []

        self._scores = None
        self._hypotheses = None
        self._sort_mask = None
        self._row_mask = None
        self._split_tree_builder = start_split_tree_builder

        self._is_initialized = False
        self._device = None

    def step(self, log_probs: torch.Tensor) -> Optional[torch.Tensor]:
        """Take a single search step.

        Args:
            log_probs: (batch_size, vocab_size)
                the model's log-probabilities over the vocabulary at the current step

        Return:
            sort_mask: (batch_size,)
                indices of the chosen hypotheses in range [0, batch_size)
                it should be used for sorting your model's hidden state
        """
        if not self._is_initialized:
            self._init_state(log_probs)
            self._is_initialized = True
        self._step_check(log_probs)
        log_probs = self._preprocess_log_probs(log_probs)
        sort_mask = self._step(log_probs)

        return sort_mask

    @property
    def terminated_hypotheses(self) -> List[Hypothesis]:
        """List of lists of tuples of terminated hypotheses and theirs scores"""
        return self._terminated_hypotheses

    @property
    def current_hypotheses(self) -> List[Hypothesis]:
        """List of lists of tuples of terminated hypotheses and theirs scores"""
        return [
            Hypothesis(
                hyp.tolist(),
                score.item(),
                False,
                self._split_tree_builder,
            )
            for hyp, score in zip(self._hypotheses, self._scores)
        ]

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
        assert self._scores is None and self._hypotheses is None
        self._device = log_probs.device
        self._scores = torch.zeros(1, dtype=log_probs.dtype, device=log_probs.device)
        self._hypotheses = torch.empty(1, 0, dtype=torch.long, device=log_probs.device)
        self._row_mask = torch.empty(log_probs.size(1), dtype=torch.bool, device=log_probs.device)

    def _step_check(self, log_probs: torch.Tensor) -> None:
        assert log_probs.size() == (
            self.batch_size,
            self._vocab_size,
        ), f"log_probs must have shape {(self.batch_size, self._vocab_size)}, but {log_probs.size()} was given"

    def _preprocess_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        for row_id, version_id in enumerate(self._split_tree_builder.active_versions_ids):
            possible_ids = self._split_tree_builder.get_next_possible_ids(version_id)
            self._row_mask[:] = 1
            self._row_mask[possible_ids] = 0
            log_probs[row_id, self._row_mask] = float("-inf")
        return torch.nn.functional.log_softmax(log_probs, dim=-1)

    def _step(self, log_probs: torch.Tensor) -> Optional[torch.Tensor]:
        log_probs.add_(self._scores.unsqueeze(1))
        log_probs = torch.flatten(log_probs)

        samples = []
        active_versions = []
        sort_mask = []
        sample_scores = []
        sorted_scores, sorted_inds = torch.sort(log_probs, descending=True)
        for ind, score in zip(sorted_inds, sorted_scores):
            if torch.isnan(score) or torch.isneginf(score):
                break
            ind = ind.item()
            hyp_ind, token_ind = divmod(ind, self._vocab_size)
            new_version = self._split_tree_builder.create_copy(self._split_tree_builder.active_versions_ids[hyp_ind])
            status = self._split_tree_builder.add_token(new_version, token_ind)
            if status == SplitTreeBuilder.ChangeStatus.TERMINATED:
                self._save_terminated(hyp_ind, token_ind, score.item())
            else:
                samples.append(token_ind)
                active_versions.append(new_version)
                sort_mask.append(hyp_ind)
                sample_scores.append(score)
            if len(samples) == self._beam_size:
                break
        if not samples:
            return None
        if len(samples) < self._beam_size:
            print(
                f"There was not enough hypotheses to process!\n"
                f"Samples drafted: {len(samples)}, beam_size: {self._beam_size}"
            )

        self._update_state(samples, sort_mask, sample_scores, active_versions)
        self._length += 1

        return self._sort_mask

    def _update_state(
            self,
            samples: List[int],
            sort_mask: List[int],
            new_scores: List[float],
            active_versions: List[int],
    ) -> None:
        self._samples = torch.tensor(samples, dtype=torch.long, device=self._device)
        self._sort_mask = torch.tensor(sort_mask, dtype=torch.long, device=self._device)
        self._scores = torch.tensor(new_scores, dtype=self._scores.dtype, device=self._device)
        self._split_tree_builder.filter_active_versions(active_versions)

        self._hypotheses = self._hypotheses[sort_mask]
        self._hypotheses = torch.cat((self._hypotheses, self._samples.unsqueeze(1)), dim=1)

    def _save_terminated(self, hyp_ind: int, sample_ind: int, score: float) -> None:
        hyp_inds = self._hypotheses[hyp_ind].tolist()
        hyp_inds.append(sample_ind)
        self._terminated_hypotheses.append(Hypothesis(hyp_inds, score, True, self._split_tree_builder))

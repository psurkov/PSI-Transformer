from typing import List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import GPT2LMHeadModel

from flccpsisrc.common.model_evaluation.beam_search.model_wrapper import ModelWrapper
from flccpsisrc.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from flccpsisrc.psi.psi_datapoint.tree_structures.split_tree import SplitTree
from flccpsisrc.psi.psi_datapoint.tree_structures.split_tree_builder import SplitTreeBuilder


class PSIGPT2Wrapper(ModelWrapper):
    def __init__(self, config: DictConfig, model: GPT2LMHeadModel):
        self._psi_facade = PSIDatapointFacade(config, diff_warning=False)
        assert self._psi_facade.is_trained
        self._model = model.eval()
        print(f"Number of parameters: {sum(p.numel() for p in self._model.parameters())}")

        self._mems = None

    def init_state(self, split_tree: SplitTree, num_iterations: int) -> Tuple[torch.Tensor, SplitTreeBuilder]:
        context_len = self._model.config.n_ctx - num_iterations

        context_ids = self._psi_facade.encode_split_tree_to_ids(split_tree)[-context_len:]

        context = torch.tensor(context_ids).unsqueeze(0)
        with torch.no_grad():
            scores, self._mems = self._model(context, use_cache=True, return_dict=False)
            log_probs = F.log_softmax(scores[:, -1, :], dim=1)

        return log_probs, self._psi_facade.get_split_tree_builder(split_tree)

    def sort_state(self, sort_mask: torch.Tensor) -> None:
        self._mems = tuple(tuple(k[sort_mask].contiguous() for k in mem) for mem in self._mems)

    def get_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        assert self._mems[0][0].size(2) < self._model.config.n_ctx
        with torch.no_grad():
            scores, self._mems = self._model(data.unsqueeze(1), self._mems, use_cache=True, return_dict=False)
            log_probs = F.log_softmax(scores.squeeze(1), dim=1)
        return log_probs

    def reset_state(self) -> None:
        self._mems = None

    @property
    def vocab_size(self) -> int:
        return self._psi_facade.vocab_size

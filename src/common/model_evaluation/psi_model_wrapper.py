from typing import List, Tuple

import torch
from transformers import GPT2LMHeadModel

from src.common.model_evaluation.beam_search.model_wrapper import ModelWrapper
from src.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from src.psi.psi_datapoint.tree_structures.line_breaker import LineBreaker
from src.psi.psi_datapoint.tree_structures.node import Node


class PSIGPT2Wrapper(ModelWrapper):
    def __init__(
        self,
        model: GPT2LMHeadModel,
        psi_facade: PSIDatapointFacade,
        device: torch.device,
    ):
        self._psi_facade = psi_facade

        self._model = model.to(device).eval()
        print(f"Number of parameters: {sum(p.numel() for p in self._model.parameters())}")

        self._device = device
        self._model_context_len = self._model.config.n_ctx

        self._mems = None
        self.reset_state()

    def init_state(
        self,
        nodes: List[Node],
        terminal_chars: str,
        iters: int,
        context_len: int,
        start_ids: List[int] = None,
    ) -> Tuple[torch.Tensor, str]:
        line_breaker = LineBreaker()

        # tree_builder_base = self._psi_facade.get_tree_builder()
        # context_len = self._model.config.n_ctx - num_iters
        # out_context_ids = input_ids[:-context_len]
        # for id_ in out_context_ids:
        #     tree_builder_base.add_id(id_)
        # context_ids = input_ids[len(tree_builder_base.ids):]
        #
        # logits_preprocessor = PSILogits(tree_builder_base)
        #
        # self._model.beam_search()
        #
        #
        #
        # context, suffix = self._tokenize_context(context, terminal_chars)
        # self._last_prefix_len = len(suffix)
        #
        # context = self._truncate_context(context, context_len, iters, start_ids)
        # assert context[0].item() in start_ids
        # context.unsqueeze_(0)
        #
        # with torch.no_grad():
        #     scores, self._mems = self._model(context)
        #     log_probs = F.log_softmax(scores[:, -1, :], dim=1)
        #
        # return log_probs, suffix

    def sort_state(self, sort_mask: torch.Tensor) -> None:
        pass
        # self._mems = [mem[:, sort_mask].contiguous() for mem in self._mems]

    def get_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        pass
        # assert self._mems[0].size(3) < self._model_context_len
        # with torch.no_grad():
        #     scores, self._mems = self._model(data.unsqueeze(1), self._mems)
        #     log_probs = F.log_softmax(scores.squeeze(1), dim=1)
        # return log_probs

    def reset_state(self) -> None:
        pass
        # self._mems = None

    @property
    def vocab_size(self) -> int:
        pass
        # return self.tokenizer.vocab_size

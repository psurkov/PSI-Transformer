from typing import Tuple, List

import torch
from transformers import GPT2LMHeadModel, LogitsProcessor

from src.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from src.psi_datapoint.stateful.tokenizer import TreeBuilder
from src.psi_datapoint.tree_structures.tree import Tree


class HuggingFaceBeamSearch:
    def __init__(self, model: GPT2LMHeadModel, psi_facade: PSIDatapointFacade):
        self._model = model
        self._psi_facade = psi_facade

    def generate(self, input_ids: List[int], beam_size: int, num_iters: int) -> List[Tuple[Tree, float]]:
        tree_builder_base = self._psi_facade.get_tree_builder()
        context_len = self._model.config.n_ctx - num_iters
        out_context_ids = input_ids[:-context_len]
        for id_ in out_context_ids:
            tree_builder_base.add_id(id_)
        context_ids = input_ids[len(tree_builder_base.ids):]

        logits_preprocessor = PSILogits(tree_builder_base)

        self._model.beam_search()


class PSILogits(LogitsProcessor):
    def __init__(self, tree_builder_base: TreeBuilder):
        super().__init__()
        self._tree_builder_base = tree_builder_base

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for ids in input_ids:
            tree_builder = self._tree_builder_base.copy()
            for id_ in ids:
                tree_builder.add_id(id_)



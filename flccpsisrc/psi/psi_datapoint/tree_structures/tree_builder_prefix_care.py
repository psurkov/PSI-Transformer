import copy
from typing import List, Tuple

from flccpsisrc.psi.psi_datapoint.stateful.tokenizer import TreeTokenizer
from flccpsisrc.psi.psi_datapoint.token_utils import match_tokens, cut_matched_tokens
from flccpsisrc.psi.psi_datapoint.tree_structures.line_breaker import LineBreaker
from flccpsisrc.psi.psi_datapoint.tree_structures.node import Node
from flccpsisrc.psi.psi_datapoint.tree_structures.tree import Tree
from flccpsisrc.psi.psi_datapoint.tree_structures.tree_builder import TreeBuilder, ChangeStatus


class TreeBuilderPrefixCare(TreeBuilder):
    def __init__(self, tree: Tree, tokenizer: TreeTokenizer, rollback_prefix: List[str]):
        super().__init__(tree, tokenizer)
        assert "" not in rollback_prefix
        self._remaining_prefix = rollback_prefix

    def get_next_possible_ids(self):
        ids = super().get_next_possible_ids()
        filtered_ids = []
        for id_ in ids:
            _, nodes = super().add_id(id_, dry_run=True)
            tokens = LineBreaker.node_to_code(nodes)
            fully_matched, next_len = match_tokens(tokens, self._remaining_prefix)
            if fully_matched != -1:
                filtered_ids.append(id_)
        return filtered_ids

    def add_id(self, id_: int, dry_run: bool = False) -> Tuple[ChangeStatus, List[Node]]:
        if dry_run:
            return super().add_id(id_, dry_run=dry_run)
        change_status, nodes = super().add_id(id_, dry_run=dry_run)
        tokens = LineBreaker.node_to_code(nodes)
        _, self._remaining_prefix = cut_matched_tokens(tokens, self._remaining_prefix)
        return change_status, nodes

    def copy(self) -> "TreeBuilderPrefixCare":
        new_nodes = list(copy.deepcopy(self._tree.nodes[0]).dfs_order)
        new_tree = Tree(new_nodes, self._tree.stats_collector)
        new_remaining_prefix = list(copy.deepcopy(self._remaining_prefix))
        tree_builder = TreeBuilderPrefixCare(new_tree, self._tokenizer, new_remaining_prefix)
        tree_builder._cur_arbitrary_ids = copy.copy(self._cur_arbitrary_ids)
        return tree_builder

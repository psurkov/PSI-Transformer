import copy
from typing import List, Tuple

from flccpsisrc.psi.psi_datapoint.stateful.tokenizer import TreeTokenizer
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
            fully_matched, next_len = self._match_prefix(tokens)
            if fully_matched != -1:
                filtered_ids.append(id_)
        return filtered_ids

    def add_id(self, id_: int, dry_run: bool = False) -> Tuple[ChangeStatus, List[Node]]:
        if dry_run:
            return super().add_id(id_, dry_run=dry_run)
        change_status, nodes = super().add_id(id_, dry_run=dry_run)
        tokens = LineBreaker.node_to_code(nodes)
        fully_matched, next_len = self._match_prefix(tokens)
        assert fully_matched != -1
        self._remaining_prefix = self._remaining_prefix[fully_matched:]
        if next_len > 0:
            self._remaining_prefix[0] = self._remaining_prefix[0][next_len:]
        return change_status, nodes

    def copy(self) -> "TreeBuilderPrefixCare":
        new_nodes = list(copy.deepcopy(self._tree.nodes[0]).dfs_order)
        new_tree = Tree(new_nodes, self._tree.stats_collector)
        new_remaining_prefix = list(copy.deepcopy(self._remaining_prefix))
        tree_builder = TreeBuilderPrefixCare(new_tree, self._tokenizer, new_remaining_prefix)
        tree_builder._cur_arbitrary_ids = copy.copy(self._cur_arbitrary_ids)
        return tree_builder

    def _match_prefix(self, tokens: List[str]) -> Tuple[int, int]:
        """
        :return: (fully matched tokens, matched length of next token) or (-1, -1) if doesn't match
        """
        fully = 0
        for pref_token, token in zip(self._remaining_prefix, tokens):
            if pref_token != token:
                min_len = min(len(pref_token), len(token))
                if pref_token[:min_len] != token[:min_len]:
                    return -1, -1
                if len(pref_token) <= len(token):
                    return fully + 1, 0
                else:
                    return fully, len(token)
            fully += 1
        return fully, 0

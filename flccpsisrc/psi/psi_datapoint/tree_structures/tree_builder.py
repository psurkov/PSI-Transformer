import copy
from enum import Enum
from typing import List, Set, Tuple

from flccpsisrc.psi.psi_datapoint.stateful.tokenizer import TreeTokenizer
from flccpsisrc.psi.psi_datapoint.tree_structures.line_breaker import LineBreaker
from flccpsisrc.psi.psi_datapoint.tree_structures.node import TreeConstants, Node
from flccpsisrc.psi.psi_datapoint.tree_structures.tree import Tree


class ChangeStatus(Enum):
    NO_CHANGE = 0
    END_TOKEN = 1
    END_LINE = 2


class TreeBuilder:
    def __init__(self, tree: Tree, tokenizer: TreeTokenizer):
        self._tree = tree
        self._tree.complete_compressed_nodes()
        self._tokenizer = tokenizer

        self._line_breaker = LineBreaker()
        for node in tree.nodes:
            self._line_breaker(node)

        self._cur_arbitrary_ids = []

    @property
    def tree(self) -> Tree:
        return self._tree

    @property
    def ids(self) -> List[int]:
        return self._tokenizer.encode_tree(self._tree)

    def copy(self) -> "TreeBuilder":
        new_nodes = list(copy.deepcopy(self._tree.nodes[0]).dfs_order)
        new_tree = Tree(new_nodes, self._tree.stats_collector)
        tree_builder = TreeBuilder(new_tree, self._tokenizer)
        tree_builder._cur_arbitrary_ids = copy.copy(self._cur_arbitrary_ids)
        return tree_builder

    def add_id(self, id_: int, dry_run: bool = False) -> Tuple[ChangeStatus, List[Node]]:
        is_arbitrary, _, _ = self._tokenizer.classify_ids(id_)

        added_nodes = []
        if id_ == self._tokenizer.eov_id:
            token = self._tokenizer.decode_arbitrary_string(self._cur_arbitrary_ids)
            if not dry_run:
                self._cur_arbitrary_ids = []
            added_nodes.extend(self._tree.add_node(token, is_arbitrary=True, dry_run=dry_run))
        elif is_arbitrary:
            if not dry_run:
                self._cur_arbitrary_ids.append(id_)
        else:
            token = self._tokenizer.decode(id_)
            added_nodes.extend(self._tree.add_node(token, is_arbitrary=False, dry_run=dry_run))

        added_token = False
        added_new_line = False
        for node in added_nodes:
            if node.is_leaf and node.name:
                added_token = True
            new_line, indent = self._line_breaker(node)
            if new_line:
                added_new_line = True

        if added_new_line:
            return ChangeStatus.END_LINE, added_nodes
        elif added_token:
            return ChangeStatus.END_TOKEN, added_nodes
        else:
            return ChangeStatus.NO_CHANGE, added_nodes

    def get_next_possible_ids(self) -> Set[int]:
        if self._cur_arbitrary_ids:
            arbitrary_tokens = set(self._tokenizer.arbitrary_ids)
            arbitrary_tokens.add(self._tokenizer.eov_id)
            return arbitrary_tokens
        else:
            next_suitable_nodes = self._tree.get_next_suitable_nodes()
            if next_suitable_nodes is None:
                return set()
            ret = set()
            for node_name in next_suitable_nodes:
                if node_name == TreeConstants.ARBITRARY_REPR.value:
                    ret.update(self._tokenizer.arbitrary_ids)
                else:
                    ret.add(self._tokenizer.encode_non_arbitrary_token(node_name))
            return ret

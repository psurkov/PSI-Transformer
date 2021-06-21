import copy

from src.psi.psi_datapoint.stateful.tokenizer import TreeTokenizer
from src.psi.psi_datapoint.tree_structures.node import TreeConstants
from src.psi.psi_datapoint.tree_structures.tree import Tree


class TreeBuilder:
    def __init__(self, tree: Tree, tokenizer: TreeTokenizer):
        self._tree = tree
        self._tree.complete_compressed_nodes()
        self._tokenizer = tokenizer

        self._cur_arbitrary_ids = []

    @property
    def tree(self) -> Tree:
        return self._tree

    @property
    def ids(self) -> list[int]:
        return self._tokenizer.encode_tree(self._tree)

    def copy(self) -> "TreeBuilder":
        new_nodes = list(copy.deepcopy(self._tree.nodes[0]).dfs_order)
        new_tree = Tree(new_nodes, self._tree.stats_collector)
        return TreeBuilder(new_tree, self._tokenizer)

    def get_next_possible_ids(self) -> set[int]:
        if self._cur_arbitrary_ids:
            return set(self._tokenizer.arbitrary_ids)
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

    def add_id(self, id_: int) -> None:
        is_arbitrary, _, _ = self._tokenizer.classify_ids(id_)

        if id_ == self._tokenizer.eov_id:
            token = self._tokenizer.decode_arbitrary_string(self._cur_arbitrary_ids)
            self._cur_arbitrary_ids = []
            self._tree.add_node(name=token, is_arbitrary=True)
        elif is_arbitrary:
            self._cur_arbitrary_ids.append(id_)
        else:
            token = self._tokenizer.decode(id_)
            self._tree.add_node(token, is_arbitrary=False)

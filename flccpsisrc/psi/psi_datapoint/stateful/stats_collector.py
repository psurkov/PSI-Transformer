import json
import os
from typing import Optional, Dict, List, Set

from tqdm import tqdm

from flccpsisrc.psi.psi_datapoint.stateful.abstract_stateful import Stateful
from flccpsisrc.psi.psi_datapoint.tree_structures.node import Node, TreeConstants


class StatsCollector(Stateful):
    _filename = "psi/tree_stats_collector.json"

    def __init__(self, inference_mode: bool):
        # Set: first nodes' name
        self._first_node_names: Optional[Set[str]] = None
        # Dict: fixed length node name -> sets of occurred child names for each position
        self._fixed_children_names: Optional[Dict[str, List[Set[str]]]] = None
        # Dict: non-fixed length node name -> set of all possible children on all positions
        self._arbitrary_children_names: Optional[Dict[str, Set[str]]] = None
        # Set with nodes which always has the same number of children
        self._children_amount: Optional[Dict[str, int]] = None
        # Set: leaf node names with non arbitrary value
        self._non_arbitrary_leafs: Optional[Set[str]] = None

        self._is_trained: bool = False
        self._inference_mode = inference_mode

    def save_pretrained(self, path: str) -> None:
        path = os.path.join(path, StatsCollector._filename)
        first_node_names = list(self._first_node_names) if self._first_node_names is not None else None
        fixed_children_names = (
            {
                node_name: [list(children_set) for children_set in children_sets]
                for node_name, children_sets in self._fixed_children_names.items()
            }
            if self._fixed_children_names is not None
            else None
        )
        arbitrary_children_names = (
            {node_name: list(children_set) for node_name, children_set in self._arbitrary_children_names.items()}
            if self._arbitrary_children_names is not None
            else None
        )
        children_amount = self._children_amount
        non_arbitrary_leafs = list(self._non_arbitrary_leafs) if self._non_arbitrary_leafs is not None else None
        is_trained = self._is_trained

        with open(path, "w") as f:
            json.dump(
                [
                    first_node_names,
                    fixed_children_names,
                    arbitrary_children_names,
                    children_amount,
                    non_arbitrary_leafs,
                    is_trained,
                ],
                f,
            )

    @staticmethod
    def from_pretrained(path: str, inference_mode: bool) -> "StatsCollector":
        path = os.path.join(path, StatsCollector._filename)
        stats = StatsCollector(inference_mode=inference_mode)
        with open(path) as f:
            [
                first_node_names,
                fixed_children_names,
                arbitrary_children_names,
                children_amount,
                non_arbitrary_leafs,
                is_trained,
            ] = json.load(f)
        stats._first_node_names = set(first_node_names) if first_node_names is not None else None
        stats._fixed_children_names = (
            {
                node_name: [set(children_set) for children_set in children_sets]
                for node_name, children_sets in fixed_children_names.items()
            }
            if fixed_children_names is not None
            else None
        )
        stats._arbitrary_children_names = (
            {node_name: list(children_set) for node_name, children_set in arbitrary_children_names.items()}
            if arbitrary_children_names is not None
            else None
        )
        stats._children_amount = children_amount
        stats._non_arbitrary_leafs = set(non_arbitrary_leafs) if non_arbitrary_leafs is not None else None
        stats._is_trained = is_trained

        return stats

    @staticmethod
    def pretrained_exists(path: str) -> bool:
        return os.path.exists(os.path.join(path, StatsCollector._filename))

    def train(self, trees: List[List[Node]]) -> List[List[Node]]:
        children_amount = dict()
        non_arbitrary_leafs = set()
        first_node_names = set()

        for tree in tqdm(trees, desc="Collecting stats from trees 1/2..."):
            if tree:
                first_node_names.add(tree[0].name)
            for node in tree:
                if node.is_leaf:
                    if not node.is_arbitrary:
                        non_arbitrary_leafs.add(node.name)
                else:
                    if node.name not in children_amount:
                        children_amount[node.name] = set()
                    children_amount[node.name].add(len(node.children))

        self._children_amount = {
            name: next(iter(children_amount_list)) if len(children_amount_list) == 1 else -1
            for name, children_amount_list in children_amount.items()
        }
        non_arbitrary_leafs.add(TreeConstants.END_OF_CHILDREN.value)
        self._non_arbitrary_leafs = non_arbitrary_leafs
        self._first_node_names = first_node_names

        fixed_children_names = {}
        arbitrary_children_names = {}
        trees_with_end_of_children = [
            self._set_arbitrary_children_amount(tree) for tree in tqdm(trees, desc="Setting end of children child...")
        ]

        for tree in tqdm(trees_with_end_of_children, desc="Collecting stats from trees 2/2..."):
            for node in tree:
                if not node.is_leaf:
                    if self.has_static_children_amount(node.name):
                        if node.name not in fixed_children_names:
                            fixed_children_names[node.name] = [
                                set() for _ in range(self.get_children_amount(node.name))
                            ]

                        for i, child in enumerate(node.children):
                            child_name = child.name if not child.is_arbitrary else TreeConstants.ARBITRARY_REPR.value
                            fixed_children_names[node.name][i].add(child_name)
                    else:
                        if node.name not in arbitrary_children_names:
                            arbitrary_children_names[node.name] = set()
                        for child in node.children:
                            child_name = child.name if not child.is_arbitrary else TreeConstants.ARBITRARY_REPR.value
                            arbitrary_children_names[node.name].add(child_name)

        self._fixed_children_names = fixed_children_names
        self._arbitrary_children_names = arbitrary_children_names

        self._is_trained = True
        return [self._compress(tree) for tree in tqdm(trees_with_end_of_children, desc="Compressing trees...")]

    def transform(self, nodes: List[Node]) -> Optional[List[Node]]:
        try:
            return self._compress(self._set_arbitrary_children_amount(nodes))
        except (KeyError, AssertionError, IndexError) as e:
            print(f"Failed to compress tree:\n {e}")
            return None

    def _set_arbitrary_children_amount(self, nodes: List[Node]) -> List[Node]:
        incomplete_nodes = set()
        if self._inference_mode:
            right_node = nodes[0]
            incomplete_nodes.add(id(right_node))
            while not right_node.is_leaf:
                right_node = right_node.children[-1]
                incomplete_nodes.add(id(right_node))

        for node in nodes:
            if (
                not node.is_leaf
                and not self.has_static_children_amount(node.name)
                and id(node) not in incomplete_nodes
            ):
                end_of_children_node = Node(
                    TreeConstants.END_OF_CHILDREN.value,
                    is_arbitrary=False,
                    is_leaf=True,
                )
                node.add_child(end_of_children_node)
        return list(nodes[0].dfs_order) if nodes else []

    def _compress(self, nodes: List[Node]) -> List[Node]:
        for node in nodes:
            if not node.is_leaf:
                for i, child_to_compress in enumerate(node.children):
                    possible_children = self.get_children_names(node.name, i)
                    if len(possible_children) == 1:
                        the_possible_child = next(iter(possible_children))
                        if the_possible_child != TreeConstants.ARBITRARY_REPR.value:
                            assert child_to_compress.name == the_possible_child, (
                                f"{node.name} {i}th child: "
                                f"has {child_to_compress.name} node but expected {the_possible_child}"
                            )
                            child_to_compress.set_visible(False)
        return nodes

    def get_children_names(self, node_name: str, child_index: int) -> Set[str]:
        assert self._fixed_children_names is not None and self._arbitrary_children_names is not None
        if self.has_static_children_amount(node_name):
            return self._fixed_children_names[node_name][child_index]
        else:
            return self._arbitrary_children_names[node_name]

    def get_children_amount(self, node_name: str) -> int:
        assert self._children_amount is not None
        return self._children_amount[node_name]

    def has_static_children_amount(self, node_name: str) -> bool:
        assert self._children_amount is not None
        return self._children_amount[node_name] != -1

    def is_non_arbitrary_leaf(self, node_name: str) -> bool:
        assert self._non_arbitrary_leafs is not None
        return node_name in self._non_arbitrary_leafs

    def get_first_node_names(self) -> Set[str]:
        return self._first_node_names

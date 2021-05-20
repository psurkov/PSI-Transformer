import json
import os
from typing import Optional, Dict, List, Set

from tqdm import tqdm

from psi_datapoint.stateful.abstract_stateful import Stateful
from psi_datapoint.tree_structures.node import Node, TreeConstants


class StatsCollector(Stateful):
    _filename = "tree_stats_collector.json"

    def __init__(self):
        # Set: first nodes' name
        self._first_node_names: Optional[Set[str]] = None
        # Dict: node name -> sets of occurred child names for each position
        self._children_names: Optional[Dict[str, List[Set[str]]]] = None
        # Dict: node name -> set of last child (before [END OF CHILDREN]
        # self._last_child_names: Optional[Dict[str, Set[str]]] = None
        # Set with nodes which always has the same number of children
        self._nodes_with_static_children: Optional[Set[str]] = None
        # Set: leaf node names with non arbitrary value
        self._non_arbitrary_leafs: Optional[Set[str]] = None

        self._is_trained: bool = False

    def save_pretrained(self, path: str) -> None:
        path = os.path.join(path, StatsCollector._filename)
        first_node_names = list(self._first_node_names) if self._first_node_names is not None else None
        children_names = (
            {
                node_name: [list(children_set) for children_set in children_sets]
                for node_name, children_sets in self._children_names.items()
            }
            if self._children_names is not None
            else None
        )
        # last_child_names = (
        #     {node_name: list(children_set) for node_name, children_set in self._last_child_names.items()}
        #     if self._last_child_names is not None
        #     else None
        # )
        nodes_with_static_children = list(self._nodes_with_static_children)
        non_arbitrary_leafs = list(self._non_arbitrary_leafs) if self._non_arbitrary_leafs is not None else None
        is_trained = self._is_trained

        with open(path, "w") as f:
            json.dump(
                [
                    first_node_names,
                    children_names,
                    # last_child_names,
                    nodes_with_static_children,
                    non_arbitrary_leafs,
                    is_trained,
                ],
                f,
            )

    @staticmethod
    def from_pretrained(path: str) -> "StatsCollector":
        path = os.path.join(path, StatsCollector._filename)
        stats = StatsCollector()
        with open(path) as f:
            [
                first_node_names,
                children_names,
                # last_child_names,
                nodes_with_static_children,
                non_arbitrary_leafs,
                is_trained,
            ] = json.load(f)
        stats._first_node_names = set(first_node_names) if first_node_names is not None else None
        stats._children_names = (
            {
                node_name: [set(children_set) for children_set in children_sets]
                for node_name, children_sets in children_names.items()
            }
            if children_names is not None
            else None
        )
        # stats._last_child_names = (
        #     {node_name: set(children_set) for node_name, children_set in last_child_names.items()}
        #     if last_child_names is not None
        #     else None
        # )
        stats._nodes_with_static_children = list(nodes_with_static_children)
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

        self._nodes_with_static_children = set(
            name for name, children_amount_list in children_amount.items() if len(children_amount_list) == 1
        )
        non_arbitrary_leafs.add(TreeConstants.END_OF_CHILDREN.value)
        self._non_arbitrary_leafs = non_arbitrary_leafs
        self._first_node_names = first_node_names

        children_names = {}
        # last_child_names = {}
        trees_with_end_of_children = [
            self._set_arbitrary_children_amount(tree) for tree in tqdm(trees, desc="Setting end of children child...")
        ]

        for tree in tqdm(trees_with_end_of_children, desc="Collecting stats from trees 2/2..."):
            for node in tree:
                if not node.is_leaf:
                    if node.name not in children_names:
                        children_names[node.name] = [set() for _ in range(len(node.children))]

                    elif len(children_names[node.name]) != len(node.children):
                        children_stats_to_add = len(node.children) - len(children_names[node.name])
                        if children_stats_to_add > 0:
                            children_names[node.name].extend(set() for _ in range(children_stats_to_add))

                    for i, child in enumerate(node.children):
                        child_name = child.name if not child.is_arbitrary else TreeConstants.ARBITRARY_REPR.value
                        children_names[node.name][i].add(child_name)

                    # if not self.has_static_children_amount(node.name):
                    #     if node.name not in last_child_names:
                    #         last_child_names[node.name] = set()
                    #     assert node.children[-1].name == TreeConstants.END_OF_CHILDREN.value
                    #     if len(node.children) > 1:
                    #         last_child = node.children[-2]
                    #         child_name = (
                    #             last_child.name if not last_child.is_arbitrary else TreeConstants.ARBITRARY_REPR.value
                    #         )
                    #         last_child_names[node.name].add(child_name)

        self._children_names = children_names
        # self._last_child_names = last_child_names

        self._is_trained = True
        return [self._compress(tree) for tree in tqdm(trees_with_end_of_children, desc="Compressing trees...")]

    def transform(self, nodes: List[Node]) -> List[Node]:
        return self._compress(self._set_arbitrary_children_amount(nodes))

    def _set_arbitrary_children_amount(self, nodes: List[Node]) -> List[Node]:
        for node in nodes:
            if not node.is_leaf and not self.has_static_children_amount(node.name):
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
                possible_children = self.get_children_names(node.name)
                for child_to_compress, possible_child in zip(node.children, possible_children):
                    if len(possible_child) == 1:
                        the_possible_child = next(iter(possible_child))
                        if the_possible_child != TreeConstants.ARBITRARY_REPR.value:
                            assert child_to_compress.name == the_possible_child
                            child_to_compress.set_visible(False)
                # if not self.has_static_children_amount(node.name) and len(self.get_last_child_names(node.name)) == 1:
                #     if len(node.children) > 1:
                #         assert node.children[-2].name == next(iter(self.get_last_child_names(node.name)))
                #         node.children[-2].set_visible(False)
        return nodes

    def get_children_names(self, node_name: str) -> List[Set[str]]:
        assert self._children_names is not None
        return self._children_names[node_name]

    # def get_last_child_names(self, node_name: str) -> Set[str]:
    #     assert self._last_child_names is not None
    #     return self._last_child_names[node_name]

    def has_static_children_amount(self, node_name: str) -> bool:
        assert self._nodes_with_static_children is not None
        return node_name in self._nodes_with_static_children

    def is_non_arbitrary_leaf(self, node_name: str) -> bool:
        assert self._non_arbitrary_leafs is not None
        return node_name in self._non_arbitrary_leafs

    def get_first_node_names(self) -> Set[str]:
        return self._first_node_names

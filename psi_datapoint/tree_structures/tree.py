from typing import List, Optional, Set

from psi_datapoint.stateful.stats_collector import StatsCollector
from psi_datapoint.tree_structures.node import Node, TreeConstants


class Tree:
    def __init__(self, nodes_dfs_order: List[Node], stats: StatsCollector):
        self._nodes_dfs_order = nodes_dfs_order
        self._stats = stats

    @property
    def nodes(self) -> List[Node]:
        return self._nodes_dfs_order

    @property
    def size(self) -> int:
        return len(self._nodes_dfs_order)

    @property
    def compressed_size(self) -> int:
        return sum(1 for node in self._nodes_dfs_order if node.is_visible)

    def _get_node_with_next_child(self) -> Optional[Node]:
        assert self._nodes_dfs_order

        for node in reversed(self._nodes_dfs_order):
            if node.is_leaf:
                continue
            elif self._stats.has_static_children_amount(node.name):
                children_names = self._stats.get_children_names(node.name)
                if len(children_names) == len(node.children):
                    continue
                else:
                    return node
            else:  # Arbitrary number of children
                if node.children and node.children[-1].name == TreeConstants.END_OF_CHILDREN.value:
                    continue
                return node
        return None

    def get_next_suitable_nodes(self) -> Optional[Set[str]]:
        if not self._nodes_dfs_order:
            return self._stats.get_first_node_names()
        parent_with_next_child = self._get_node_with_next_child()
        if parent_with_next_child is None:
            return None

        children_names = self._stats.get_children_names(parent_with_next_child.name)
        res = children_names[len(parent_with_next_child.children)]

        if self._stats.has_static_children_amount(parent_with_next_child.name):
            res.add(TreeConstants.ARBITRARY_REPR.value)
        # res.remove(PSIConstants.ERROR_NAME.value)
        return res

    def add_node(self, name: str, is_arbitrary: bool) -> None:
        is_leaf = is_arbitrary or self._stats.is_non_arbitrary_leaf(name)
        node = Node(name, is_arbitrary, is_leaf)

        if not self._nodes_dfs_order:
            self._nodes_dfs_order = [node]
            self._complete_compressed_nodes()
            return

        parent_with_next_child = self._get_node_with_next_child()
        assert parent_with_next_child is not None

        # if name == TreeConstants.END_OF_CHILDREN.value:
        #     assert not self._stats.has_static_children_amount(parent_with_next_child.name)
        #     self._complete_compressed_last_child(parent_with_next_child)

        parent_with_next_child.add_child(node)
        self._nodes_dfs_order.append(node)
        self._complete_compressed_nodes()

    def _complete_compressed_nodes(self) -> None:
        parent_with_next_child = self._get_node_with_next_child()
        if parent_with_next_child is None:
            return
        next_children_names = self._stats.get_children_names(parent_with_next_child.name)[
            len(parent_with_next_child.children)
        ]
        if len(next_children_names) == 1:
            name = next(iter(next_children_names))
            if name == TreeConstants.ARBITRARY_REPR.value:
                return
            is_leaf = self._stats.is_non_arbitrary_leaf(name)
            node = Node(name, is_arbitrary=False, is_leaf=is_leaf)
            parent_with_next_child.add_child(node)
            self._nodes_dfs_order.append(node)
            self._complete_compressed_nodes()

    # def _complete_compressed_last_child(self, node: Node) -> None:
    #     assert not self._stats.has_static_children_amount(node.name)
    #     last_child_names = self._stats.get_last_child_names(node.name)
    #     if len(last_child_names) == 1:
    #         last_child_name = next(iter(last_child_names))
    #         assert last_child_name != TreeConstants.ARBITRARY_REPR.value
    #         assert node.children
    #         if last_child_name != node.children[-1]:
    #             is_leaf = self._stats.is_non_arbitrary_leaf(last_child_name)
    #             last_child_node = Node(last_child_name, is_arbitrary=False, is_leaf=is_leaf)
    #             node.add_child(last_child_node)
    #             self._nodes_dfs_order.append(node)

    @property
    def program(self) -> str:
        return self._nodes_dfs_order[0].program

    @property
    def tree_representation(self) -> str:
        return self._nodes_dfs_order[0].tree_representation

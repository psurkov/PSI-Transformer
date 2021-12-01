from typing import List, Optional, Set

from flccpsisrc.psi.psi_datapoint.stateful.stats_collector import StatsCollector
from flccpsisrc.psi.psi_datapoint.tree_structures.node import Node, TreeConstants


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
        return sum(node.is_visible for node in self._nodes_dfs_order)

    @property
    def stats_collector(self) -> StatsCollector:
        return self._stats

    def _get_node_with_next_child(self) -> Optional[Node]:
        for node in reversed(self._nodes_dfs_order):
            if node.is_leaf:
                continue
            elif self._stats.has_static_children_amount(node.name):
                if self._stats.get_children_amount(node.name) == len(node.children):
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

        children_names = self._stats.get_children_names(
            parent_with_next_child.name, len(parent_with_next_child.children)
        )

        if self._stats.has_static_children_amount(parent_with_next_child.name):
            children_names.add(TreeConstants.ARBITRARY_REPR.value)
        return children_names

    def add_node(self, name: str, is_arbitrary: bool, dry_run: bool) -> List[Node]:
        is_leaf = is_arbitrary or self._stats.is_non_arbitrary_leaf(name)
        node = Node(name, is_arbitrary, is_leaf)

        added_nodes = []
        if not self._nodes_dfs_order:
            self._nodes_dfs_order = [node]
            added_nodes.append(node)
            added_nodes.extend(self.complete_compressed_nodes())
            if dry_run:
                self._nodes_dfs_order = None  # restore
            return added_nodes

        parent_with_next_child = self._get_node_with_next_child()
        assert parent_with_next_child is not None

        parent_with_next_child.add_child(node)
        self._nodes_dfs_order.append(node)
        added_nodes.append(node)
        added_nodes.extend(self.complete_compressed_nodes(dry_run=dry_run))
        if dry_run:
            self._nodes_dfs_order.pop()  # restore: remove added node
            parent_with_next_child.pop_child()  # restore: remove added child
        return added_nodes

    def complete_compressed_nodes(self, dry_run: bool = False) -> List[Node]:
        completed_nodes = []
        parent_modifications = []
        while True:
            parent_with_next_child = self._get_node_with_next_child()
            if parent_with_next_child is None:
                if dry_run:
                    self._complete_compressed_nodes_restore(parent_modifications)
                return completed_nodes
            next_children_names = self._stats.get_children_names(
                parent_with_next_child.name, len(parent_with_next_child.children)
            )
            if len(next_children_names) == 1:
                name = next(iter(next_children_names))
                if name == TreeConstants.ARBITRARY_REPR.value:
                    if dry_run:
                        self._complete_compressed_nodes_restore(parent_modifications)
                    return completed_nodes
                is_leaf = self._stats.is_non_arbitrary_leaf(name)
                node = Node(name, is_arbitrary=False, is_leaf=is_leaf)

                parent_with_next_child.add_child(node)
                parent_modifications.append(parent_with_next_child)

                self._nodes_dfs_order.append(node)
                completed_nodes.append(node)
            else:
                if dry_run:
                    self._complete_compressed_nodes_restore(parent_modifications)
                return completed_nodes

    def _complete_compressed_nodes_restore(self, parent_modifications):
        if len(parent_modifications) == 0:
            return
        del self._nodes_dfs_order[-len(parent_modifications):]
        for p in parent_modifications:
            p.pop_child()

from typing import List

from psi_datapoint.stateless_transformations.abstract_transformation import Transformation
from psi_datapoint.tree_structures.node import Node, PSIConstants


class WhitespaceNormalizer(Transformation):
    WHITE_SPACE = "WS:WHITE_SPACE"
    NEW_LINE = "WS:NEW_LINE"

    def transform(self, nodes: List[Node]) -> List[Node]:
        for node in nodes:
            if node.name == PSIConstants.WHITE_SPACE_NAME.value:
                # assert len(node.children) == 1
                if "\n" in node.children[0].name:
                    node._name = WhitespaceNormalizer.NEW_LINE
                    node._set_children((Node("\n", is_arbitrary=False, is_leaf=True),))
                else:
                    node._name = WhitespaceNormalizer.WHITE_SPACE
                    node._set_children((Node(" ", is_arbitrary=False, is_leaf=True),))
        return list(nodes[0].dfs_order)

    def inverse_transform(self, nodes: List[Node]) -> List[Node]:
        indent_level = 0
        prev_node = None
        for node in nodes:
            if node.name == WhitespaceNormalizer.NEW_LINE:
                node.children[0]._name = f"\n{'    ' * indent_level}"
            elif node.name in PSIConstants.INDENT_IN_NAMES.value:
                indent_level += 1
            elif node.name in PSIConstants.INDENT_OUT_NAMES.value:
                if "\n" in prev_node.name:
                    assert len(prev_node.name) > 4
                    prev_node._name = prev_node.name[:-4]
                indent_level -= 1

            prev_node = node

        return nodes

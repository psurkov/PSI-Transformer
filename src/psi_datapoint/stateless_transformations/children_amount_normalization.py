import re
from typing import List

from src.psi_datapoint.stateless_transformations.abstract_transformation import Transformation
from src.psi_datapoint.tree_structures.node import Node


_EMPTY_CHILDREN_COMPRESSION = re.compile("|".join(("\\w*_LIST$",)))
_SINGLE_CHILD_COMPRESSION = re.compile("|".join(("MODIFIER_LIST",)))
_NO_COMPRESSION = re.compile(
    "|".join(
        (
            "java.FILE",
            "DOC_COMMENT",
            "DOC_INLINE_TAG",
            "CODE_BLOCK",
            "DOC_TAG",
            "DOC_TAG_VALUE_ELEMENT",
            "ARRAY_INITIALIZER_EXPRESSION",
            "POLYADIC_EXPRESSION",
            "ANNOTATION_ARRAY_INITIALIZER",
            "\\w*CLASS$",
        )
    )
)
# THE REST OF THE NODES WILL BE FULLY COMPRESSED !!!


class ChildrenAmountNormalizer(Transformation):
    def transform(self, nodes: List[Node]) -> List[Node]:
        for node in nodes:
            if not node.is_leaf:
                if not _NO_COMPRESSION.search(node.name):
                    is_empty_compression = bool(_EMPTY_CHILDREN_COMPRESSION.search(node.name))
                    is_single_compression = bool(_SINGLE_CHILD_COMPRESSION.search(node.name))
                    if is_empty_compression and len(node.children) == 0:
                        ChildrenAmountNormalizer._modify_node(node, f"{node.name}|EMPTY")
                    elif is_single_compression and len(node.children) == 1:
                        ChildrenAmountNormalizer._modify_node(node, f"{node.name}|{node.children[0].name}")
                    elif is_empty_compression or is_single_compression:
                        ChildrenAmountNormalizer._modify_node(node, f"{node.name}|ARB")
                    else:
                        ChildrenAmountNormalizer._modify_node(node, f"{node.name}|{len(node.children)}")
        return list(nodes[0].dfs_order)

    def inverse_transform(self, nodes: List[Node]) -> List[Node]:
        return nodes

    @staticmethod
    def _modify_node(node: Node, desired_name: str) -> None:
        node._name = f"{node.name}|SUPER"
        fiction_child = Node(desired_name, is_arbitrary=False, is_leaf=False)
        fiction_child._set_children(node.children)
        node._set_children((fiction_child,))

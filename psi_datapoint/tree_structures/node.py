import json
from enum import Enum
from json import JSONDecodeError
from typing import Tuple, Iterable, Optional, List


class PSIConstants(Enum):
    EMPTY_VALUE = "<EMPTY>"
    LEAF_TYPE = "<VALUE>"
    ERROR_NAME = "ERROR_ELEMENT"
    ARBITRARY_VALUES = {
        "IDENTIFIER",
        "LITERAL_EXPRESSION",
        "DOC_COMMENT_DATA",
        "DOC_PARAMETER_REF",
        "DOC_TAG_VALUE_ELEMENT",
        "DOC_TAG_VALUE_TOKEN",
        "END_OF_LINE_COMMENT",
        "C_STYLE_COMMENT",
    }
    WHITE_SPACE_NAME = "WHITE_SPACE"
    INDENT_IN_NAMES = {"LBRACE"}
    INDENT_OUT_NAMES = {"RBRACE"}


class TreeConstants(Enum):
    END_OF_CHILDREN = "[EOC]"
    ARBITRARY_REPR = "[ARB]"


class Node:
    def __init__(
        self,
        name: str,
        is_arbitrary: bool,
        is_leaf: bool,
    ):
        self._name = name
        self._is_arbitrary = is_arbitrary
        self._is_leaf = is_leaf
        self._is_visible = True

        # Can be None in case of leafs
        # Can be tuple() in case of 0 children
        # Can be non-empty tuple in case of >0 children
        self._children = None if is_leaf else tuple()

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_arbitrary(self) -> bool:
        return self._is_arbitrary

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    @property
    def is_visible(self) -> bool:
        return self._is_visible

    def set_visible(self, visible: bool) -> None:
        self._is_visible = visible

    @property
    def children(self) -> Optional[Tuple["Node"]]:
        return self._children

    def _set_children(self, children: Tuple["Node"]):
        assert self._children is not None
        self._children = children

    def add_child(self, child: "Node"):
        assert self._children is not None
        self._children = self._children + (child,)

    @property
    def dfs_order(self) -> Iterable["Node"]:
        yield self
        if self.children:
            for child in self.children:
                yield from child.dfs_order

    @property
    def program(self) -> str:
        return "".join(
            node.name for node in self.dfs_order if node.is_leaf and node.name != TreeConstants.END_OF_CHILDREN.value
        )

    @property
    def tree_representation(self) -> str:
        return "\n".join(self._show_tree())

    def _show_tree(self, indent: int = 0) -> Iterable[str]:
        yield (
            f"{'->' * indent}"
            f"{'<' if not self.is_visible else ''}"
            f"{'~' if self.is_arbitrary else ''}"
            f"{repr(self.name)}"
            f"{'~' if self.is_arbitrary else ''}"
            f"{'>' if not self.is_visible else ''}"
        )
        if self.children:
            for child in self.children:
                yield from child._show_tree(indent=indent + 1)

    @staticmethod
    def load_psi_miner_nodes(json_string: str) -> Optional[List["Node"]]:
        try:
            json_dict = json.loads(json_string)
        except JSONDecodeError:
            return None
        node_dicts, label = json_dict["AST"], json_dict["label"]

        nodes = [
            Node._load_from_psi_miner_format(node_dict["node"], node_dict["token"], is_leaf="children" not in node_dict)
            for node_dict in node_dicts
        ]
        for node_dict, node in zip(node_dicts, nodes):
            if "children" in node_dict:
                children = tuple(nodes[i] for i in node_dict["children"])
                node._set_children(children)
        return list(nodes[0].dfs_order)

    @staticmethod
    def _load_from_psi_miner_format(
        node_name: str, node_value: str, is_leaf: bool, arbitrary_value: bool = False
    ) -> "Node":
        children = None

        if node_value != PSIConstants.EMPTY_VALUE.value:
            assert is_leaf
            is_leaf = False

            if node_value != "":
                base_names = node_name.split("|")
                is_arbitrary = any(base_name in PSIConstants.ARBITRARY_VALUES.value for base_name in base_names)

                value_child = Node._load_from_psi_miner_format(
                    node_name=node_value,
                    node_value=PSIConstants.EMPTY_VALUE.value,
                    is_leaf=True,
                    arbitrary_value=is_arbitrary,
                )
                children = (value_child,)
            else:
                children = tuple()

        node = Node(node_name, arbitrary_value, is_leaf)
        if children is not None:
            node._set_children(children)
        return node

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Iterable, Optional, List


class PSIConstants(Enum):
    EMPTY_VALUE = "<EMPTY>"
    ERROR_NAME = "ERROR_ELEMENT"


_NODE_NAME_TO_VALUE = {"DOC_COMMENT_START": "/*", "DOC_COMMENT_END": "*/"}

# Whitespaces stuff
_IS_NEED_NEWLINE = re.compile(
    "|".join(
        f"{node}\\w*"
        for node in (
            "SEMICOLON",
            "LBRACE",
            "RBRACE",
            "DOC_COMMENT_END",
            "END_OF_LINE_COMMENT",
            "C_STYLE_COMMENT",
        )
    )
)
_TO_OFF_NEWLINE = re.compile("|".join(f"{node}\\w*" for node in ("FOR_STATEMENT",)))
_START_OFF = re.compile("|".join(f"{node}\\w*" for node in ("LPARENTH",)))
_END_OFF = re.compile("|".join(f"{node}\\w*" for node in ("RPARENTH",)))
_INDENT_IN = re.compile("|".join(f"{node}\\w*" for node in ("LBRACE",)))
_INDENT_OUT = re.compile("|".join(f"{node}\\w*" for node in ("RBRACE",)))

_ARBITRARY_NAME_REGEX = re.compile(
    "|".join(
        f"{node}"
        for node in (
            "\\w*_LITERAL",
            "IDENTIFIER",
            "DOC_COMMENT_DATA",
            "DOC_PARAMETER_REF",
            "DOC_TAG_VALUE_TOKEN",
            "END_OF_LINE_COMMENT",
            "C_STYLE_COMMENT",
        )
    )
)


class TreeConstants(Enum):
    END_OF_CHILDREN = "[EOC]"
    ARBITRARY_REPR = "[ARB]"


@dataclass
class Node:
    __slots__ = ("_name", "_is_arbitrary", "_is_leaf", "_is_visible", "_children")

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
        indent = "    "
        need_newline = False
        indent_level = 0
        parenth_cnt = None
        strings = []
        for node in self.dfs_order:
            if _TO_OFF_NEWLINE.search(node.name):
                parenth_cnt = -1  # -1 means waiting for the first _START_OFF
            if parenth_cnt == -1 and _START_OFF.search(node.name):
                parenth_cnt = 1
            elif parenth_cnt is not None and _START_OFF.search(node.name):
                parenth_cnt += 1
            elif parenth_cnt is not None and _END_OFF.search(node.name):
                parenth_cnt -= 1

            if parenth_cnt is None or parenth_cnt == 0:
                parenth_cnt = None
                if _IS_NEED_NEWLINE.search(node.name):
                    need_newline = True
                if _INDENT_IN.search(node.name):
                    indent_level += 1
                if _INDENT_OUT.search(node.name):
                    assert f"\n{indent}" in strings[-1]
                    strings[-1] = strings[-1][:-4]
                    indent_level -= 1

            if node.is_leaf and node.name != TreeConstants.END_OF_CHILDREN.value:
                strings.append(node.name)
                if need_newline:
                    strings.append(f"\n{indent * indent_level}")
                    need_newline = False
                else:
                    strings.append(" ")

        return "".join(strings)

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
    def load_psi_miner_nodes(json_dict: dict) -> Optional[List["Node"]]:
        node_dicts, label = json_dict["AST"], json_dict["label"]

        nodes = []
        offsets = []

        offset = 0
        for node_dict in node_dicts:
            cur_nodes, cur_offset = Node._load_from_psi_miner_format(
                node_dict["node"],
                node_dict["token"],
                is_leaf="children" not in node_dict,
            )
            nodes.extend(cur_nodes)
            offsets.append(offset)
            offset += cur_offset

        for i, (node_dict, offset) in enumerate(zip(node_dicts, offsets)):
            node = nodes[i + offset]
            if "children" in node_dict:
                children = tuple(nodes[j + offsets[j]] for j in node_dict["children"])
                node._set_children(children)
        return nodes

    @staticmethod
    def _load_from_psi_miner_format(
        node_name: str,
        node_value: str,
        is_leaf: bool,
        arbitrary_value: bool = False,
    ) -> Tuple[Tuple["Node", ...], int]:
        children = None

        if node_value != PSIConstants.EMPTY_VALUE.value:
            assert is_leaf
            is_leaf = False

            node_value = _NODE_NAME_TO_VALUE.get(node_name, node_value)

            if node_value != "":
                is_arbitrary = bool(_ARBITRARY_NAME_REGEX.search(node_name))

                value_child, _ = Node._load_from_psi_miner_format(
                    node_name=node_value,
                    node_value=PSIConstants.EMPTY_VALUE.value,
                    is_leaf=True,
                    arbitrary_value=is_arbitrary,
                )
                [value_child] = value_child
                children = (value_child,)
            else:
                children = tuple()

        node = Node(node_name, arbitrary_value, is_leaf)
        if children is not None:
            node._set_children(children)
        if children:
            return (node,) + children, len(children)
        else:
            return (node,), 0

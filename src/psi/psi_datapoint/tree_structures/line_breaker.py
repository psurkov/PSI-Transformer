import re
from typing import Tuple, List

from src.psi.psi_datapoint.tree_structures.node import Node, TreeConstants


class LineBreaker:

    _new_line_nodes = [
        "SEMICOLON",
        "LBRACE",
        "RBRACE",
        "DOC_COMMENT_END",
        "END_OF_LINE_COMMENT",
        "C_STYLE_COMMENT",
    ]
    _is_breaking_line = re.compile("|".join(f"^{node}\\w*" for node in (_new_line_nodes)))
    _turn_off_break = re.compile("|".join(f"^{node}\\w*" for node in ("FOR_STATEMENT",)))
    _start_off_break = re.compile("|".join(f"^{node}\\w*" for node in ("LPARENTH",)))
    _on_break = re.compile("|".join(f"^{node}\\w*" for node in ("RPARENTH",)))
    _indent_in = re.compile("|".join(f"^{node}\\w*" for node in ("LBRACE",)))
    _indent_out = re.compile("|".join(f"^{node}\\w*" for node in ("RBRACE",)))

    num_terminals = len(_new_line_nodes)

    def __init__(self):
        self._need_newline: bool = False
        self._parenth_cnt: int
        self._indentation_change: int

    @staticmethod
    def get_program(nodes: List[Node], indent: str = "    ", delimeter: str = " ") -> str:
        lines_str, lines_nodes = LineBreaker.get_lines(nodes, indent, delimeter)
        return "\n".join(lines_str)

    @staticmethod
    def get_num_newline_nodes() -> int:
        return len(LineBreaker._new_line_nodes)

    def __call__(self, node: Node) -> Tuple[bool, int]:
        if not self._need_newline:
            if LineBreaker._turn_off_break.search(node.name):
                self._parenth_cnt = -1  # -1 means waiting for the first _START_OFF
            if self._parenth_cnt == -1 and LineBreaker._start_off_break.search(node.name):
                self._parenth_cnt = 1
            elif self._parenth_cnt is not None and LineBreaker._start_off_break.search(node.name):
                self._parenth_cnt += 1
            elif self._parenth_cnt is not None and LineBreaker._on_break.search(node.name):
                self._parenth_cnt -= 1

            if self._parenth_cnt == 0:
                if LineBreaker._is_breaking_line.search(node.name):
                    self._need_newline = True
                if LineBreaker._indent_in.search(node.name):
                    self._indentation_change = 1
                if LineBreaker._indent_out.search(node.name):
                    self._indentation_change = -1
            return False, 0
        elif not node.is_leaf:
            return False, 0
        else:
            indent_level_to_return = self._indentation_change if self._indentation_change is not None else 0

            self._need_newline = False
            self._parenth_cnt = 0
            self._indentation_change = 0

            return True, indent_level_to_return

    @staticmethod
    def get_lines(
        nodes: List[Node], indent: str = "    ", delimeter: str = " "
    ) -> Tuple[List[str], List[List["Node"]]]:
        line_breaker = LineBreaker()
        indents: List[int] = []
        nodes_lines: List[List["Node"]] = []

        cur_node_line: List["Node"] = []
        cur_indent_level: int = 0
        for node in nodes:
            cur_node_line.append(node)
            is_new_line, indent_change = line_breaker(node)
            if is_new_line:
                if indent_change >= 0:
                    indents.append(cur_indent_level)
                    nodes_lines.append(cur_node_line)
                    cur_indent_level += indent_change
                else:
                    assert indent_change == -1
                    cur_indent_level += indent_change
                    indents.append(cur_indent_level)
                    nodes_lines.append(cur_node_line)
                cur_node_line = []
        nodes_lines.append(cur_node_line)
        indents.append(cur_indent_level)

        return [
            f"{indent * indent_level}"
            + delimeter.join(
                node.name for node in nodes_line if node.is_leaf and node.name != TreeConstants.END_OF_CHILDREN.value
            )
            for indent_level, nodes_line in zip(indents, nodes_lines)
        ], nodes_lines

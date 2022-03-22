from typing import List


class SplitTree:
    class Node:
        def __init__(
                self,
                node_type: int,
                placeholders: List[List[int]],
                children: List["SplitTree.Node"],
        ):
            self._node_type = node_type
            self._placeholders = placeholders
            self._children = children

        @property
        def node_type(self) -> int:
            return self._node_type

        @property
        def placeholders(self) -> List[List[int]]:
            return self._placeholders

        @property
        def children(self) -> List["SplitTree.Node"]:
            return self._children

        def __str__(self) -> str:
            return "Node(node_type={0}, placeholders=[{1}], children=[{2}])".format(
                self._node_type,
                ", ".join("-".join(map(lambda x: str(x), placeholder_ids)) for placeholder_ids in self._placeholders),
                ", ".join(child.__str__() for child in self._children)
            )

    def __init__(self, root: Node):
        self._root = root

    @property
    def root(self) -> Node:
        return self._root

    def __str__(self) -> str:
        return self._root.__str__()

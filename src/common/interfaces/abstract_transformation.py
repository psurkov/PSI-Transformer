from abc import ABC, abstractmethod
from typing import List

from src.psi.psi_datapoint.tree_structures.node import Node


class Transformation(ABC):
    @abstractmethod
    def transform(self, nodes: List[Node]) -> List[Node]:
        pass

    @abstractmethod
    def inverse_transform(self, nodes: List[Node]) -> List[Node]:
        pass

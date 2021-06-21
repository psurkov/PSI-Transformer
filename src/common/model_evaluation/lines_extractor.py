from dataclasses import dataclass
from typing import List

from omegaconf import DictConfig

from src.psi_datapoint.tree_structures.node import Node


@dataclass
class LineExample:
    context_str: str
    context_nodes: List[Node]
    line_str: str
    line_nodes: List[Node]


def extract_lines(config: DictConfig, holdout: str) -> List[LineExample]:
    pass
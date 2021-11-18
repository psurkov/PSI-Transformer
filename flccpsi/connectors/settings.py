from dataclasses import dataclass

from typing import Tuple


@dataclass
class GenerationSettings:
    num_iterations: int = 10
    beam_size: int = 6
    context_len: int = -1
    n_diversity_groups: int = 1
    diversity_strength: int = 0.0
    group_top_n: int = 10
    only_full_lines: bool = False
    len_norm_base: float = 2.0
    len_norm_pow: float = 0.7

    min_prefix_dist: float = 0.2
    min_edit_dist: float = 0.0
    keep_kinds: Tuple[str, ...] = ("short", "prob")  # Also "long" is available

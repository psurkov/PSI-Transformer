import itertools
import random
from dataclasses import dataclass
from typing import Optional, List

from omegaconf import DictConfig

from tqdm import tqdm

from src.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from src.psi.psi_datapoint.tree_structures.line_breaker import LineBreaker
from src.psi.psi_datapoint.tree_structures.tree_builder import TreeBuilder


@dataclass
class LineExample:
    context_str: str
    tree_builder: TreeBuilder
    target_str: str


def extract_examples(
    json_string: str,
    facade: PSIDatapointFacade,
    prompt_part: float,
    num_examples: int,
    filter_doc: bool,
    min_length: int,
    rng: random.Random,
) -> Optional[List[LineExample]]:
    res = facade.transform(json_string, to_filter=False)
    if res is None:
        return None

    tree, _ = res
    lines, lines_nodes = LineBreaker.get_lines(tree.nodes, indent="")

    filtered_line_inds = []
    for i, (line, line_nodes) in enumerate(zip(lines, lines_nodes)):
        if len(line) < min_length:
            continue
        if filter_doc and any(node.name.startswith("DOC") for node in line_nodes):
            continue
        filtered_line_inds.append(i)

    selected_inds = rng.choices(filtered_line_inds, k=num_examples)

    examples = []
    for i in selected_inds:
        context_nodes = list(itertools.chain(*lines_nodes[:i]))
        line_nodes = lines_nodes[i]

        # Constructing string which contains context and the line
        tb = facade.get_tree_builder()
        for id_ in facade.get_tree_builder(list(itertools.chain(context_nodes, line_nodes))).ids:
            tb.add_id(id_)
        whole_context_str = LineBreaker.program(tb.tree.nodes, indent="")

        leaf_inds = [i for i, node in enumerate(line_nodes) if node.is_leaf]
        cut_ind = leaf_inds[int(prompt_part * len(leaf_inds))]
        prompt_nodes = line_nodes[:cut_ind]

        # Constructing string which contains context and the prompt
        ids = facade.get_tree_builder(list(itertools.chain(context_nodes, prompt_nodes))).ids
        tb = facade.get_tree_builder()
        for id_ in ids:
            tb.add_id(id_)
        context_str = LineBreaker.program(tb.tree.nodes, indent="")

        target_str = whole_context_str[len(context_str) :]
        examples.append(LineExample(context_str, tb, target_str))

    return examples


def extract_lines(
    config: DictConfig, holdout: str, prompt_part: float, examples_amount: int, seed: int = 42
) -> List[LineExample]:
    assert 0.0 <= prompt_part <= 1.0, f"Invalid prompt part: {prompt_part}. Must be between 0.0 and 1.0"

    facade = PSIDatapointFacade(config, diff_warning=False)
    if holdout == "mock":
        data_path = config.source_data.mock
    elif holdout == "train":
        data_path = config.source_data.train
    elif holdout == "val":
        data_path = config.source_data.val
    elif holdout == "test":
        data_path = config.source_data.test
    else:
        raise ValueError(f"Invalid holdout {holdout}")

    with open(data_path) as f:
        json_strings = f.readlines()

    rng = random.Random(seed)
    json_strings = rng.choices(json_strings, k=examples_amount)

    total_examples = []
    for json_string in tqdm(json_strings, desc=f"Extracting examples ({holdout} with prompt {prompt_part})"):
        try:
            examples = extract_examples(
                json_string, facade, prompt_part, num_examples=1, filter_doc=True, min_length=5, rng=rng
            )
        except (KeyError, IndexError) as e:
            print(f"Failed to extract examples {e}")
            continue
        if not examples:
            continue
        total_examples.extend(examples)

    return total_examples

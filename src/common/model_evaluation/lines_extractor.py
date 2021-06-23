from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Optional, Iterable, List

from omegaconf import DictConfig

from tqdm import tqdm

from src.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade
from src.psi.psi_datapoint.tree_structures.line_breaker import LineBreaker
from src.psi.psi_datapoint.tree_structures.tree_builder import TreeBuilder


@dataclass
class LineExample:
    context_str: str
    context_tree_builder: TreeBuilder
    target_str: str


def extract_example(json_string: str, config: DictConfig, prompt_part: float) -> List[Optional[LineExample]]:
    examples = []

    facade = PSIDatapointFacade(config)
    res = facade.transform(json_string, to_filter=False)
    if res is None:
        examples.append(None)
    else:
        tree, _ = res
        lines, lines_nodes = LineBreaker.get_lines(tree.nodes, indent="")

        context_nodes = []
        for line, line_nodes in zip(lines, lines_nodes):
            leaf_inds = [i for i, node in enumerate(line_nodes) if node.is_leaf]
            cut_ind = leaf_inds[int(prompt_part * len(leaf_inds))]
            prompt_nodes = line_nodes[:cut_ind]

            example_context_nodes = context_nodes + prompt_nodes
            tree_builder = facade.get_tree_builder()
            for id_ in facade.get_tree_builder(example_context_nodes).ids:
                tree_builder.add_id(id_)

            example_context_str = LineBreaker.program(tree_builder.tree.nodes, indent="")

            context_nodes.extend(line_nodes)
            tb = facade.get_tree_builder()
            for id_ in facade.get_tree_builder(context_nodes).ids:
                tb.add_id(id_)
            whole_context_str = LineBreaker.program(tb.tree.nodes, indent="")

            target_str = whole_context_str[len(example_context_str) :]

            examples.append(LineExample(example_context_str, tree_builder, target_str))

    return examples


def extract_lines(config: DictConfig, holdout: str, prompt_part: float) -> List[LineExample]:
    assert 0.0 <= prompt_part <= 1.0, f"Invalid prompt part: {prompt_part}. Must be between 0.0 and 1.0"

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

    with Pool(cpu_count()) as pool:
        line_examples_lists = pool.starmap(
            extract_example,
            tqdm([(json_string, config, prompt_part) for json_string in json_strings], desc="Extracting line examples"),
        )

    return [example for line_examples in line_examples_lists for example in line_examples]

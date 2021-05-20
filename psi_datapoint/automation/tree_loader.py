import os
from typing import List, Optional, Tuple

import tqdm
from omegaconf import OmegaConf

from psi_datapoint.stateful.tokenizer import TreeTokenizer, TreeBuilder
from psi_datapoint.stateful.stats_collector import StatsCollector
from psi_datapoint.stateless_transformations.abstract_transformation import Transformation
from psi_datapoint.stateless_transformations.whitespace_normalizer import WhitespaceNormalizer
from psi_datapoint.tree_structures.node import Node
from psi_datapoint.tree_structures.tree import Tree


TRANSFORMATIONS = {  # Order in the dict must be preserved
    "whitespace_normalize": WhitespaceNormalizer,
}


class TreeLoader:
    def __init__(self, config_path: Optional[str] = "psi_datapoint/automation/config.yaml"):
        self._config = OmegaConf.load(config_path).psi_pretraining

        self._transformations: List[Transformation] = []
        for transform_name, transform_cls in TRANSFORMATIONS.items():
            if transform_name in self._config.transformations:
                self._transformations.append(transform_cls())

        self._overwrite = self._config.overwrite
        if self._overwrite:
            self._trained = False
            self._stats = None
            self._tokenizer = None
        else:
            pretrained_path = self._config.pretrained_path
            if not StatsCollector.pretrained_exists(pretrained_path) or not TreeTokenizer.pretrained_exists(
                pretrained_path
            ):
                assert not StatsCollector.pretrained_exists(pretrained_path) and not TreeTokenizer.pretrained_exists(
                    pretrained_path
                )
                self._trained = False
                self._stats = None
                self._tokenizer = None
            else:
                self._trained = True
                self._stats = StatsCollector.from_pretrained(pretrained_path)
                self._tokenizer = TreeTokenizer.from_pretrained(pretrained_path)

    def _apply_transformations(self, nodes: List[Node]) -> List[Node]:
        for transformation in self._transformations:
            nodes = transformation.transform(nodes)
        return nodes

    def _inverse_apply_transformations(self, nodes: List[Node]) -> List[Node]:
        for transformation in reversed(self._transformations):
            nodes = transformation.inverse_transform(nodes)
        return nodes

    def train(self) -> None:
        if self._trained:
            assert self._overwrite

        with open(self._config.train_jsonl_path, "r") as f:
            json_strings = f.readlines()

        nodes_lists = []
        for json_string in tqdm.tqdm(json_strings, desc="Parsing trees..."):
            nodes = Node.load_psi_miner_nodes(json_string)
            if nodes is not None:
                nodes_lists.append(nodes)

        transformed_nodes_lists = [
            self._apply_transformations(nodes) for nodes in tqdm.tqdm(nodes_lists, desc="Applying transformations...")
        ]
        self._stats = StatsCollector()
        transformed_nodes_lists = self._stats.train(transformed_nodes_lists)
        self._stats.save_pretrained(self._config.pretrained_path)

        trees = [Tree(nodes_list, self._stats) for nodes_list in transformed_nodes_lists]

        print(
            f"Trees was compressed to "
            f"{sum(tree.compressed_size / tree.size for tree in trees) / len(trees) * 100}% "
            f"of its size in average"
        )

        self._tokenizer = TreeTokenizer(
            self._config.tokenizer.vocab_size, self._config.tokenizer.min_frequency, self._config.tokenizer.dropout
        )
        self._tokenizer.train(trees)
        self._tokenizer.save_pretrained(self._config.pretrained_path)

        self._trained = True

    def transform(self, json_string: str) -> Tuple[Tree, List[int]]:
        assert self._trained
        nodes = Node.load_psi_miner_nodes(json_string)
        transformed_nodes = self._apply_transformations(nodes)
        transformed_nodes = self._stats.transform(transformed_nodes)
        tree = Tree(transformed_nodes, self._stats)
        ids = self._tokenizer.encode(tree)
        return tree, ids

    def get_tree_builder(self, tree: Optional[Tree] = None) -> TreeBuilder:
        return TreeBuilder(tree if tree else Tree([], self._stats), self._tokenizer)

    def inverse_transform(self, tree: Tree) -> List[Node]:
        return self._inverse_apply_transformations(tree.nodes)


if __name__ == "__main__":

    def main():
        tree_loader = TreeLoader()
        tree_loader.train()
        with open("/Users/Yaroslav.Sokolov/work/psiminer/mock_data/mock_data.train.jsonl") as f:
            [json_string] = f.readlines()

        tree, ids = tree_loader.transform(json_string)
        # print(tree.tree_representation)
        # print(tree.program)
        orig_nodes = tree_loader.inverse_transform(tree)
        print(orig_nodes[0].tree_representation)
        print(orig_nodes[0].program)
        print(f"Tree size: {tree.size}, compressed_size: {tree.compressed_size}, ids amount: {len(ids)}")

        tree_builder = tree_loader.get_tree_builder()
        for id_ in ids:
            # name = tree_builder._tokenizer.bpe_tokenizer.id_to_token(id_)
            # assert id_ in tree_builder.get_next_possible_ids()
            tree_builder.add_id(id_)

        orig_nodes = tree_loader.inverse_transform(tree_builder.tree)
        print(orig_nodes[0].program)

    main()

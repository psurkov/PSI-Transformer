import difflib
import json
import os
from json import JSONDecodeError
from typing import List, Optional, Tuple, Union

import numpy as np
import tqdm
from omegaconf import OmegaConf, DictConfig

from psi_datapoint.stateful.tokenizer import TreeTokenizer, TreeBuilder
from psi_datapoint.stateful.stats_collector import StatsCollector
from psi_datapoint.stateless_transformations.whitespace_normalizer import WhitespaceNormalizer
from psi_datapoint.tree_structures.node import Node
from psi_datapoint.tree_structures.tree import Tree


TRANSFORMATIONS = [  # Order in the dict must be preserved
    ("whitespace_normalize", WhitespaceNormalizer),
]


class PSIDatapointFacade:
    _stats_filename = "dataset_stats.json"
    _config_filename = "config.yaml"

    def __init__(self, config: DictConfig):
        self._config = config

        self._overwrite = self._config.psi_pretraining.overwrite
        pretrained_path = self._config.save_path
        if PSIDatapointFacade.pretrained_exists(pretrained_path):
            self._trained = True
            self._stats_collector = StatsCollector.from_pretrained(pretrained_path)
            self._tokenizer = TreeTokenizer.from_pretrained(pretrained_path)
            with open(os.path.join(pretrained_path, PSIDatapointFacade._stats_filename)) as f:
                self._stats = json.load(f)

            config = OmegaConf.load(os.path.join(pretrained_path, PSIDatapointFacade._config_filename))
            if self._config != config:
                print(f"WARNING:\nLoaded config doesn't match current config! Diff:")
                for text in difflib.unified_diff(
                    OmegaConf.to_yaml(self._config).split("\n"), OmegaConf.to_yaml(config).split("\n")
                ):
                    if text[:3] not in ("+++", "---", "@@ "):
                        print(f"    {text}")
        else:
            self._trained = False
            self._stats_collector = None
            self._tokenizer = None
            self._stats = {}

    def _save_pretrained(self, path: str) -> None:
        stats_path = os.path.join(path, PSIDatapointFacade._stats_filename)
        config_path = os.path.join(path, PSIDatapointFacade._config_filename)
        with open(stats_path, "w") as f:
            json.dump(self._stats, f)
        OmegaConf.save(self._config, config_path)
        self._stats_collector.save_pretrained(self._config.save_path)
        self._tokenizer.save_pretrained(self._config.save_path)

    @staticmethod
    def pretrained_exists(path: str) -> bool:
        stats_exists = os.path.exists(os.path.join(path, PSIDatapointFacade._stats_filename))
        config_exists = os.path.exists(os.path.join(path, PSIDatapointFacade._config_filename))
        return (
            StatsCollector.pretrained_exists(path)
            and TreeTokenizer.pretrained_exists(path)
            and stats_exists
            and config_exists
        )

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def tokenizer(self) -> TreeTokenizer:
        return self._tokenizer

    def get_tokenized_sizes(self) -> List[int]:
        return self._stats["tree_tokenized_sizes"]

    def _apply_transformations(self, nodes: List[Node]) -> List[Node]:
        for transform_name, transform_cls in TRANSFORMATIONS:
            if transform_name in self._config.psi_pretraining.transformations:
                nodes = transform_cls().transform(nodes)
        return nodes

    def _inverse_apply_transformations(self, nodes: List[Node]) -> List[Node]:
        for transform_name, transform_cls in reversed(TRANSFORMATIONS):
            if transform_name in self._config.psi_pretraining.transformations:
                nodes = transform_cls().inverse_transform(nodes)
        return nodes

    def train(self) -> "PSIDatapointFacade":
        if self._trained:
            assert self._overwrite

        if not os.path.exists(self._config.save_path):
            os.mkdir(self._config.save_path)

        # stats calculation
        lines_amount = 0
        nodes_amount_list = []
        with open(self._config.source_data.train_jsonl, "r") as f:
            for line in tqdm.tqdm(f, desc="Calculating stats of jsonl..."):
                lines_amount += 1
                try:
                    nodes_amount_list.append(len(json.loads(line)["AST"]))
                except JSONDecodeError:
                    nodes_amount_list.append(np.iinfo(np.int32).max)
        nodes_amount_perc = np.percentile(
            np.array(nodes_amount_list, dtype=np.int32), self._config.psi_pretraining.max_percentile
        )
        jsonl_mask = [bool(amount <= nodes_amount_perc) for amount in nodes_amount_list]
        self._stats["jsonl_mask"] = jsonl_mask
        self._stats["nodes_amount_list"] = nodes_amount_list
        self._stats["nodes_amount_perc"] = nodes_amount_perc
        self._stats["trees_amount"] = lines_amount

        # creating nodes
        with open(self._config.source_data.train_jsonl, "r") as f:
            nodes_lists = []
            for json_string, is_ok in tqdm.tqdm(zip(f, jsonl_mask), desc="Parsing trees...", total=lines_amount):
                if is_ok:
                    try:
                        json_dict = json.loads(json_string)
                    except JSONDecodeError:
                        logging.warning("Skipping tree...")
                        continue
                    nodes = Node.load_psi_miner_nodes(json_dict)
                    if nodes is not None:
                        nodes_lists.append(nodes)
        # transforming trees
        transformed_nodes_lists = [
            self._apply_transformations(nodes) for nodes in tqdm.tqdm(nodes_lists, desc="Applying transformations...")
        ]
        # training stats collector
        self._stats_collector = StatsCollector()
        transformed_nodes_lists = self._stats_collector.train(transformed_nodes_lists)
        # creating trees
        trees = [Tree(nodes_list, self._stats_collector) for nodes_list in transformed_nodes_lists]
        tree_compressed_sizes = [tree.compressed_size for tree in trees]
        compress_ratios = [compressed_size / tree.size for tree, compressed_size in zip(trees, tree_compressed_sizes)]
        print(
            f"Trees was compressed to " f"{sum(compress_ratios) / len(trees) * 100}% " f"of its size in average"
        )
        self._stats["tree_compressed_sizes"] = tree_compressed_sizes

        # training tokenizer
        self._tokenizer = TreeTokenizer(
            self._config.tokenizer.vocab_size, self._config.tokenizer.min_frequency, self._config.tokenizer.dropout
        )
        self._tokenizer.train(trees)
        tree_tokenized_sizes = [
            len(self._tokenizer.encode(tree))
            for tree in tqdm.tqdm(trees, desc="Collecting stats about tokenized trees...")
        ]
        self._stats["tree_tokenized_sizes"] = tree_tokenized_sizes

        self._trained = True
        self._save_pretrained(self._config.save_path)

        return self

    def transform(self, json_string: str, to_filter: bool = False) -> Optional[Tuple[Tree, List[int]]]:
        assert self._trained
        try:
            json_dict = json.loads(json_string)
        except JSONDecodeError:
            return None
        if to_filter and len(json_dict["AST"]) > self._stats["nodes_amount_perc"]:
            return None

        nodes = Node.load_psi_miner_nodes(json_dict)
        transformed_nodes = self._apply_transformations(nodes)
        transformed_nodes = self._stats_collector.transform(transformed_nodes)
        tree = Tree(transformed_nodes, self._stats_collector)
        ids = self._tokenizer.encode(tree)
        return tree, ids

    def get_tree_builder(self, tree: Optional[Tree] = None) -> TreeBuilder:
        assert self._trained
        return TreeBuilder(tree if tree else Tree([], self._stats_collector), self._tokenizer)

    def inverse_transform(self, tree: Tree) -> List[Node]:
        assert self._trained
        return self._inverse_apply_transformations(tree.nodes)

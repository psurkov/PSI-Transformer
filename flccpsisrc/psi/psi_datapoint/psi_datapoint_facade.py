import difflib
import json
import os
from json import JSONDecodeError
from typing import List, Optional, Union

import numpy as np
import tqdm
import youtokentome as yttm
from omegaconf import OmegaConf, DictConfig

from flccpsisrc.psi.psi_datapoint.tree_structures.special_ids import SpecialIds, SPECIAL_IDS_RESERVED_SIZE
from flccpsisrc.psi.psi_datapoint.tree_structures.split_tree import SplitTree
from flccpsisrc.psi.psi_datapoint.tree_structures.split_tree_builder import SplitTreeBuilder
from flccpsisrc.psi.psi_datapoint.tree_structures.structure_decompression import StructureDecompression


class PSIDatapointFacade:
    _stats_filename = "psi/dataset_stats.json"
    _config_filename = "config.yaml"
    _placeholder_bpe_filename = "placeholders.bpe"
    _structure_compression_data_filename = "structureTreeCompressionData.json"
    _type_coder_data_filename = "typeCoderData.json"

    def __init__(self, config: DictConfig, diff_warning: bool = True):
        self._config = config

        self._overwrite = self._config.psi_pretraining.overwrite
        pretrained_path = self._config.save_path
        self._placeholders_bpe = yttm.BPE(
            model=os.path.join(pretrained_path, PSIDatapointFacade._placeholder_bpe_filename)
        )
        self._structure_decompression = StructureDecompression(
            os.path.join(
                pretrained_path,
                PSIDatapointFacade._type_coder_data_filename,
            ),
            os.path.join(
                pretrained_path,
                PSIDatapointFacade._structure_compression_data_filename
            )
        )
        if PSIDatapointFacade.pretrained_exists(pretrained_path):
            self._trained = True
            with open(os.path.join(pretrained_path, PSIDatapointFacade._stats_filename)) as f:
                self._stats = json.load(f)

            config = OmegaConf.load(os.path.join(pretrained_path, PSIDatapointFacade._config_filename))
            if diff_warning and self._config != config:
                print(f"WARNING:\nLoaded config doesn't match current config! Diff:")
                for text in difflib.unified_diff(
                        OmegaConf.to_yaml(config).split("\n"), OmegaConf.to_yaml(self._config).split("\n")
                ):
                    if text[:3] not in ("+++", "---", "@@ "):
                        print(f"    {text}")
        else:
            self._trained = False
            self._stats = {}

    def _save_pretrained(self, path: str) -> None:
        stats_path = os.path.join(path, PSIDatapointFacade._stats_filename)
        config_path = os.path.join(path, PSIDatapointFacade._config_filename)
        with open(stats_path, "w") as f:
            json.dump(self._stats, f)
        OmegaConf.save(self._config, config_path)

    @staticmethod
    def pretrained_exists(path: str) -> bool:
        stats_exists = os.path.exists(os.path.join(path, PSIDatapointFacade._stats_filename))
        config_exists = os.path.exists(os.path.join(path, PSIDatapointFacade._config_filename))
        return stats_exists and config_exists

    @property
    def is_trained(self) -> bool:
        return self._trained

    def get_tokenized_sizes(self, holdout: str) -> List[int]:
        return self._stats[f"tree_tokenized_sizes_{holdout}"]

    def train(self) -> "PSIDatapointFacade":
        if self._trained:
            assert self._overwrite

        save_path = os.path.join(self._config.save_path, "psi")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # stats calculation
        trees_amount = 0
        nodes_amount_list = []
        with open(self._config.source_data.train, "r") as f:
            for line in tqdm.tqdm(f, desc="Calculating stats of jsonl..."):
                trees_amount += 1
                try:
                    nodes_amount_list.append(self._subtree_nodes(json.loads(line)["root"]))
                except JSONDecodeError:
                    nodes_amount_list.append(np.iinfo(np.int32).max)
        nodes_amount_perc = np.percentile(
            np.array(nodes_amount_list, dtype=np.int32), self._config.psi_pretraining.max_percentile
        )
        jsonl_mask = [bool(amount <= nodes_amount_perc) for amount in nodes_amount_list]
        self._stats["nodes_amount_perc"] = nodes_amount_perc
        skipped_nodes_count = sum(amount for is_ok, amount in zip(jsonl_mask, nodes_amount_list) if not is_ok)

        # creating nodes
        bar = tqdm.tqdm(total=sum(nodes_amount_list) - skipped_nodes_count, desc="Parsing trees...")
        skipped_trees_count = (100 - self._config.psi_pretraining.max_percentile) * 0.01 * trees_amount
        split_trees_ids_len = []
        with open(self._config.source_data.train, "r") as f:
            for json_string, is_ok in zip(f, jsonl_mask):
                if is_ok:
                    json_dict = json.loads(json_string)
                    split_tree = self.json_dict_to_split_tree(json_dict, to_filter=True)
                    if split_tree is None:
                        skipped_trees_count += 1
                        continue
                    else:
                        split_trees_ids_len.append(len(self.encode_split_tree_to_ids(split_tree)))
                    bar.update(self._subtree_nodes(json_dict["root"]))
                else:
                    skipped_trees_count += 1
        bar.close()
        print(f"Skipped {int(skipped_trees_count)} trees!")

        self._trained = True

        tree_tokenized_sizes = [
            ids_len for ids_len in
            tqdm.tqdm(split_trees_ids_len, desc="Collecting stats about tokenized trees train...")
        ]
        self._stats["tree_tokenized_sizes_train"] = tree_tokenized_sizes
        self._stats["tree_tokenized_sizes_val"] = self._count_tokenized_sizes(self._config.source_data.val)
        self._stats["tree_tokenized_sizes_test"] = self._count_tokenized_sizes(self._config.source_data.test)

        self._save_pretrained(self._config.save_path)

        return self

    def _count_tokenized_sizes(self, path: str) -> List[int]:
        sizes = []
        with open(path) as f:
            for json_string in tqdm.tqdm(f, desc=f"Collecting stats about tokenized trees val/test..."):
                split_tree = self.json_dict_to_split_tree(json_string, to_filter=True)
                if split_tree is not None:
                    sizes.append(len(self.encode_split_tree_to_ids(split_tree)))
        return sizes

    def json_dict_to_split_tree(self, json_tree: Union[str, dict], to_filter: bool = False) -> Optional[SplitTree]:
        if isinstance(json_tree, str):
            try:
                json_dict = json.loads(json_tree)
            except JSONDecodeError:
                return None
        else:
            json_dict = json_tree
        if to_filter and self._subtree_nodes(json_dict["root"]) >= self._stats["nodes_amount_perc"]:
            return None

        def from_subtree(cur_dict: dict) -> SplitTree.Node:
            return SplitTree.Node(
                cur_dict["nodeTypeId"],
                cur_dict["placeholders"],
                [from_subtree(child_dict) for child_dict in cur_dict["children"]],
            )

        return SplitTree(from_subtree(json_dict["root"]))

    def encode_split_tree_to_ids(self, tree: SplitTree) -> List[int]:
        res = []

        def encode_subtree(cur: SplitTree.Node) -> None:
            res.append(cur.node_type + SPECIAL_IDS_RESERVED_SIZE)

            for index, placeholder_ids in enumerate(cur.placeholders):
                res.extend(
                    [placeholder_id + SPECIAL_IDS_RESERVED_SIZE + self._structure_decompression.vocab_size
                     for placeholder_id in placeholder_ids]
                )
                res.append(SpecialIds.END_OF_PLACEHOLDER.value)

            for child in cur.children:
                encode_subtree(child)
            res.append(SpecialIds.END_OF_NODE_CHILDREN.value)

        encode_subtree(tree.root)
        return res

    def get_split_tree_builder(self, init_tree: SplitTree):
        return SplitTreeBuilder(self._structure_decompression, self._placeholders_bpe)

    @property
    def vocab_size(self):
        return SPECIAL_IDS_RESERVED_SIZE + self._structure_decompression.vocab_size + self._placeholders_bpe.vocab_size()

    @staticmethod
    def _subtree_nodes(json_dict: dict) -> int:
        return 1 + sum(map(lambda t: PSIDatapointFacade._subtree_nodes(t), json_dict["children"]))

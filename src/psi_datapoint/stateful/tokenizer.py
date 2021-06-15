import itertools
import json
import os
from typing import List, Optional, Iterable, Union, Tuple

# from any_case import to_snake_case
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
import tqdm

from src.psi_datapoint.stateful.abstract_stateful import Stateful
from src.psi_datapoint.tree_structures.node import TreeConstants
from src.psi_datapoint.tree_structures.tree import Tree


class TreeTokenizer(Stateful):
    _filename = "psi/tree_tokenizer"

    def __init__(
        self,
        vocab_size: int,
        min_frequency: int,
        dropout: Optional[float],
        # snake_camel_case_pretok: bool = False,
    ) -> None:
        self._vocab_size = vocab_size
        self._min_frequency = min_frequency
        self._dropout = dropout if dropout is not None and 0.0 < dropout < 1.0 else None
        # self._snake_camel_case_pretok = snake_camel_case_pretok

        self._bpe_tokenizer = None
        self._special_vocab = None
        self._special_vocab_inversed = None
        self._arbitrary_end_ind = None
        self._non_leaf_start_ind = None
        self._eov_id = None

    def save_pretrained(self, path: str) -> None:
        path = os.path.join(path, TreeTokenizer._filename)
        if not os.path.exists(path):
            os.mkdir(path)
        self._bpe_tokenizer.save(os.path.join(path, "bpe_tokenizer.json"))
        with open(os.path.join(path, "tokenizer_stuff.json"), "w") as f:
            json.dump(
                [
                    self._vocab_size,
                    self._min_frequency,
                    self._dropout,
                    self._arbitrary_end_ind,
                    self._non_leaf_start_ind,
                    self._eov_id,
                ],
                f,
            )

    @staticmethod
    def from_pretrained(path: str) -> "TreeTokenizer":
        path = os.path.join(path, TreeTokenizer._filename)
        with open(os.path.join(path, "tokenizer_stuff.json")) as f:
            [vocab_size, min_frequency, dropout, arbitrary_end_ind, non_leaf_start_ind, eov_id] = json.load(f)
        tokenizer = TreeTokenizer(vocab_size, min_frequency, dropout)
        tokenizer._arbitrary_end_ind = arbitrary_end_ind
        tokenizer._non_leaf_start_ind = non_leaf_start_ind
        tokenizer._eov_id = eov_id
        tokenizer._bpe_tokenizer = Tokenizer.from_file(os.path.join(path, "bpe_tokenizer.json"))
        return tokenizer

    @staticmethod
    def pretrained_exists(path: str) -> bool:
        return os.path.exists(os.path.join(path, TreeTokenizer._filename))

    @property
    def eov_id(self) -> int:
        return self._eov_id

    @property
    def bpe_tokenizer(self) -> Tokenizer:
        return self._bpe_tokenizer

    @property
    def arbitrary_ids(self) -> Iterable[int]:
        return range(self._arbitrary_end_ind)

    def encode_non_arbitrary(self, token: str) -> int:
        return self._special_vocab[token]

    def decode_non_arbitrary_token(self, id_: int) -> str:
        return self._special_vocab_inversed[id_]

    def classify_ids(
        self, ids: Union[int, List[int], torch.Tensor]
    ) -> Union[
        Tuple[bool, bool, bool],
        Tuple[List[bool], List[bool], List[bool]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Returns 3 bools: is_arbitrary, is_non_arbitrary_leaf, is_internal_node"""
        if isinstance(ids, list):
            zip(*(self.classify_ids(id_) for id_ in ids))
        else:
            return (
                ids <= self._arbitrary_end_ind,
                self._arbitrary_end_ind < ids < self._non_leaf_start_ind,
                self._non_leaf_start_ind <= ids,
            )

    def train(self, trees: List[Tree]) -> None:
        non_leaf_tokens = list(
            (
                set(
                    node.name
                    for tree in tqdm.tqdm(trees, desc="Collecting nodes tokens for tokenizer 1/2...")
                    for node in tree.nodes
                    if not node.is_leaf and node.is_visible
                )
            )
        )
        non_arbitrary_leaf_tokens = list(
            set(
                node.name
                for tree in tqdm.tqdm(trees, desc="Collecting nodes tokens for tokenizer 2/2...")
                for node in tree.nodes
                if node.is_leaf and not node.is_arbitrary and node.is_visible
            )
        )
        special_tokens_amount = len(non_leaf_tokens) + len(non_arbitrary_leaf_tokens)
        print(f"There are {special_tokens_amount} special tokens out of {self._vocab_size} vocabulary")

        bpe_vocab_size = self._vocab_size - special_tokens_amount
        bpe_special_tokens = ["[UNK]", "[PAD]", "[EOV]"]
        self._eov_id = len(bpe_special_tokens) - 1

        tokenizer = Tokenizer(BPE(dropout=self._dropout, unk_token="[UNK]", fuse_unk=True))
        tokenizer.pre_tokenizer = Metaspace()
        trainer = BpeTrainer(
            vocab_size=bpe_vocab_size, min_frequency=self._min_frequency, special_tokens=bpe_special_tokens
        )
        bpe_nodes_iterator = (
            node.name for tree in trees for node in tree.nodes if node.is_arbitrary and node.is_visible
        )
        print("Training tokenizer...")
        tokenizer.train_from_iterator(bpe_nodes_iterator, trainer)
        tokenizer.post_processor = TemplateProcessing(
            single="$A [EOV]",
            pair="$A [EOV] $B:1 [EOV]:1",
            special_tokens=[("[EOV]", tokenizer.token_to_id("[EOV]"))],
        )
        print(f"The BPE vocabulary size is {tokenizer.get_vocab_size()}")
        self._bpe_tokenizer = tokenizer

        self._special_vocab = {
            token: i
            for i, token in enumerate(
                itertools.chain(non_arbitrary_leaf_tokens, non_leaf_tokens), start=tokenizer.get_vocab_size()
            )
        }
        self._special_vocab_inversed = {
            i: token
            for i, token in enumerate(
                itertools.chain(non_arbitrary_leaf_tokens, non_leaf_tokens), start=tokenizer.get_vocab_size()
            )
        }
        self._arbitrary_end_ind = tokenizer.get_vocab_size() - 1
        self._non_leaf_start_ind = tokenizer.get_vocab_size() + len(non_arbitrary_leaf_tokens) + 1

    def encode(self, tree: Tree) -> List[int]:
        node_ids = []
        for node in tree.nodes:
            if node.is_visible:
                if node.is_arbitrary:
                    node_ids.extend(self._bpe_tokenizer.encode(node.name).ids)
                else:
                    node_ids.append(self._special_vocab[node.name])
        return node_ids

    def decode_string(self, ids: List[int]) -> str:
        return self.bpe_tokenizer.decode(ids).replace(" ", "").replace("▁", " ")[1:]

    # @staticmethod
    # def _parse_any_case(word: str) -> str:
    #     return to_snake_case(word, sep_numbers=True).replace("_", " ")


class TreeBuilder:
    def __init__(self, tree: Tree, tokenizer: TreeTokenizer):
        self._tree = tree
        self._tokenizer = tokenizer

        self._cur_arbitrary_ids = []

    @property
    def tree(self) -> Tree:
        return self._tree

    def get_next_possible_ids(self) -> List[int]:
        if self._cur_arbitrary_ids:
            return list(self._tokenizer.arbitrary_ids)
        else:
            ret = []
            next_suitable_nodes = self._tree.get_next_suitable_nodes()
            for node_name in next_suitable_nodes:
                if node_name == TreeConstants.ARBITRARY_REPR.value:
                    ret.extend(self._tokenizer.arbitrary_ids)
                else:
                    ret.append(self._tokenizer.encode_non_arbitrary(node_name))
            return ret

    def add_id(self, id_: int) -> None:
        is_arbitrary, _, _ = self._tokenizer.classify_ids(id_)

        if id_ == self._tokenizer.eov_id:
            token = self._tokenizer.decode_string(self._cur_arbitrary_ids)
            self._cur_arbitrary_ids = []
            self._tree.add_node(name=token, is_arbitrary=True)
        elif is_arbitrary:
            self._cur_arbitrary_ids.append(id_)
        else:
            token = self._tokenizer.decode_non_arbitrary_token(id_)
            self._tree.add_node(token, is_arbitrary=False)

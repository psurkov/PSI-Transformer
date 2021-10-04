import itertools
import json
import os
from typing import List, Optional, Iterable, Union, Tuple, Dict

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
import tqdm

from src.common.interfaces.abstract_stateful import Stateful
from src.psi.psi_datapoint.tree_structures.tree import Tree


class TreeTokenizer(Stateful):
    _filename = "psi/tree_tokenizer"
    _special_unk = "[UNK_SPECIAL]"

    def __init__(
        self,
        vocab_size: int,
        min_frequency: int,
        dropout: Optional[float],
    ) -> None:
        self._vocab_size: int = vocab_size
        self._min_frequency: int = min_frequency
        self._dropout: Optional[float] = dropout if dropout is not None and 0.0 < dropout < 1.0 else None

        self._bpe_tokenizer: Tokenizer
        self._special_vocab: Dict[str, int]
        self._special_vocab_inversed: Dict[int, str]
        self._arbitrary_end_ind: int
        self._non_leaf_start_ind: int
        self._eov_id: int

    def save_pretrained(self, path: str) -> None:
        path = os.path.join(path, TreeTokenizer._filename)
        os.makedirs(path, exist_ok=True)
        self._bpe_tokenizer.save(os.path.join(path, "bpe_tokenizer.json"))
        with open(os.path.join(path, "tokenizer_stuff.json"), "w") as f:
            json.dump(
                [
                    self._special_vocab,
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
            [
                special_vocab,
                vocab_size,
                min_frequency,
                dropout,
                arbitrary_end_ind,
                non_leaf_start_ind,
                eov_id,
            ] = json.load(f)
        tokenizer = TreeTokenizer(vocab_size, min_frequency, dropout)
        tokenizer._special_vocab = special_vocab
        tokenizer._special_vocab_inversed = {v: k for k, v in special_vocab.items()}
        tokenizer._arbitrary_end_ind = arbitrary_end_ind
        tokenizer._non_leaf_start_ind = non_leaf_start_ind
        tokenizer._eov_id = eov_id
        tokenizer._bpe_tokenizer = Tokenizer.from_file(os.path.join(path, "bpe_tokenizer.json"))
        return tokenizer

    @staticmethod
    def pretrained_exists(path: str) -> bool:
        return os.path.exists(os.path.join(path, TreeTokenizer._filename))

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eov_id(self) -> int:
        return self._eov_id

    @property
    def arbitrary_ids(self) -> Iterable[int]:
        return range(self.eov_id + 1, self._arbitrary_end_ind + 1)

    def encode_non_arbitrary_token(self, token: str) -> int:
        return self._special_vocab.get(token, self._special_vocab[self._special_unk])

    def encode_tree(self, tree: Tree) -> List[int]:
        node_ids = []
        for node in tree.nodes:
            if node.is_visible:
                if node.is_arbitrary:
                    node_ids.extend(self._bpe_tokenizer.encode(node.name).ids)
                else:
                    node_ids.append(self.encode_non_arbitrary_token(node.name))
        return node_ids

    def decode(self, id_: int) -> str:
        return (
            self._special_vocab_inversed[id_] if id_ > self._arbitrary_end_ind else self._bpe_tokenizer.id_to_token(id_)
        )

    def decode_arbitrary_string(self, ids: List[int]) -> str:
        assert all(id_ <= self._arbitrary_end_ind for id_ in ids), f""
        return self._bpe_tokenizer.decode(ids).replace(" ", "").replace("▁", " ")[1:]

    def classify_ids(
        self, ids: Union[int, torch.Tensor]
    ) -> Union[
        Tuple[bool, bool, bool],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Returns 3 bools: is_arbitrary, is_non_arbitrary_leaf, is_internal_node"""
        if isinstance(ids, int):
            return (
                ids <= self._arbitrary_end_ind,
                self._arbitrary_end_ind < ids < self._non_leaf_start_ind,
                self._non_leaf_start_ind <= ids,
            )
        elif isinstance(ids, torch.Tensor):
            return (
                ids <= self._arbitrary_end_ind,
                torch.logical_and(self._arbitrary_end_ind < ids, ids < self._non_leaf_start_ind),
                self._non_leaf_start_ind <= ids,
            )
        else:
            raise TypeError(str(type(ids)))

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
        special_tokens_amount = len(non_leaf_tokens) + len(non_arbitrary_leaf_tokens) + 1  # special unk
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
        self._vocab_size = tokenizer.get_vocab_size() + special_tokens_amount
        self._bpe_tokenizer = tokenizer

        self._special_vocab = {
            token: i
            for i, token in enumerate(
                itertools.chain(non_arbitrary_leaf_tokens, non_leaf_tokens), start=tokenizer.get_vocab_size() + 1
            )
        }
        self._special_vocab[self._special_unk] = tokenizer.get_vocab_size()
        self._special_vocab_inversed = {token: i for i, token in self._special_vocab.items()}

        self._arbitrary_end_ind = tokenizer.get_vocab_size() - 1
        self._non_leaf_start_ind = tokenizer.get_vocab_size() + len(non_arbitrary_leaf_tokens) + 1

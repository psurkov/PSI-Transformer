import json
import os
from typing import List, Optional

# from any_case import to_snake_case
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
        self._leaf_start_ind = None
        self._arbitrary_start_ind = None
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
                    self._leaf_start_ind,
                    self._arbitrary_start_ind,
                    self._eov_id,
                ],
                f,
            )

    @staticmethod
    def from_pretrained(path: str) -> "TreeTokenizer":
        path = os.path.join(path, TreeTokenizer._filename)
        with open(os.path.join(path, "tokenizer_stuff.json")) as f:
            [vocab_size, min_frequency, dropout, leaf_start_ind, arbitrary_start_ind, eov_id] = json.load(f)
        tokenizer = TreeTokenizer(vocab_size, min_frequency, dropout)
        tokenizer._leaf_start_ind = leaf_start_ind
        tokenizer._arbitrary_start_ind = arbitrary_start_ind
        tokenizer._eov_id = eov_id
        tokenizer._bpe_tokenizer = Tokenizer.from_file(os.path.join(path, "bpe_tokenizer.json"))
        return tokenizer

    @staticmethod
    def pretrained_exists(path: str) -> bool:
        return os.path.exists(os.path.join(path, TreeTokenizer._filename))

    @property
    def leaf_start_index(self) -> int:
        return self._leaf_start_ind

    @property
    def arbitrary_start_index(self) -> int:
        return self._arbitrary_start_ind

    @property
    def eov_id(self) -> int:
        return self._eov_id

    @property
    def bpe_tokenizer(self) -> Tokenizer:
        return self._bpe_tokenizer

    @property
    def arbitrary_ids(self) -> List[int]:
        return list(range(self.arbitrary_start_index, self._vocab_size))

    def train(self, trees: List[Tree]) -> None:
        non_leaf_tokens = list(
            (
                set(
                    TreeTokenizer.non_arbitrary_to_token(node.name, reverse=False)
                    for tree in tqdm.tqdm(trees, desc="Collecting nodes tokens for tokenizer 1/2...")
                    for node in tree.nodes
                    if not node.is_leaf and node.is_visible
                )
            )
        )
        self._leaf_start_ind = len(non_leaf_tokens)
        non_arbitrary_leaf_tokens = list(
            set(
                TreeTokenizer.non_arbitrary_to_token(node.name, reverse=False)
                for tree in tqdm.tqdm(trees, desc="Collecting nodes tokens for tokenizer 2/2...")
                for node in tree.nodes
                if node.is_leaf and not node.is_arbitrary and node.is_visible
            )
        )
        special_tokens = non_leaf_tokens + non_arbitrary_leaf_tokens
        self._arbitrary_start_ind = len(special_tokens)
        special_tokens.extend(("[UNK]", "[PAD]", "[EOV]"))  # arbitrary node's tokens
        self._eov_id = len(special_tokens) - 1
        print(f"There are {len(special_tokens)} special tokens out of {self._vocab_size} vocabulary")

        tokenizer = Tokenizer(BPE(dropout=self._dropout, unk_token="[UNK]", fuse_unk=True))
        tokenizer.pre_tokenizer = Metaspace()
        trainer = BpeTrainer(
            vocab_size=self._vocab_size, min_frequency=self._min_frequency, special_tokens=special_tokens
        )
        bpe_nodes_iterator = (
            node.name for tree in trees for node in tree.nodes if node.is_arbitrary and node.is_visible
        )
        print("Training tokenizer...")
        tokenizer.train_from_iterator(bpe_nodes_iterator, trainer)

        print(f"The final vocabulary size is {tokenizer.get_vocab_size()}")

        tokenizer.post_processor = TemplateProcessing(
            single="$A [EOV]",
            pair="$A [EOV] $B:1 [EOV]:1",
            special_tokens=[("[EOV]", tokenizer.token_to_id("[EOV]"))],
        )
        self._bpe_tokenizer = tokenizer

    def encode(self, tree: Tree) -> List[int]:
        node_ids = []
        for node in tree.nodes:
            if node.is_visible:
                if node.is_arbitrary:
                    node_ids.extend(self._bpe_tokenizer.encode(node.name).ids)
                else:
                    node_ids.append(
                        self._bpe_tokenizer.token_to_id(TreeTokenizer.non_arbitrary_to_token(node.name, reverse=False))
                    )
        return node_ids

    def decode_string(self, ids: List[int]) -> str:
        return self.bpe_tokenizer.decode(ids).replace(" ", "").replace("▁", " ")[1:]

    @staticmethod
    def non_arbitrary_to_token(name: str, reverse: bool) -> str:
        return f"[{name}]" if not reverse else name[1:-1]

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
            return self._tokenizer.arbitrary_ids
        else:
            ret = []
            next_suitable_nodes = self._tree.get_next_suitable_nodes()
            for node_name in next_suitable_nodes:
                if node_name == TreeConstants.ARBITRARY_REPR.value:
                    ret.extend(self._tokenizer.arbitrary_ids)
                else:
                    ret.append(
                        self._tokenizer.bpe_tokenizer.token_to_id(
                            self._tokenizer.non_arbitrary_to_token(node_name, reverse=False)
                        )
                    )
            return ret

    def add_id(self, id_: int) -> None:
        is_arbitrary = id_ >= self._tokenizer.arbitrary_start_index

        if id_ == self._tokenizer.eov_id:
            token = self._tokenizer.decode_string(self._cur_arbitrary_ids)
            self._cur_arbitrary_ids = []
            self._tree.add_node(name=token, is_arbitrary=True)
        elif is_arbitrary:
            self._cur_arbitrary_ids.append(id_)
        else:
            token = self._tokenizer.bpe_tokenizer.id_to_token(id_)
            self._tree.add_node(TreeTokenizer.non_arbitrary_to_token(token, reverse=True), is_arbitrary=False)

import itertools
import json
import os
from typing import List, Optional

from omegaconf import DictConfig
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from src.common.utils import run_with_config
from src.psi.psi_datapoint.stateful.abstract_stateful import Stateful


class FLCCBPE(Stateful):
    _filename = "line_tokenizer"

    def __init__(self, vocab_size: int, min_frequency: int, dropout: float):
        self._vocab_size = vocab_size
        self._min_frequency = min_frequency
        self._dropout = dropout
        self._bpe_tokenizer: Optional[Tokenizer] = None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_id(self) -> int:
        return self._bpe_tokenizer.token_to_id("[EOS]")

    def encode(self, content: str) -> List[int]:
        # Add special token in order to use [EOS] for beam search stopping
        lines = content.splitlines(keepends=False)
        ids_lists = self._bpe_tokenizer.encode_batch(lines, add_special_tokens=True)
        return list(itertools.chain(*ids_lists))

    def decode(self, ids: List[int]) -> str:
        ids_lists = itertools.groupby(ids, lambda id_: id_ == self.eos_id)
        lines = self._bpe_tokenizer.decode_batch(ids_lists, skip_special_tokens=True)
        return "\n".join(lines)

    def save_pretrained(self, path: str) -> None:
        path = os.path.join(path, FLCCBPE._filename)
        os.makedirs(path, exist_ok=True)
        self._bpe_tokenizer.save(os.path.join(path, "bpe_tokenizer.json"))
        with open(os.path.join(path, "tokenizer_stuff.json"), "w") as f:
            json.dump([self._vocab_size, self._min_frequency, self._dropout], f)

    @staticmethod
    def from_pretrained(path: str) -> "FLCCBPE":
        path = os.path.join(path, FLCCBPE._filename)
        with open(os.path.join(path, "tokenizer_stuff.json")) as f:
            [vocab_size, min_frequency, dropout] = json.load(f)
        tokenizer = FLCCBPE(vocab_size, min_frequency, dropout)
        tokenizer._bpe_tokenizer = Tokenizer.from_file(os.path.join(path, "bpe_tokenizer.json"))
        return tokenizer

    @staticmethod
    def pretrained_exists(path: str) -> bool:
        return os.path.exists(os.path.join(path, FLCCBPE._filename))

    def train(self, data_path: str) -> None:
        bpe_special_tokens = ["[PAD]", "[UNK]", "[EOS]"]
        tokenizer = Tokenizer(BPE(dropout=self._dropout if self._dropout else None, unk_token="[UNK]", fuse_unk=True))
        trainer = BpeTrainer(
            vocab_size=self._vocab_size, min_frequency=self._min_frequency, special_tokens=bpe_special_tokens
        )

        print("Training tokenizer...")
        with open(data_path) as f:
            lines = [line for content in f for line in json.loads(content).splitlines(keepends=False)]

        tokenizer.train_from_iterator(lines, trainer)
        tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]",
            pair="$A [EOS] $B:1 [EOS]:1",
            special_tokens=[("[EOS]", tokenizer.token_to_id("[EOS]"))],
        )

        self._bpe_tokenizer = tokenizer
        self._vocab_size = tokenizer.get_vocab_size()
        print(f"The final vocab size is {self._vocab_size}")


if __name__ == "__main__":

    def main(config: DictConfig):
        tokenizer = FLCCBPE(config.tokenizer.vocab_size, config.tokenizer.min_frequency, config.tokenizer.dropout)
        data = config.source_data.train
        tokenizer.train(data)
        tokenizer.save_pretrained(config.save_path)
        _ = FLCCBPE.from_pretrained(config.save_path)

    run_with_config(main, "src/common/configs/config_flcc.yaml")

from typing import Set, List

import youtokentome

from flccpsisrc.common.token_holder import TokenHolder


class PlaceholderBpe:
    def __init__(self, placeholderBpeFolderPath: str):
        with open(placeholderBpeFolderPath + "/order.txt") as f:
            self._types = f.read().splitlines()
        self._models = [youtokentome.BPE(model=placeholderBpeFolderPath + "/" + t, n_threads=1) for t in self._types]
        self._vocab_size = sum(m.vocab_size() for m in self._models)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def types(self) -> Set[str]:
        return set(self._types)

    def decode(self, token_id) -> TokenHolder.Token:
        for m, t in zip(self._models, self._types):
            if token_id >= m.vocab_size():
                token_id -= m.vocab_size()
            else:
                return TokenHolder.Token(t, m.decode([token_id])[0])
        raise Exception()

    def encode(self, token: TokenHolder.Token) -> List[int]:
        offset = 0
        for m, t in zip(self._models, self._types):
            if token.token_type == t:
                return [x + offset for x in m.encode([token.token_text])[0][1:]]
            offset += m.vocab_size()
        raise Exception()

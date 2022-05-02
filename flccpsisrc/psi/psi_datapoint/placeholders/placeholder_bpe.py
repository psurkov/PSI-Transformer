import youtokentome


class PlaceholderBpe:
    def __init__(self, placeholderBpeFolderPath: str):
        with open(placeholderBpeFolderPath + "/order.txt") as f:
            types = f.read().splitlines()
        self.models = [youtokentome.BPE(model=placeholderBpeFolderPath + "/" + t, n_threads=1) for t in types]
        self._vocab_size = sum(m.vocab_size() for m in self.models)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def decode(self, token_id) -> str:
        for m in self.models:
            if token_id >= m.vocab_size():
                token_id -= m.vocab_size()
            else:
                return m.decode(token_id)[0]
        raise Exception()

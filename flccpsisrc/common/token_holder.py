import copy
import itertools
from typing import List, Optional


class TokenHolder:
    def __init__(self, data: List[Optional[str]]):
        self._data = data

    @staticmethod
    def from_tokens(tokens: List[str], last_token_finished) -> "TokenHolder":
        if len(tokens) == 0:
            if last_token_finished:
                return TokenHolder([None])
            else:
                return TokenHolder([])

        none_list = []
        for token in tokens:
            none_list.extend(token)
            none_list.append(None)
        if not last_token_finished:
            none_list.pop()
        return TokenHolder(none_list)

    def copy(self) -> "TokenHolder":
        return TokenHolder(copy.deepcopy(self._data))

    def __str__(self) -> str:
        return "".join(map(lambda x: x if x is not None else " ", self._data))

    def matches(self, other: "TokenHolder") -> bool:
        min_len = min(len(self._data), len(other._data))
        return self._data[:min_len] == other._data[:min_len]

    def remove_prefix(self, other: "TokenHolder") -> None:
        min_len = min(len(self._data), len(other._data))
        assert self._data[:min_len] == other._data[:min_len]
        self._data = self._data[min_len:]

    @property
    def is_empty(self) -> bool:
        return len(self._data) == 0

    @property
    def has_at_least_one_full_token(self) -> bool:
        return None in self._data

    @property
    def first_full(self) -> Optional[str]:
        assert not self.is_empty
        if self._data[0] is None:
            return None
        return "".join(itertools.takewhile(lambda x: x is not None, self._data))

    def extend(self, other: "TokenHolder") -> None:
        self._data.extend(other._data)

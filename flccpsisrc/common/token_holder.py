import copy
import itertools
from typing import List, Optional


class TokenHolder:
    class Token:
        def __init__(self, token_type, token_text):
            self._type = token_type
            self._text = token_text

        @property
        def token_type(self):
            return self._type

        @property
        def token_text(self):
            return self._text

        def __str__(self):
            return self.token_type + ":" + self.token_text

    def __init__(self, data_text: List[Optional[str]], data_types: List[str]):
        self._data_text = data_text
        self._data_types = data_types

    @staticmethod
    def empty():
        return TokenHolder([], [])

    def copy(self) -> "TokenHolder":
        return TokenHolder(copy.deepcopy(self._data_text), copy.deepcopy(self._data_types))

    def str_text(self) -> str:
        return "".join(map(lambda x: x if x is not None else " ", self._data_text))

    def str_types(self) -> str:
        return "(" + ", ".join(self._data_types) + ")"

    def __str__(self) -> str:
        return self.str_text() + self.str_types()

    def _text_matches(self, other: "TokenHolder") -> bool:
        text_min_len = min(len(self._data_text), len(other._data_text))
        return self._data_text[:text_min_len] == other._data_text[:text_min_len]

    def _types_matches(self, other: "TokenHolder") -> bool:
        types_min_len = min(len(self._data_types), len(other._data_types))
        for i in range(types_min_len):
            if self._data_types[i] == "UNKNOWN":
                continue
            if other._data_types[i] == "UNKNOWN":
                continue

            if self._data_types[i] != other._data_types[i]:
                return False
        return True

    def matches(self, other: "TokenHolder") -> bool:
        return self._types_matches(other) and self._text_matches(other)

    def remove_prefix(self, other: "TokenHolder") -> None:
        assert self.matches(other)
        # is_first_part_of_token =
        text_min_len = min(len(self._data_text), len(other._data_text))
        self._data_text = self._data_text[text_min_len:]
        # len([t for t in self._data_text if t is None])
        types_min_len = min(len(self._data_types), len(other._data_types))
        self._data_types = self._data_types[types_min_len:]

    @property
    def is_empty(self) -> bool:
        return len(self._data_text) == 0 and len(self._data_types) == 0

    @property
    def full_tokens(self) -> int:
        return len([t for t in self._data_text if t is None])

    @property
    def has_at_least_one_full_token(self) -> bool:
        return None in self._data_text

    @property
    def first_full(self) -> Optional[str]:
        assert not self.is_empty
        if self._data_text[0] is None:
            return None
        return "".join(itertools.takewhile(lambda x: x is not None, self._data_text))

    @property
    def first_type(self) -> Optional[str]:
        assert len(self._data_types) > 0
        return self._data_types[0]

    def add_part_of_token_text(self, text: str) -> "TokenHolder":
        self._data_text.extend(text)
        return self

    def add_token_type(self, token_type: str) -> "TokenHolder":
        self._data_types.append(token_type)
        return self

    def add_tokens(self, tokens: List[Token]) -> "TokenHolder":
        if len(tokens) == 0:
            return self
        none_list = []
        for token in tokens:
            none_list.extend(token.token_text)
            none_list.append(None)
        none_list.pop()
        self.extend(TokenHolder(none_list, [t.token_type for t in tokens]))
        return self

    def finish_last_token(self) -> "TokenHolder":
        self._data_text.append(None)
        return self

    def extend(self, other: "TokenHolder") -> None:
        self._data_text.extend(other._data_text)
        self._data_types.extend(other._data_types)

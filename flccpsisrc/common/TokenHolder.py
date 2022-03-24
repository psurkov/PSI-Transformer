from typing import List


class TokenHolder:
    def __init__(self, tokens: List[str], create_new: bool):
        self._tokens = tokens
        self._create_new = create_new

    def add_part_of_token(self, part_of_token: str) -> None:
        if self._create_new:
            self._tokens.append("")
            self._create_new = False
        self._tokens[-1] += part_of_token

    def finish_token(self) -> None:
        self._create_new = True

    def __str__(self) -> str:
        res = " ".join(self._tokens)
        if self._create_new:
            res += " "
        return res

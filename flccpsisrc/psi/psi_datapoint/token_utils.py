from typing import List, Tuple


def match_tokens(first: List[str], second: List[str]) -> Tuple[int, int]:
    """
    :return: (fully matched tokens, matched length of next token) or (-1, -1) if doesn't match
    """
    fully = 0
    for pref_token, token in zip(second, first):
        if pref_token != token:
            min_len = min(len(pref_token), len(token))
            if pref_token[:min_len] != token[:min_len]:
                return -1, -1
            return fully, min_len
        fully += 1
    return fully, 0


def cut_matched_tokens(first: List[str], second: List[str]) -> Tuple[List[str], List[str]]:
    """
    :return: (cut first, cut second)
    """
    fully, next_len = match_tokens(first, second)
    assert fully != -1
    first = first[fully:]
    second = second[fully:]
    if next_len > 0:
        first[0] = first[0][next_len:]
        second[0] = second[0][next_len:]
        if len(first[0]) == 0:
            first = first[1:]
        if len(second[0]) == 0:
            second = second[1:]
    return first, second

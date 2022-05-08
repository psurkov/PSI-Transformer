import dataclasses
from typing import List, Callable

from flccpsisrc.common.token_holder import TokenHolder
from flccpsisrc.psi.psi_datapoint.placeholders.placeholder_bpe import PlaceholderBpe
from flccpsisrc.psi.psi_datapoint.tree_structures.special_ids import SPECIAL_IDS_RESERVED_SIZE, SpecialIds
from flccpsisrc.psi.psi_datapoint.tree_structures.split_tree_version import Version
from flccpsisrc.psi.psi_datapoint.tree_structures.structure_decompression import StructureDecompression


class NextIdsSuggester:
    @dataclasses.dataclass
    class PossibleIdSuggestion:
        exactly_fit: List[int]
        need_check: List[int]
        can_stop: bool

        @staticmethod
        def empty_continue():
            return NextIdsSuggester.PossibleIdSuggestion([], [], False)

        @staticmethod
        def empty_stop():
            return NextIdsSuggester.PossibleIdSuggestion([], [], True)

    _pipeline: List[Callable[[Version], PossibleIdSuggestion]]

    def _placeholder_with_at_least_one_full_token(self, version: Version) -> PossibleIdSuggestion:
        if version.state not in [
            Version.State.AWAIT_PLACEHOLDER_FIRST_TOKEN, Version.State.AWAIT_PLACEHOLDER_CONTINUATION_TOKEN
        ]:
            return NextIdsSuggester.PossibleIdSuggestion.empty_continue()
        if not version.remaining_prefix_holder.has_at_least_one_full_token:
            return NextIdsSuggester.PossibleIdSuggestion.empty_continue()

        placeholder_text = version.remaining_prefix_holder.first_full
        if version.state == Version.State.AWAIT_PLACEHOLDER_FIRST_TOKEN:
            # assert version.remaining_prefix_holder.first_type == version.current_placeholder_type todo
            if version.remaining_prefix_holder.first_type != version.current_placeholder_type:
                return NextIdsSuggester.PossibleIdSuggestion.empty_stop()
        if placeholder_text is None:
            assert version.state == Version.State.AWAIT_PLACEHOLDER_CONTINUATION_TOKEN
            return NextIdsSuggester.PossibleIdSuggestion([], [SpecialIds.END_OF_PLACEHOLDER.value], True)
        else:
            placeholder_id = self._placeholders_bpe.encode(
                TokenHolder.Token(version.current_placeholder_type, placeholder_text)
            )[0]
            token_id = SPECIAL_IDS_RESERVED_SIZE + \
                       self._structure_decompression.vocab_size + \
                       placeholder_id
            return NextIdsSuggester.PossibleIdSuggestion([token_id], [], True)

    def _structure_tokens_matching_prefix(self, version: Version) -> PossibleIdSuggestion:
        if version.state not in (Version.State.AWAIT_START, Version.State.AWAIT_STRUCTURE_TOKEN):
            return NextIdsSuggester.PossibleIdSuggestion.empty_continue()
        return NextIdsSuggester.PossibleIdSuggestion.empty_continue()  # todo

    def _all_placeholders(self, version: Version) -> PossibleIdSuggestion:
        if version.state in [
            Version.State.AWAIT_PLACEHOLDER_FIRST_TOKEN, Version.State.AWAIT_PLACEHOLDER_CONTINUATION_TOKEN
        ]:
            return NextIdsSuggester.PossibleIdSuggestion(
                [],
                self._placeholder_ids_of_type[version.current_placeholder_type],
                False
            )
        else:
            return NextIdsSuggester.PossibleIdSuggestion.empty_continue()

    def _all_structures(self, version: Version) -> PossibleIdSuggestion:
        if version.state in [
            Version.State.AWAIT_START,
            Version.State.AWAIT_STRUCTURE_TOKEN
        ]:
            return NextIdsSuggester.PossibleIdSuggestion([], self._structure_ids, False)
        else:
            return NextIdsSuggester.PossibleIdSuggestion.empty_continue()

    def _eoc(self, version: Version) -> PossibleIdSuggestion:
        if version.can_eof():
            return NextIdsSuggester.PossibleIdSuggestion([], [SpecialIds.END_OF_NODE_CHILDREN.value], False)
        else:
            return NextIdsSuggester.PossibleIdSuggestion.empty_continue()

    def _eop(self, version: Version) -> PossibleIdSuggestion:
        if version.state == Version.State.AWAIT_PLACEHOLDER_CONTINUATION_TOKEN:
            return NextIdsSuggester.PossibleIdSuggestion([], [SpecialIds.END_OF_PLACEHOLDER.value], False)
        else:
            return NextIdsSuggester.PossibleIdSuggestion.empty_continue()

    def __init__(
            self,
            structure_decompression: StructureDecompression,
            placeholders_bpe: PlaceholderBpe
    ):
        self._structure_decompression = structure_decompression
        self._placeholders_bpe = placeholders_bpe
        self._structure_ids = list(
            SPECIAL_IDS_RESERVED_SIZE + token_id
            for token_id in range(structure_decompression.vocab_size)
        )
        self._placeholder_ids_of_type = {t: [
            x + SPECIAL_IDS_RESERVED_SIZE + self._structure_decompression.vocab_size
            for x in self._placeholders_bpe.ids_of_type(t)
        ] for t in self._placeholders_bpe.types
        }
        self._all_placeholder_ids = [i + SPECIAL_IDS_RESERVED_SIZE + self._structure_decompression.vocab_size
                                     for i in range(self._placeholders_bpe.vocab_size)]
        self._pipeline = [
            self._placeholder_with_at_least_one_full_token,
            self._eop,
            self._eoc,
            self._structure_tokens_matching_prefix,
            self._all_placeholders,
            self._all_structures,
        ]

    def possible_next_id(self, version: Version) -> List[int]:
        res = []
        for p in self._pipeline:
            suggestion = p(version)
            res.extend(suggestion.exactly_fit)
            if version.remaining_prefix_holder.is_empty:
                res.extend(suggestion.need_check)
            else:
                res.extend(token_id for token_id in suggestion.need_check
                           if version.remaining_prefix_holder.matches(version.decode_if_add_token_id(token_id))
                           )
            if suggestion.can_stop:
                break
        return res

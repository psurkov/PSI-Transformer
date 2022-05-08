import copy
import dataclasses
from enum import Enum
from typing import List, Optional, Callable

from flccpsisrc.common.token_holder import TokenHolder
from flccpsisrc.psi.psi_datapoint.placeholders.placeholder_bpe import PlaceholderBpe
from flccpsisrc.psi.psi_datapoint.tree_structures.special_ids import SPECIAL_IDS_RESERVED_SIZE, SpecialIds
from flccpsisrc.psi.psi_datapoint.tree_structures.structure_decompression import StructureDecompression, \
    NodeContentFragment


class ChangeStatus(Enum):
    IN_PROGRESS = 0
    TERMINATED = 1


class VersionsShared:
    def __init__(
            self,
            structure_decompression: StructureDecompression,
            placeholders_bpe: PlaceholderBpe
    ):
        self.structure_decompression = structure_decompression
        self.placeholders_bpe = placeholders_bpe


class Version:
    class State(Enum):
        AWAIT_START = 0
        AWAIT_STRUCTURE_TOKEN = 1
        AWAIT_PLACEHOLDER_FIRST_TOKEN = 2
        AWAIT_PLACEHOLDER_CONTINUATION_TOKEN = 3

    def __init__(
            self,
            nodes: List["Version.Node"],
            visit_stack: List[int],
            state: State,
            remaining_prefix_holder: TokenHolder,
            versions_shared: "VersionsShared",
            current_placeholder_type: Optional[str]
    ):
        self._nodes = nodes
        self._visit_stack = visit_stack
        self._state = state
        self._remaining_prefix_holder = remaining_prefix_holder
        self._versions_shared = versions_shared
        self._current_placeholder_type = current_placeholder_type

    @staticmethod
    def empty_version(
            rollback_prefix_holder: TokenHolder,
            versions_shared: "VersionsShared"
    ) -> "Version":
        return Version(
            [],
            [],
            Version.State.AWAIT_START,
            rollback_prefix_holder.copy(),
            versions_shared,
            None,
        )

    @dataclasses.dataclass
    class Node:
        node_type: int
        children: List[int]
        placeholders: List[List[int]]

    @property
    def remaining_prefix_holder(self) -> TokenHolder:
        return self._remaining_prefix_holder

    @property
    def state(self) -> State:
        return self._state

    @property
    def current_placeholder_type(self) -> Optional[str]:
        return self._current_placeholder_type

    def copy(self) -> "Version":
        return Version(
            copy.deepcopy(self._nodes),
            copy.deepcopy(self._visit_stack),
            self._state,
            self._remaining_prefix_holder.copy(),
            self._versions_shared,
            self._current_placeholder_type
        )

    def can_eof(self) -> bool:
        if self._state == Version.State.AWAIT_START:
            return True
        if self._state == Version.State.AWAIT_STRUCTURE_TOKEN:
            if len(self._visit_stack) == 0:
                return False
            current = self._nodes[self._visit_stack[-1]]
            current_content = self._versions_shared.structure_decompression.get_content_fragments_for(
                current.node_type)
            if len(current.children) >= current_content.children:
                return True
            return False
        return False

    def _on_next_token(
            self,
            token_id: int,
            on_end_of_placeholder: Callable,
            on_end_of_node_children: Callable,
            on_structure_token: Callable,
            on_placeholder_token: Callable
    ):
        if token_id == SpecialIds.END_OF_PLACEHOLDER.value:
            assert self._state in [
                Version.State.AWAIT_PLACEHOLDER_FIRST_TOKEN,
                Version.State.AWAIT_PLACEHOLDER_CONTINUATION_TOKEN
            ]
            current = self._nodes[self._visit_stack[-1]]
            current_content = self._versions_shared.structure_decompression.get_content_fragments_for(
                current.node_type)
            assert len(current.placeholders) <= current_content.placeholders
            return on_end_of_placeholder()
        elif token_id == SpecialIds.END_OF_NODE_CHILDREN.value:
            if self._state == Version.State.AWAIT_START:
                return on_end_of_node_children()
            assert self._state == Version.State.AWAIT_STRUCTURE_TOKEN
            current = self._nodes[self._visit_stack[-1]]
            current_content = self._versions_shared.structure_decompression.get_content_fragments_for(
                current.node_type)
            assert len(current.children) >= current_content.children
            return on_end_of_node_children()
        elif SPECIAL_IDS_RESERVED_SIZE <= token_id \
                < SPECIAL_IDS_RESERVED_SIZE + self._versions_shared.structure_decompression.vocab_size:
            assert self._state == Version.State.AWAIT_STRUCTURE_TOKEN \
                   or self._state == Version.State.AWAIT_START
            return on_structure_token(token_id - SPECIAL_IDS_RESERVED_SIZE)
        elif SPECIAL_IDS_RESERVED_SIZE + self._versions_shared.structure_decompression.vocab_size <= token_id \
                < SPECIAL_IDS_RESERVED_SIZE + self._versions_shared.structure_decompression.vocab_size \
                + self._versions_shared.placeholders_bpe.vocab_size:
            assert self._state in [
                Version.State.AWAIT_PLACEHOLDER_FIRST_TOKEN,
                Version.State.AWAIT_PLACEHOLDER_CONTINUATION_TOKEN
            ]
            return on_placeholder_token(
                token_id - SPECIAL_IDS_RESERVED_SIZE - self._versions_shared.structure_decompression.vocab_size)
        else:
            assert False

    def add_token(self, token_id: int) -> "ChangeStatus":
        def update_state():
            current = self._nodes[self._visit_stack[-1]]
            current_content = self._versions_shared.structure_decompression.get_content_fragments_for(
                current.node_type)
            next_gen_index = len(current.placeholders) + len(current.children)

            if next_gen_index >= current_content.generation_places \
                    or current_content.generation_place(next_gen_index).fragment_type \
                    == NodeContentFragment.FragmentType.CHILD:
                self._state = Version.State.AWAIT_STRUCTURE_TOKEN
                self._current_placeholder_type = None
            else:
                self._state = Version.State.AWAIT_PLACEHOLDER_FIRST_TOKEN
                self._current_placeholder_type = current_content.generation_place(next_gen_index).placeholder_type
                self._nodes[self._visit_stack[-1]].placeholders.append([])

        def on_end_of_placeholder():
            update_state()
            return ChangeStatus.IN_PROGRESS

        def on_end_of_node_children():
            if self._state == Version.State.AWAIT_START:
                return ChangeStatus.IN_PROGRESS
            self._visit_stack.pop()
            if len(self._visit_stack) == 0:
                return ChangeStatus.TERMINATED
            update_state()
            return ChangeStatus.IN_PROGRESS

        def on_structure_token(token_id):
            if self._state == Version.State.AWAIT_START:
                self._state = Version.State.AWAIT_STRUCTURE_TOKEN

            new_node = Version.Node(token_id, [], [])
            if len(self._visit_stack) > 0:
                self._nodes[self._visit_stack[-1]].children.append(len(self._nodes))
            self._visit_stack.append(len(self._nodes))
            self._nodes.append(new_node)
            update_state()
            if token_id in self._versions_shared.structure_decompression.can_terminate_if_start_generate:
                return ChangeStatus.TERMINATED
            else:
                return ChangeStatus.IN_PROGRESS

        def on_placeholder_token(token_id):
            self._nodes[self._visit_stack[-1]].placeholders[-1].append(token_id)
            self._state = Version.State.AWAIT_PLACEHOLDER_CONTINUATION_TOKEN
            return ChangeStatus.IN_PROGRESS

        self._remaining_prefix_holder.remove_prefix(self.decode_if_add_token_id(token_id))
        return self._on_next_token(
            token_id,
            on_end_of_placeholder=on_end_of_placeholder,
            on_end_of_node_children=on_end_of_node_children,
            on_structure_token=on_structure_token,
            on_placeholder_token=on_placeholder_token
        )

    def decode_if_add_token_id(self, token_id: int) -> TokenHolder:

        def on_end_of_placeholder():
            current = self._nodes[self._visit_stack[-1]]
            current_content = self._versions_shared.structure_decompression.get_content_fragments_for(
                current.node_type)
            tokens = current_content.text_after_n_placeholders(len(current.placeholders))
            if len(tokens) == 0:
                res = TokenHolder.empty().finish_last_token()
            else:
                res = TokenHolder.empty().finish_last_token().add_tokens(
                    [TokenHolder.Token("STRUCTURE", text) for text in tokens]
                ).finish_last_token()
            return res

        def on_end_of_node_children():
            if self._state == Version.State.AWAIT_START:
                return TokenHolder.empty()
            if len(self._visit_stack) == 1:
                return TokenHolder.empty()
            pred = self._nodes[self._visit_stack[-2]]
            pred_content = self._versions_shared.structure_decompression.get_content_fragments_for(pred.node_type)
            if len(pred.children) <= pred_content.children:
                tokens = pred_content.text_after_n_children(len(pred.children))
                res = TokenHolder.empty().add_tokens(
                    [TokenHolder.Token("STRUCTURE", text) for text in tokens]
                )
                if len(tokens) > 0:
                    res.finish_last_token()
                return res
            return TokenHolder.empty()

        def on_structure_token(token_id):
            tokens = self._versions_shared.structure_decompression.get_content_fragments_for(token_id).text_prefix()
            res = TokenHolder.empty().add_tokens(
                [TokenHolder.Token("STRUCTURE", text) for text in tokens]
            )
            if len(tokens) > 0:
                res.finish_last_token()
            return res

        def on_placeholder_token(token_id):
            token = self._versions_shared.placeholders_bpe.decode(token_id)
            res = TokenHolder.empty().add_part_of_token_text(
                token.token_text
            )
            if self._state == Version.State.AWAIT_PLACEHOLDER_FIRST_TOKEN:
                res.add_token_type(token.token_type)
            return res

        return self._on_next_token(
            token_id,
            on_end_of_placeholder=on_end_of_placeholder,
            on_end_of_node_children=on_end_of_node_children,
            on_structure_token=on_structure_token,
            on_placeholder_token=on_placeholder_token
        )

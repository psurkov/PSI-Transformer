import copy
import dataclasses
from enum import Enum
from typing import List, Callable

from youtokentome import BPE

from flccpsisrc.common.token_holder import TokenHolder
from flccpsisrc.psi.psi_datapoint.tree_structures.special_ids import SpecialIds, SPECIAL_IDS_RESERVED_SIZE
from flccpsisrc.psi.psi_datapoint.tree_structures.structure_decompression import StructureDecompression, \
    NodeContentFragment


class SplitTreeBuilder:
    class ChangeStatus(Enum):
        IN_PROGRESS = 0
        TERMINATED = 1

    class VersionsShared:
        def __init__(
                self,
                structure_decompression: StructureDecompression,
                placeholders_bpe: BPE
        ):
            self.structure_decompression = structure_decompression
            self.placeholders_bpe = placeholders_bpe

            self.placeholders_tokens = list(
                SPECIAL_IDS_RESERVED_SIZE + structure_decompression.vocab_size + token_id
                for token_id in range(placeholders_bpe.vocab_size())
            )
            self.placeholders_tokens.append(SpecialIds.END_OF_PLACEHOLDER.value)

            self.structure_tokens = list(
                SPECIAL_IDS_RESERVED_SIZE + token_id
                for token_id in range(structure_decompression.vocab_size)
            )

            self.structure_and_end_of_node_children_tokens = copy.deepcopy(self.structure_tokens)
            self.structure_and_end_of_node_children_tokens.append(SpecialIds.END_OF_NODE_CHILDREN.value)

    class Version:
        class State(Enum):
            AWAIT_START = 0
            AWAIT_STRUCTURE_TOKEN = 1
            AWAIT_PLACEHOLDER_TOKEN = 2

        def __init__(
                self,
                nodes: List["SplitTreeBuilder.Version.Node"],
                visit_stack: List[int],
                state,
                remaining_prefix_holder: TokenHolder,
                versions_shared: "SplitTreeBuilder.VersionsShared",
        ):
            self._nodes = nodes
            self._visit_stack = visit_stack
            self._state = state
            self._remaining_prefix_holder = remaining_prefix_holder
            self._versions_shared = versions_shared

        @staticmethod
        def empty_version(
                rollback_prefix_holder: TokenHolder,
                versions_shared: "SplitTreeBuilder.VersionsShared"
        ) -> "SplitTreeBuilder.Version":
            return SplitTreeBuilder.Version(
                [],
                [],
                SplitTreeBuilder.Version.State.AWAIT_START,
                rollback_prefix_holder.copy(),
                versions_shared
            )

        @dataclasses.dataclass
        class Node:
            node_type: int
            children: List[int]
            placeholders: List[List[int]]

        def copy(self) -> "SplitTreeBuilder.Version":
            return SplitTreeBuilder.Version(
                copy.deepcopy(self._nodes),
                copy.deepcopy(self._visit_stack),
                self._state,
                self._remaining_prefix_holder.copy(),
                self._versions_shared
            )

        def _get_next_possible_by_state_ids(self) -> List[int]:
            if self._state == SplitTreeBuilder.Version.State.AWAIT_START:
                return self._versions_shared.structure_and_end_of_node_children_tokens
            elif self._state == SplitTreeBuilder.Version.State.AWAIT_PLACEHOLDER_TOKEN:
                return self._versions_shared.placeholders_tokens
            elif self._state == SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN:
                if len(self._visit_stack) > 0:
                    current = self._nodes[self._visit_stack[-1]]
                    current_content = self._versions_shared.structure_decompression.get_content_fragments_for(
                        current.node_type)
                    if len(current.children) >= current_content.children:
                        return self._versions_shared.structure_and_end_of_node_children_tokens
                return self._versions_shared.structure_tokens

        def get_next_possible_ids(self) -> List[int]:
            possible_by_state = self._get_next_possible_by_state_ids()

            if self._remaining_prefix_holder.is_empty:
                return possible_by_state

            if self._state == SplitTreeBuilder.Version.State.AWAIT_PLACEHOLDER_TOKEN:
                if self._remaining_prefix_holder.has_at_least_one_full_token:
                    placeholder_token = self._remaining_prefix_holder.first_full
                    if placeholder_token is None:
                        eof_matches = self._remaining_prefix_holder.matches(
                            self.decode_if_add_token_id(SpecialIds.END_OF_PLACEHOLDER.value)
                        )
                        if eof_matches:
                            return SpecialIds.END_OF_PLACEHOLDER.value
                        else:
                            return []
                    else:
                        placeholder_id = self._versions_shared.placeholders_bpe.encode([placeholder_token])[0][1]
                        return SPECIAL_IDS_RESERVED_SIZE + \
                               self._versions_shared.structure_decompression.vocab_size + \
                               placeholder_id

            return [
                token_id for token_id in possible_by_state if
                self._remaining_prefix_holder.matches(self.decode_if_add_token_id(token_id))
            ]

        def _on_next_token(
                self,
                token_id: int,
                on_end_of_placeholder: Callable,
                on_end_of_node_children: Callable,
                on_structure_token: Callable,
                on_placeholder_token: Callable
        ):
            if token_id == SpecialIds.END_OF_PLACEHOLDER.value:
                assert self._state == SplitTreeBuilder.Version.State.AWAIT_PLACEHOLDER_TOKEN
                current = self._nodes[self._visit_stack[-1]]
                current_content = self._versions_shared.structure_decompression.get_content_fragments_for(
                    current.node_type)
                assert len(current.placeholders) <= current_content.placeholders
                return on_end_of_placeholder()
            elif token_id == SpecialIds.END_OF_NODE_CHILDREN.value:
                if self._state == SplitTreeBuilder.Version.State.AWAIT_START:
                    return on_end_of_node_children()
                assert self._state == SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN
                current = self._nodes[self._visit_stack[-1]]
                current_content = self._versions_shared.structure_decompression.get_content_fragments_for(
                    current.node_type)
                assert len(current.children) >= current_content.children
                return on_end_of_node_children()
            elif SPECIAL_IDS_RESERVED_SIZE <= token_id \
                    < SPECIAL_IDS_RESERVED_SIZE + self._versions_shared.structure_decompression.vocab_size:
                assert self._state == SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN \
                       or self._state == SplitTreeBuilder.Version.State.AWAIT_START
                return on_structure_token(token_id - SPECIAL_IDS_RESERVED_SIZE)
            elif SPECIAL_IDS_RESERVED_SIZE + self._versions_shared.structure_decompression.vocab_size <= token_id \
                    < SPECIAL_IDS_RESERVED_SIZE + self._versions_shared.structure_decompression.vocab_size \
                    + self._versions_shared.placeholders_bpe.vocab_size():
                assert self._state == SplitTreeBuilder.Version.State.AWAIT_PLACEHOLDER_TOKEN
                return on_placeholder_token(
                    token_id - SPECIAL_IDS_RESERVED_SIZE - self._versions_shared.structure_decompression.vocab_size)
            else:
                assert False

        def add_token(self, token_id: int) -> "SplitTreeBuilder.ChangeStatus":
            def on_end_of_placeholder():
                current = self._nodes[self._visit_stack[-1]]
                current_content = self._versions_shared.structure_decompression.get_content_fragments_for(
                    current.node_type)
                next_gen_index = len(current.placeholders) + len(current.children)
                if next_gen_index == current_content.generation_places:
                    self._state = SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN
                else:
                    if current_content.generation_place(next_gen_index) == NodeContentFragment.FragmentType.CHILD:
                        self._state = SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN
                    else:
                        self._nodes[self._visit_stack[-1]].placeholders.append([])
                return SplitTreeBuilder.ChangeStatus.IN_PROGRESS

            def on_end_of_node_children():
                if self._state == SplitTreeBuilder.Version.State.AWAIT_START:
                    return SplitTreeBuilder.ChangeStatus.IN_PROGRESS
                self._visit_stack.pop()
                if len(self._visit_stack) == 0:
                    return SplitTreeBuilder.ChangeStatus.TERMINATED
                return SplitTreeBuilder.ChangeStatus.IN_PROGRESS

            def on_structure_token(token_id):
                if self._state == SplitTreeBuilder.Version.State.AWAIT_START:
                    self._state = SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN

                new_node = SplitTreeBuilder.Version.Node(token_id, [], [])
                if len(self._visit_stack) > 0:
                    self._nodes[self._visit_stack[-1]].children.append(len(self._nodes))
                self._visit_stack.append(len(self._nodes))
                self._nodes.append(new_node)
                new_node_content = self._versions_shared.structure_decompression.get_content_fragments_for(
                    new_node.node_type)
                if new_node_content.generation_places > 0 and \
                        new_node_content.generation_place(0) == NodeContentFragment.FragmentType.PLACEHOLDER:
                    self._state = SplitTreeBuilder.Version.State.AWAIT_PLACEHOLDER_TOKEN
                    new_node.placeholders.append([])
                if token_id in self._versions_shared.structure_decompression.can_terminate_if_start_generate:
                    return SplitTreeBuilder.ChangeStatus.TERMINATED
                else:
                    return SplitTreeBuilder.ChangeStatus.IN_PROGRESS

            def on_placeholder_token(token_id):
                self._nodes[self._visit_stack[-1]].placeholders[-1].append(token_id)
                return SplitTreeBuilder.ChangeStatus.IN_PROGRESS

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
                    return TokenHolder.from_tokens([], True)
                else:
                    return TokenHolder.from_tokens([""] + tokens, True)

            def on_end_of_node_children():
                if self._state == SplitTreeBuilder.Version.State.AWAIT_START:
                    return TokenHolder.from_tokens([], False)
                if len(self._visit_stack) == 1:
                    return TokenHolder.from_tokens([], False)
                pred = self._nodes[self._visit_stack[-2]]
                pred_content = self._versions_shared.structure_decompression.get_content_fragments_for(pred.node_type)
                if len(pred.children) <= pred_content.children:
                    tokens = pred_content.text_after_n_children(len(pred.children))
                    return TokenHolder.from_tokens(
                        tokens,
                        len(tokens) > 0
                    )
                return TokenHolder.from_tokens([], False)

            def on_structure_token(token_id):
                tokens = self._versions_shared.structure_decompression.get_content_fragments_for(token_id).text_prefix()
                return TokenHolder.from_tokens(
                    tokens,
                    len(tokens) > 0
                )

            def on_placeholder_token(token_id):
                return TokenHolder.from_tokens(self._versions_shared.placeholders_bpe.decode([token_id]), False)

            return self._on_next_token(
                token_id,
                on_end_of_placeholder=on_end_of_placeholder,
                on_end_of_node_children=on_end_of_node_children,
                on_structure_token=on_structure_token,
                on_placeholder_token=on_placeholder_token
            )

    def __init__(
            self,
            structure_decompression: StructureDecompression,
            placeholders_bpe: BPE,
            rollback_prefix_holder: TokenHolder
    ):
        self._versions_shared = SplitTreeBuilder.VersionsShared(structure_decompression, placeholders_bpe)
        self._fixed_order_version_ids = [0]
        self._versions = {0: SplitTreeBuilder.Version.empty_version(rollback_prefix_holder, self._versions_shared)}
        self._next_version_id = 1
        self._rollback_prefix_holder = rollback_prefix_holder

    @property
    def active_versions_ids(self) -> List[int]:
        return self._fixed_order_version_ids

    def filter_active_versions(self, required_active_versions: List[int]) -> None:
        required_active_versions_set = set(required_active_versions)
        assert required_active_versions_set.issubset(self.active_versions_ids)
        self._fixed_order_version_ids = required_active_versions
        self._versions = {key: self._versions[key] for key in self._versions if key in required_active_versions_set}

    def create_copy(self, version_id_to_copy) -> int:
        assert version_id_to_copy in self._versions
        created_version_id = self._next_version_id
        self._next_version_id += 1
        self._versions[created_version_id] = self._versions[version_id_to_copy].copy()
        self._fixed_order_version_ids.append(created_version_id)
        return created_version_id

    def get_next_possible_ids(self, version_id) -> List[int]:
        assert version_id in self._versions
        return self._versions[version_id].get_next_possible_ids()

    def add_token(self, version_id, token_id) -> ChangeStatus:
        assert version_id in self._versions
        return self._versions[version_id].add_token(token_id)

    def decode_if_add(self, version_id, token_id) -> TokenHolder:
        assert version_id in self._versions
        return self._versions[version_id].decode_if_add_token_id(token_id)

    def decode_generated_ids(self, ids: List[int]) -> str:
        new_version = SplitTreeBuilder.Version.empty_version(
            self._rollback_prefix_holder,
            self._versions_shared
        )
        suggestion = TokenHolder.from_tokens([], False)
        for token_id in ids:
            suggestion.extend(new_version.decode_if_add_token_id(token_id))
            new_version.add_token(token_id)
        suggestion.remove_prefix(self._rollback_prefix_holder)
        return suggestion.__str__()

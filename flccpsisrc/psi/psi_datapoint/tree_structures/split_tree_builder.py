import copy
import dataclasses
from enum import Enum
from typing import List, Callable, Set

from youtokentome import BPE

from flccpsisrc.psi.psi_datapoint.tree_structures.special_ids import SpecialIds, SPECIAL_IDS_RESERVED_SIZE
from flccpsisrc.psi.psi_datapoint.tree_structures.structure_decompression import StructureDecompression


class SplitTreeBuilder:
    class ChangeStatus(Enum):
        IN_PROGRESS = 0
        TERMINATED = 1

    class Version:
        class State(Enum):
            AWAIT_STRUCTURE_TOKEN = 0
            AWAIT_PLACEHOLDER_TOKEN = 1

        def __init__(
                self,
                nodes: List["SplitTreeBuilder.Version.Node"],
                visit_stack: List[int],
                state,
                structure_decompression: StructureDecompression,
                placeholders_bpe: BPE
        ):
            self._nodes = nodes
            self._visit_stack = visit_stack
            self._state = state
            self._structure_decompression = structure_decompression
            self._placeholders_bpe = placeholders_bpe

        @staticmethod
        def empty_version(
                structure_decompression: StructureDecompression,
                placeholders_bpe: BPE
        ) -> "SplitTreeBuilder.Version":
            return SplitTreeBuilder.Version(
                [],
                [],
                SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN,
                structure_decompression,
                placeholders_bpe
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
                self._structure_decompression,
                self._placeholders_bpe
            )

        def get_next_possible_ids(self) -> Set[int]:
            if self._state == SplitTreeBuilder.Version.State.AWAIT_PLACEHOLDER_TOKEN:
                res = set(
                    SPECIAL_IDS_RESERVED_SIZE + self._structure_decompression.vocab_size + token_id
                    for token_id in range(self._placeholders_bpe.vocab_size())
                )
                res.add(SpecialIds.END_OF_PLACEHOLDER.value)
                return res
            elif self._state == SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN:
                res = set(
                    SPECIAL_IDS_RESERVED_SIZE + token_id
                    for token_id in range(self._structure_decompression.vocab_size)
                )
                if len(self._visit_stack) > 0:
                    current = self._nodes[self._visit_stack[-1]]
                    current_content = self._structure_decompression.get_content_fragments_for(current.node_type)
                    if len(current.children) >= current_content.children:
                        res.add(SpecialIds.END_OF_NODE_CHILDREN.value)
                return res

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
                current_content = self._structure_decompression.get_content_fragments_for(current.node_type)
                assert len(current.placeholders) <= current_content.placeholders
                return on_end_of_placeholder()
            elif token_id == SpecialIds.END_OF_NODE_CHILDREN.value:
                assert self._state == SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN
                current = self._nodes[self._visit_stack[-1]]
                current_content = self._structure_decompression.get_content_fragments_for(current.node_type)
                assert len(current.children) >= current_content.children
                return on_end_of_node_children()
            elif SPECIAL_IDS_RESERVED_SIZE <= token_id \
                    < SPECIAL_IDS_RESERVED_SIZE + self._structure_decompression.vocab_size:
                assert self._state == SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN
                return on_structure_token(token_id - SPECIAL_IDS_RESERVED_SIZE)
            elif SPECIAL_IDS_RESERVED_SIZE + self._structure_decompression.vocab_size <= token_id \
                    < SPECIAL_IDS_RESERVED_SIZE + self._structure_decompression.vocab_size \
                    + self._placeholders_bpe.vocab_size():
                assert self._state == SplitTreeBuilder.Version.State.AWAIT_PLACEHOLDER_TOKEN
                return on_placeholder_token(
                    token_id - SPECIAL_IDS_RESERVED_SIZE - self._structure_decompression.vocab_size)
            else:
                assert False

        def add_token(self, token_id: int) -> "SplitTreeBuilder.ChangeStatus":
            def on_end_of_placeholder():
                current = self._nodes[self._visit_stack[-1]]
                current_content = self._structure_decompression.get_content_fragments_for(current.node_type)
                if len(current.placeholders) == current_content.placeholders:
                    self._state = SplitTreeBuilder.Version.State.AWAIT_STRUCTURE_TOKEN
                else:
                    self._nodes[self._visit_stack[-1]].placeholders.append([])
                return SplitTreeBuilder.ChangeStatus.IN_PROGRESS

            def on_end_of_node_children():
                self._visit_stack.pop()
                if len(self._visit_stack) == 0:
                    return SplitTreeBuilder.ChangeStatus.TERMINATED
                return SplitTreeBuilder.ChangeStatus.IN_PROGRESS

            def on_structure_token(token_id):
                new_node = SplitTreeBuilder.Version.Node(token_id, [], [])
                if len(self._visit_stack) > 0:
                    self._nodes[self._visit_stack[-1]].children.append(len(self._nodes))
                self._visit_stack.append(len(self._nodes))
                self._nodes.append(new_node)
                new_node_content = self._structure_decompression.get_content_fragments_for(new_node.node_type)
                if new_node_content.placeholders > 0:
                    self._state = SplitTreeBuilder.Version.State.AWAIT_PLACEHOLDER_TOKEN
                return SplitTreeBuilder.ChangeStatus.IN_PROGRESS # todo

            def on_placeholder_token(token_id):
                if len(self._visit_stack) == 0:
                    return
                insert_to = self._nodes[self._visit_stack[-1]].placeholders
                if len(insert_to) == 0:
                    insert_to.append([token_id])
                else:
                    insert_to[-1].append(token_id)
                return SplitTreeBuilder.ChangeStatus.IN_PROGRESS

            return self._on_next_token(
                token_id,
                on_end_of_placeholder=on_end_of_placeholder,
                on_end_of_node_children=on_end_of_node_children,
                on_structure_token=on_structure_token,
                on_placeholder_token=on_placeholder_token
            )

        def decode_if_add_token_id(self, token_id: int) -> List[str]:

            def on_end_of_placeholder():
                current = self._nodes[self._visit_stack[-1]]
                current_content = self._structure_decompression.get_content_fragments_for(current.node_type)
                return current_content.text_after_n_placeholders(len(current.placeholders))

            def on_end_of_node_children():
                if len(self._visit_stack) == 1:
                    return []
                pred = self._nodes[self._visit_stack[-2]]
                pred_content = self._structure_decompression.get_content_fragments_for(pred.node_type)
                if len(pred.children) <= pred_content.children:
                    return pred_content.text_after_n_children(len(pred.children))
                return []

            def on_structure_token(token_id):
                return self._structure_decompression.get_content_fragments_for(token_id).text_prefix()

            def on_placeholder_token(token_id):
                return self._placeholders_bpe.decode([token_id])

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
            placeholders_bpe: BPE
    ):
        self._fixed_order_version_ids = [0]
        self._versions = {0: SplitTreeBuilder.Version.empty_version(structure_decompression, placeholders_bpe)}
        self._next_version_id = 1
        self._structure_decompression = structure_decompression
        self._placeholders_bpe = placeholders_bpe

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

    def get_next_possible_ids(self, version_id) -> Set[int]:
        assert version_id in self._versions
        return self._versions[version_id].get_next_possible_ids()

    def add_token(self, version_id, token_id) -> ChangeStatus:
        assert version_id in self._versions
        return self._versions[version_id].add_token(token_id)

    def decode_if_add(self, version_id, token_id) -> List[str]:
        assert version_id in self._versions
        return self._versions[version_id].decode_if_add_token_id(token_id)

    def decode_ids_to_text(self, ids: List[int]) -> List[str]:
        empty_version = SplitTreeBuilder.Version.empty_version(self._structure_decompression, self._placeholders_bpe)
        res = []
        for token_id in ids:
            res.extend(empty_version.decode_if_add_token_id(token_id))
            empty_version.add_token(token_id)
        return res

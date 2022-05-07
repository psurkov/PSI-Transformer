from typing import List

from flccpsisrc.common.token_holder import TokenHolder
from flccpsisrc.psi.psi_datapoint.placeholders.placeholder_bpe import PlaceholderBpe
from flccpsisrc.psi.psi_datapoint.tree_structures.next_ids_suggester import NextIdsSuggester
from flccpsisrc.psi.psi_datapoint.tree_structures.split_tree_version import VersionsShared, Version, ChangeStatus
from flccpsisrc.psi.psi_datapoint.tree_structures.structure_decompression import StructureDecompression


class SplitTreeBuilder:

    def __init__(
            self,
            structure_decompression: StructureDecompression,
            placeholders_bpe: PlaceholderBpe,
            rollback_prefix_holder: TokenHolder
    ):
        self._versions_shared = VersionsShared(structure_decompression, placeholders_bpe)
        self._fixed_order_version_ids = [0]
        self._versions = {0: Version.empty_version(rollback_prefix_holder, self._versions_shared)}
        self._next_version_id = 1
        self._rollback_prefix_holder = rollback_prefix_holder
        self._next_ids_suggester = NextIdsSuggester(structure_decompression, placeholders_bpe)

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
        return self._next_ids_suggester.possible_next_id(self._versions[version_id])

    def add_token(self, version_id, token_id) -> ChangeStatus:
        assert version_id in self._versions
        return self._versions[version_id].add_token(token_id)

    def decode_if_add(self, version_id, token_id) -> TokenHolder:
        assert version_id in self._versions
        return self._versions[version_id].decode_if_add_token_id(token_id)

    def collect_full_token_holder(self, ids: List[int]) -> TokenHolder:
        new_version = Version.empty_version(
            self._rollback_prefix_holder,
            self._versions_shared
        )
        suggestion = TokenHolder.empty()
        for token_id in ids:
            suggestion.extend(new_version.decode_if_add_token_id(token_id))
            new_version.add_token(token_id)
        suggestion.remove_prefix(self._rollback_prefix_holder)
        return suggestion

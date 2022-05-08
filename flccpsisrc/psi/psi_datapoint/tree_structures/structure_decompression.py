import json
from enum import Enum
from typing import Optional, List


class NodeContentFragment:
    class FragmentType(Enum):
        TEXT = 0
        PLACEHOLDER = 1
        CHILD = 2

    def __init__(self, fragment_type, self_text: Optional[str], placeholder_type: Optional[str]):
        self._fragment_type = fragment_type
        self._self_text = self_text
        self._placeholder_type = placeholder_type

    @property
    def fragment_type(self) -> FragmentType:
        return self._fragment_type

    @property
    def self_text(self) -> str:
        return self._self_text

    @property
    def placeholder_type(self) -> str:
        return self._placeholder_type

    def __str__(self) -> str:
        if self.fragment_type == NodeContentFragment.FragmentType.TEXT:
            return self.self_text
        if self.fragment_type == NodeContentFragment.FragmentType.PLACEHOLDER:
            return "<placeholder>"
        return "<child>"


class NodeContentFragments:
    _child_fragment = NodeContentFragment(NodeContentFragment.FragmentType.CHILD, None, None)

    def __init__(self):
        self._fragments = []

    @property
    def fragments(self) -> List[NodeContentFragment]:
        return self._fragments

    @property
    def placeholders(self) -> int:
        return len([f for f in self.fragments if f.fragment_type == NodeContentFragment.FragmentType.PLACEHOLDER])

    @property
    def children(self) -> int:
        return len([f for f in self.fragments if f.fragment_type == NodeContentFragment.FragmentType.CHILD])

    @property
    def generation_places(self) -> int:
        return self.placeholders + self.children

    def generation_place(self, ind: int) -> NodeContentFragment:
        assert ind < self.generation_places
        return [f for f in self.fragments
                if f.fragment_type != NodeContentFragment.FragmentType.TEXT][ind]

    def text_prefix(self) -> List[str]:
        res = []
        for f in self.fragments:
            if f.fragment_type != NodeContentFragment.FragmentType.TEXT:
                break
            res.append(f.self_text)
        return res

    def text_after_n_placeholders(self, n: int) -> List[str]:
        res = []
        was_placeholders = 0
        for f in self.fragments:
            if was_placeholders >= n:
                if f.fragment_type == NodeContentFragment.FragmentType.TEXT:
                    res.append(f.self_text)
                else:
                    break
            if f.fragment_type == NodeContentFragment.FragmentType.PLACEHOLDER:
                was_placeholders += 1
        assert was_placeholders >= n
        return res

    def text_after_n_children(self, n: int) -> List[str]:
        res = []
        was_children = 0
        for f in self.fragments:
            if was_children >= n:
                if f.fragment_type == NodeContentFragment.FragmentType.TEXT:
                    res.append(f.self_text)
                else:
                    break
            if f.fragment_type == NodeContentFragment.FragmentType.CHILD:
                was_children += 1
        assert was_children >= n
        return res

    def append_new_text_fragment(self, text: str) -> "NodeContentFragments":
        self._fragments.append(NodeContentFragment(NodeContentFragment.FragmentType.TEXT, text, None))
        return self

    def append_new_placeholder_fragment(self, placeholder_type) -> "NodeContentFragments":
        fragment = NodeContentFragment(NodeContentFragment.FragmentType.PLACEHOLDER, None, placeholder_type)
        self._fragments.append(fragment)
        return self

    def append_new_child_fragment(self) -> "NodeContentFragments":
        self._fragments.append(NodeContentFragments._child_fragment)
        return self

    def append(self, fragment: NodeContentFragment) -> "NodeContentFragments":
        self._fragments.append(fragment)
        return self

    def extend(self, other: "NodeContentFragments") -> "NodeContentFragments":
        self._fragments.extend(other.fragments)
        return self

    def __str__(self) -> str:
        return " ".join(map(lambda fragment: fragment.__str__(), self.fragments))


class StructureDecompression:

    def __init__(self, type_coder_data: str, structure_compression_data: str):
        self._can_terminate_if_start_generate = []
        self._id_to_content_fragments = {0: NodeContentFragments()}
        with open(type_coder_data) as f:
            coder_data = json.load(f)
            self._can_terminate_if_start_generate = [coder_data["structureTypenameToType"]["LBRACE"]["id"]]
            for placeholder in coder_data["placeholderTypenameToType"]:
                placeholder_id = coder_data["placeholderTypenameToType"][placeholder]["id"]
                self._id_to_content_fragments[placeholder_id] = NodeContentFragments() \
                    .append_new_placeholder_fragment(placeholder)
            for structure in coder_data["structureTypenameToType"]:
                structure_id = coder_data["structureTypenameToType"][structure]["id"]
                structure_text = coder_data["structureTypenameToType"][structure]["text"]
                self._id_to_content_fragments[structure_id] = NodeContentFragments()
                if len(structure_text) > 0:
                    self._id_to_content_fragments[structure_id].append_new_text_fragment(structure_text)

        with open(structure_compression_data) as f:
            compression_data = json.load(f)
            self._vocab_size = compression_data["vocabSize"]
            for rule in compression_data["rules"]:
                upper_type_id = rule["upperTypeId"]
                lower_type_id = rule["lowerTypeId"]
                lower_index_in_upper = rule["lowerIndexInUpper"]
                children_number_in_lower = rule["childrenNumberInLower"]
                new_type_id = rule["newTypeId"]

                upper_fragments = self._id_to_content_fragments[upper_type_id]

                complete_lower_fragments = NodeContentFragments().extend(self._id_to_content_fragments[lower_type_id])
                need_extra_children = children_number_in_lower - self._id_to_content_fragments[lower_type_id].children
                assert need_extra_children >= 0
                for i in range(need_extra_children):
                    complete_lower_fragments.append_new_child_fragment()

                new_fragments = NodeContentFragments()
                was_child_fragments_in_upper = 0
                for upper_fragment in upper_fragments.fragments:
                    if upper_fragment.fragment_type == NodeContentFragment.FragmentType.CHILD:
                        if was_child_fragments_in_upper == lower_index_in_upper:
                            new_fragments.extend(complete_lower_fragments)
                        else:
                            new_fragments.append(upper_fragment)
                        was_child_fragments_in_upper += 1
                    else:
                        new_fragments.append(upper_fragment)
                if was_child_fragments_in_upper <= lower_index_in_upper:
                    for _ in range(lower_index_in_upper - was_child_fragments_in_upper):
                        new_fragments.append_new_child_fragment()
                    new_fragments.extend(complete_lower_fragments)
                self._id_to_content_fragments[new_type_id] = new_fragments

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def can_terminate_if_start_generate(self) -> List[int]:
        return self._can_terminate_if_start_generate

    def get_content_fragments_for(self, node_type_id) -> NodeContentFragments:
        return self._id_to_content_fragments[node_type_id]

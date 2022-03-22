from enum import Enum

SPECIAL_IDS_RESERVED_SIZE = 5


# must be in [0-SPECIAL_IDS_RESERVED_SIZE)
class SpecialIds(Enum):
    END_OF_PLACEHOLDER = 0
    END_OF_NODE_CHILDREN = 1

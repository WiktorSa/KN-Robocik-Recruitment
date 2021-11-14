from enum import Enum, unique


@unique
class GateEnum(Enum):
    """
    Enum that maps gate location to it's code
    """

    fully_visible = 0
    up = 1
    right = 2
    down = 3
    left = 4

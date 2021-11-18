from enum import Enum, unique


@unique
class GateEnum(Enum):
    """
    Enum that maps gate location to it's code
    Note - codes will be used for classification so they all need to be integers starting from 0
    """

    fully_visible = 0
    up = 1
    right = 2
    down = 3
    left = 4

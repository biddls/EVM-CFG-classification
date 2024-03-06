"""
Stores the type for the node in a typed dictionary
"""
from typing import TypedDict

class CFG_Node(TypedDict):
    parsedOpcode: list[str]
    index: int

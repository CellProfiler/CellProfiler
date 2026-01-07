from enum import Enum

class BackgroundExclusionMode(str, Enum):
    """Background exclusion method"""
    THRESHOLD = "Threshold"
    MASK = "Mask"
    OBJECTS = "Within Objects"
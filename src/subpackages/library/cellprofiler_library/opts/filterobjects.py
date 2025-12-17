from enum import Enum

class FilterMethod(str, Enum):
    """Minimal filter - pick a single object per image by minimum measured value"""
    MINIMAL = "Minimal"

    """Maximal filter - pick a single object per image by maximum measured value"""
    MAXIMAL = "Maximal"

    """Pick one object per containing object by minimum measured value"""
    MINIMAL_PER_OBJECT = "Minimal per object"

    """Pick one object per containing object by maximum measured value"""
    MAXIMAL_PER_OBJECT = "Maximal per object"

    """Keep all objects whose values fall between set limits"""
    LIMITS = "Limits"

FI_ALL = [
    FilterMethod.MINIMAL,
    FilterMethod.MAXIMAL,
    FilterMethod.MINIMAL_PER_OBJECT,
    FilterMethod.MAXIMAL_PER_OBJECT,
    FilterMethod.LIMITS,
]

class FilterMode(str, Enum):
    RULES = "Rules"
    CLASSIFIERS = "Classifiers"
    MEASUREMENTS = "Measurements"
    BORDER = "Image or mask border"

DIR_CUSTOM = "Custom folder"

class OverlapAssignment(str, Enum):
    BOTH = "Both parents"
    PARENT_WITH_MOST_OVERLAP = "Parent with most overlap"
PO_ALL = [OverlapAssignment.BOTH, OverlapAssignment.PARENT_WITH_MOST_OVERLAP]

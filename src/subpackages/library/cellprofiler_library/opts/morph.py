from enum import Enum


class MorphFunction(str, Enum):
    """Morphological operations available in the Morph module."""
    BRANCHPOINTS = "branchpoints"
    BRIDGE = "bridge"
    CLEAN = "clean"
    CONVEX_HULL = "convex hull"
    DIAG = "diag"
    DISTANCE = "distance"
    ENDPOINTS = "endpoints"
    FILL = "fill"
    HBREAK = "hbreak"
    MAJORITY = "majority"
    OPENLINES = "openlines"
    REMOVE = "remove"
    SHRINK = "shrink"
    SKELPE = "skelpe"
    SPUR = "spur"
    THICKEN = "thicken"
    THIN = "thin"
    VBREAK = "vbreak"


class RepeatMethod(str, Enum):
    """Options for how many times to repeat morphological operations."""
    ONCE = "Once"
    FOREVER = "Forever"
    CUSTOM = "Custom"

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

F_ALL = [
    MorphFunction.BRANCHPOINTS.value,
    MorphFunction.BRIDGE.value,
    MorphFunction.CLEAN.value,
    MorphFunction.CONVEX_HULL.value,
    MorphFunction.DIAG.value,
    MorphFunction.DISTANCE.value,
    MorphFunction.ENDPOINTS.value,
    MorphFunction.FILL.value,
    MorphFunction.HBREAK.value,
    MorphFunction.MAJORITY.value,
    MorphFunction.OPENLINES.value,
    MorphFunction.REMOVE.value,
    MorphFunction.SHRINK.value,
    MorphFunction.SKELPE.value,
    MorphFunction.SPUR.value,
    MorphFunction.THICKEN.value,
    MorphFunction.THIN.value,
    MorphFunction.VBREAK.value,
]

R_ALL = [
    RepeatMethod.ONCE.value, 
    RepeatMethod.FOREVER.value, 
    RepeatMethod.CUSTOM.value
]
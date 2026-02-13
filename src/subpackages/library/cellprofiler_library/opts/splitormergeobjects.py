from enum import Enum

class RelabelOption(str, Enum):
    MERGE = "Merge"
    SPLIT = "Split"

class MergeOption(str, Enum):
    UNIFY_DISTANCE = "Distance"
    UNIFY_PARENT = "Per-parent"

class MergingMethod(str, Enum):
    DISCONNECTED = "Disconnected"
    CONVEX_HULL = "Convex hull"

class ObjectIntensityMethod(str, Enum):
    CLOSEST_POINT = "Closest point"
    CENTROIDS = "Centroids"

C_PARENT = "Parent"
from enum import Enum

class OperationMethod(str, Enum):
    ENHANCE = "Enhance"
    SUPPRESS = "Suppress"

class EnhanceMethod(str, Enum):
    SPECKLES = "Speckles"
    NEURITES = "Neurites"
    DARK_HOLES = "Dark holes"
    CIRCLES = "Circles"
    TEXTURE = "Texture"
    DIC = "DIC"

class SpeckleAccuracy(str, Enum):
    SLOW = "Slow"
    FAST = "Fast"

class NeuriteMethod(str, Enum):
    GRADIENT = "Line structures"
    TUBENESS = "Tubeness"

from enum import Enum

class SecondaryObjectMethod(str, Enum):
    PROPAGATION = "Propagation"
    WATERSHED_GRADIENT = "Watershed - Gradient"
    WATERSHED_IMAGE = "Watershed - Image"
    DISTANCE_N = "Distance - N"
    DISTANCE_B = "Distance - B"
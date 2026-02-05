from enum import Enum

class DistanceMethod(str, Enum):
    ADJACENT = "Adjacent"
    EXPAND = "Expand until adjacent"
    WITHIN = "Within a specified distance"

class Measurement(str, Enum):
    NUMBER_OF_NEIGHBORS = "NumberOfNeighbors"
    PERCENT_TOUCHING = "PercentTouching"
    FIRST_CLOSEST_OBJECT_NUMBER = "FirstClosestObjectNumber"
    FIRST_CLOSEST_DISTANCE = "FirstClosestDistance"
    SECOND_CLOSEST_OBJECT_NUMBER = "SecondClosestObjectNumber"
    SECOND_CLOSEST_DISTANCE = "SecondClosestDistance"
    ANGLE_BETWEEN_NEIGHBORS = "AngleBetweenNeighbors"

D_ALL = [
    DistanceMethod.ADJACENT.value,
    DistanceMethod.EXPAND.value,
    DistanceMethod.WITHIN.value,
]

M_ALL = [
    Measurement.NUMBER_OF_NEIGHBORS.value,
    Measurement.PERCENT_TOUCHING.value,
    Measurement.FIRST_CLOSEST_OBJECT_NUMBER.value,
    Measurement.FIRST_CLOSEST_DISTANCE.value,
    Measurement.SECOND_CLOSEST_OBJECT_NUMBER.value,
    Measurement.SECOND_CLOSEST_DISTANCE.value,
    Measurement.ANGLE_BETWEEN_NEIGHBORS.value,
]

class MeasurementScale(str, Enum):
    EXPANDED = "Expanded"
    ADJACENT = "Adjacent"

C_NEIGHBORS = "Neighbors"

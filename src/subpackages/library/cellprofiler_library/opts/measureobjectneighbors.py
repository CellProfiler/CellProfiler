from enum import Enum

class DistanceMethod(str, Enum):
    ADJACENT = "Adjacent"
    EXPAND = "Expand until adjacent"
    WITHIN = "Within a specified distance"
    def __str__(self):
        return self.value

class Measurement(str, Enum):
    NUMBER_OF_NEIGHBORS = "NumberOfNeighbors"
    PERCENT_TOUCHING = "PercentTouching"
    FIRST_CLOSEST_OBJECT_NUMBER = "FirstClosestObjectNumber"
    FIRST_CLOSEST_DISTANCE = "FirstClosestDistance"
    SECOND_CLOSEST_OBJECT_NUMBER = "SecondClosestObjectNumber"
    SECOND_CLOSEST_DISTANCE = "SecondClosestDistance"
    ANGLE_BETWEEN_NEIGHBORS = "AngleBetweenNeighbors"
    def __str__(self):
        return self.value

D_ALL = [
    DistanceMethod.ADJACENT,
    DistanceMethod.EXPAND,
    DistanceMethod.WITHIN,
]

M_ALL = [
    Measurement.NUMBER_OF_NEIGHBORS,
    Measurement.PERCENT_TOUCHING,
    Measurement.FIRST_CLOSEST_OBJECT_NUMBER,
    Measurement.FIRST_CLOSEST_DISTANCE,
    Measurement.SECOND_CLOSEST_OBJECT_NUMBER,
    Measurement.SECOND_CLOSEST_DISTANCE,
    Measurement.ANGLE_BETWEEN_NEIGHBORS,
]

class MeasurementScale(str, Enum):
    EXPANDED = "Expanded"
    ADJACENT = "Adjacent"
    def __str__(self):
        return self.value

C_NEIGHBORS = "Neighbors"

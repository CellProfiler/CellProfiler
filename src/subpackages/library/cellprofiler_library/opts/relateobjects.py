from enum import Enum

class DistanceMethod(str, Enum):
    NONE = "None"
    CENTROID = "Centroid"
    MINIMUM = "Minimum"
    BOTH = "Both"

class TemplateMeasurementFormat(str):
    FF_PARENT = "Parent_%s"
    FF_CHILDREN_COUNT = "Children_%s_Count"
    # Centroid distance measurement (FF_DISTANCE % parent)
    FF_CENTROID = "Distance_Centroid_%s"
    # Minimum distance measurement (FF_MINIMUM % parent)
    FF_MINIMUM = "Distance_Minimum_%s"
    FF_MEAN = "Mean_%s_%s"

class Relationship(str, Enum):
    PARENT = "Parent" 
    CHILD = "Children"

C_MEAN = "Mean"
C_PARENT = "Parent"
C_CHILDREN = "Children"
M_NUMBER_OBJECT_NUMBER = "Number_ObjectNumber"

from enum import Enum

class DistanceMethod(str, Enum):
    NONE = "None"
    CENTROID = "Centroid"
    MINIMUM = "Minimum"
    BOTH = "Both"

class TemplateMeasurementFormat(str):
    FF_PARENT = "Parent_%s"
    FF_CHILDREN_COUNT = "Children_%s_Count"
    FF_CENTROID = "Distance_Centroid_%s"
    FF_MINIMUM = "Distance_Minimum_%s"
    FF_MEAN = "Mean_%s_%s"

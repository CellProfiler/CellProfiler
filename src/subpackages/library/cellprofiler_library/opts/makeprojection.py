from enum import Enum

class ProjectionType(str, Enum):
    AVERAGE = "Average"
    MAXIMUM = "Maximum"
    MINIMUM = "Minimum"
    SUM = "Sum"
    VARIANCE = "Variance"
    POWER = "Power"
    BRIGHTFIELD = "Brightfield"
    MASK = "Mask"

P_ALL = [
    ProjectionType.AVERAGE.value,
    ProjectionType.MAXIMUM.value,
    ProjectionType.MINIMUM.value,
    ProjectionType.SUM.value,
    ProjectionType.VARIANCE.value,
    ProjectionType.POWER.value,
    ProjectionType.BRIGHTFIELD.value,
    ProjectionType.MASK.value,
]

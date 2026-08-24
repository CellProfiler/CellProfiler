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


class StateKey(str, Enum):
    IMAGE = "image"
    IMAGE_COUNT = "image_count"
    VSUM = "vsum"
    VSQUARED = "vsquared"
    POWER_IMAGE = "power_image"
    POWER_MASK = "power_mask"
    STACK_NUMBER = "stack_number"
    BRIGHT_MAX = "bright_max"
    BRIGHT_MIN = "bright_min"
    NORM0 = "norm0"
    

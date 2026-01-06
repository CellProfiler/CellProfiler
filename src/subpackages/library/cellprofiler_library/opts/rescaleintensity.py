from enum import Enum

class RescaleMethod(str, Enum):

    STRETCH = "Stretch each image to use the full intensity range"
    MANUAL_INPUT_RANGE = "Choose specific values to be reset to the full intensity range"
    MANUAL_IO_RANGE = "Choose specific values to be reset to a custom range"
    DIVIDE_BY_IMAGE_MINIMUM = "Divide by the image's minimum"
    DIVIDE_BY_IMAGE_MAXIMUM = "Divide by the image's maximum"
    DIVIDE_BY_VALUE = "Divide each image by the same value"
    DIVIDE_BY_MEASUREMENT = "Divide each image by a previously calculated value"
    SCALE_BY_IMAGE_MAXIMUM = "Match the image's maximum to another image's maximum"

M_ALL = [
    RescaleMethod.STRETCH,
    RescaleMethod.MANUAL_INPUT_RANGE,
    RescaleMethod.MANUAL_IO_RANGE,
    RescaleMethod.DIVIDE_BY_IMAGE_MINIMUM,
    RescaleMethod.DIVIDE_BY_IMAGE_MAXIMUM,
    RescaleMethod.DIVIDE_BY_VALUE,
    RescaleMethod.DIVIDE_BY_MEASUREMENT,
    RescaleMethod.SCALE_BY_IMAGE_MAXIMUM,
]

# R_SCALE = "Scale similarly to others"
# R_MASK = "Mask pixels"
# R_SET_TO_ZERO = "Set to zero"
# R_SET_TO_CUSTOM = "Set to custom value"
# R_SET_TO_ONE = "Set to one"
class MinimumIntensityMethod(str, Enum):
    CUSTOM_VALUE = "Custom"
    ALL_IMAGES = "Minimum of all images"
    EACH_IMAGE = "Minimum for each image"

LOW_ALL = [MinimumIntensityMethod.CUSTOM_VALUE, MinimumIntensityMethod.EACH_IMAGE, MinimumIntensityMethod.ALL_IMAGES]

class MaximumIntensityMethod(str, Enum):
    CUSTOM_VALUE = "Custom"
    ALL_IMAGES = "Maximum of all images"
    EACH_IMAGE = "Maximum for each image"

HIGH_ALL = [MaximumIntensityMethod.CUSTOM_VALUE, MaximumIntensityMethod.EACH_IMAGE, MaximumIntensityMethod.ALL_IMAGES]
# LOW_ALL_IMAGES = "Minimum of all images"
# LOW_EACH_IMAGE = "Minimum for each image"
# CUSTOM_VALUE = "Custom"
# LOW_ALL = [CUSTOM_VALUE, LOW_EACH_IMAGE, LOW_ALL_IMAGES]

# HIGH_ALL_IMAGES = "Maximum of all images"
# HIGH_EACH_IMAGE = "Maximum for each image"

# HIGH_ALL = [CUSTOM_VALUE, HIGH_EACH_IMAGE, HIGH_ALL_IMAGES]
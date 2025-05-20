from enum import Enum

# None of these are used in cellprofiler library at the moment
# Keeping this here in case we would like to replace the const's
# on the frontend module file with these.
class ConversionMethod(str, Enum):
    COMBINE = "combine"
    SPLIT = "split"


class ImageChannelType(str, Enum):
    RGB = "rgb"
    HSV = "hsv"
    CHANNELS = "channels"

class Channel(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    HUE = "hue"
    SATURATION = "saturation"
    VALUE = "value"
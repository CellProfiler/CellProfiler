from enum import Enum

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

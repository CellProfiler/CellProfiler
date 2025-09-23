from enum import Enum

class ConversionMethod(str, Enum):
    COMBINE = "Combine"
    SPLIT = "Split"


class ImageChannelType(str, Enum):
    RGB = "RGB"
    HSV = "HSV"
    CHANNELS = "Channels"

class Channel(str, Enum):
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    HUE = "Hue"
    SATURATION = "Saturation"
    VALUE = "Value"

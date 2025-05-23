from enum import Enum

class ImageMode(str, Enum):
    BINARY = "Binary (black & white)"
    GRAYSCALE = "Grayscale"
    UINT16 = "uint16"
    COLOR = "Color"

DEFAULT_COLORMAP = "Default"
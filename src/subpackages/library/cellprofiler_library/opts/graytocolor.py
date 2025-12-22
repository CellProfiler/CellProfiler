from enum import Enum

class Scheme(str, Enum):
    RGB = "RGB"
    CMYK = "CMYK"
    STACK = "Stack"
    COMPOSITE = "Composite"
    LEAVE_THIS_BLACK = "Leave this black"

DEFAULT_COLORS = [
    "#%02x%02x%02x" % color
    for color in (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
    )
]
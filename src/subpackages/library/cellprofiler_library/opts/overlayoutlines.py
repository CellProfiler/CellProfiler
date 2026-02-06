from enum import Enum

class BrightnessMode(str, Enum):
    MAX_IMAGE = "Max of image"
    MAX_POSSIBLE = "Max possible"


class LineMode(str, Enum):
    INNER = "Inner"
    OUTER = "Outer"
    THICK = "Thick"


class SourceMode(str, Enum):
    FROM_IMAGES = "Image"
    FROM_OBJECTS = "Objects"

class OutlineMode(str, Enum):
    COLOR = "Color"
    GRAYSCALE = "Grayscale"


# Color definitions - preserved exactly from original module
COLORS = {
    "White": (1, 1, 1),
    "Black": (0, 0, 0),
    "Red": (1, 0, 0),
    "Green": (0, 1, 0),
    "Blue": (0, 0, 1),
    "Yellow": (1, 1, 0),
}

# Color order - preserved exactly from original module
COLOR_ORDER = ["Red", "Green", "Blue", "Yellow", "White", "Black"]

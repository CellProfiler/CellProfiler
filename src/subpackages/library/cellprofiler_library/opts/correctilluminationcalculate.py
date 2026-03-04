from enum import Enum
from centrosome.bg_compensate import MODE_AUTO, MODE_DARK, MODE_BRIGHT, MODE_GRAY
class IntensityChoice(str, Enum):
    """Choice of how to calculate the illumination function"""
    REGULAR = "Regular"
    BACKGROUND = "Background"

class RescaleIlluminationFunction(str, Enum):
    """The illumination function can be rescaled so that the pixel intensities
    are all equal to or greater than 1. You have the following options:"""
    YES = "Yes"
    NO = "No"
    MEDIAN = "Median"

class CalculateFunctionTarget(str, Enum):
    """Calculate function for each image individually, or based on all images?"""
    EACH = "Each"
    # ALL = "All"
    ALL_FIRST = "All: First cycle"
    ALL_ACROSS = "All: Across cycles"

class SmoothingMethod(str, Enum):
    """Smoothing method"""
    NONE = "No smoothing"
    CONVEX_HULL = "Convex Hull"
    FIT_POLYNOMIAL = "Fit Polynomial"
    MEDIAN_FILTER = "Median Filter"
    GAUSSIAN_FILTER = "Gaussian Filter"
    TO_AVERAGE = "Smooth to Average"
    SPLINES = "Splines"

class SmoothingFilterSize(str, Enum):
    """Smoothing filter size"""
    AUTOMATIC = "Automatic"
    OBJECT_SIZE = "Object size"
    MANUALLY = "Manually"

class SplineBackgroundMode(str, Enum):
    """Background mode"""
    AUTO = MODE_AUTO
    DARK = MODE_DARK
    BRIGHT = MODE_BRIGHT
    GRAY = MODE_GRAY

class StateKey(str, Enum):
    """Keys for the illumination accumulation state dictionary"""
    IMAGE_SUM = "image_sum"
    MASK_COUNT = "mask_count"


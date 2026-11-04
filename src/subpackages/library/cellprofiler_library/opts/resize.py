"""Resize module enums and configuration options."""

from enum import Enum


class ResizingMethod(str, Enum):
    """Method for specifying resize dimensions."""
    BY_FACTOR = "Resize by a fraction or multiple of the original size"
    TO_SIZE = "Resize by specifying desired final dimensions"


class DimensionMethod(str, Enum):
    """Method for specifying target dimensions when resizing to size."""
    MANUAL = "Manual"
    IMAGE = "Image"


class InterpolationMethod(str, Enum):
    """Interpolation method for resize operation."""
    NEAREST_NEIGHBOR = "Nearest Neighbor"
    BILINEAR = "Bilinear"
    BICUBIC = "Bicubic"

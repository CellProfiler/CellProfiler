from enum import Enum

class Shape(str, Enum):
    RECTANGLE = "Rectangle"
    ELLIPSE = "Ellipse"
    IMAGE = "Image"
    OBJECTS = "Objects"
    CROPPING = "Previous cropping"

class RemovalMethod(str, Enum):
    NO = "No"
    EDGES = "Edges"
    ALL = "All"

class Measurement(str, Enum):
    AREA_RETAINED = "Crop_AreaRetainedAfterCropping_%s"
    ORIGINAL_AREA = "Crop_OriginalImageArea_%s"

class CroppingMethod(str, Enum):
    COORDINATES = "Coordinates"
    MOUSE = "Mouse"

class CroppingPattern(str, Enum):
    FIRST = "First"
    INDIVIDUALLY = "Individually"

class Limits(str, Enum):
    ABSOLUTE = "Absolute"
    FROM_EDGE = "From edge"

class Ellipse(str, Enum):
    XCENTER = "xcenter"
    YCENTER = "ycenter"
    XRADIUS = "xradius"
    YRADIUS = "yradius"

class Rectangle(str, Enum):
    LEFT = "left"
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"

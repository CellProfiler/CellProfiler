from enum import Enum

class FlipDirection(str, Enum):
    NONE = "Do not flip"
    LEFT_TO_RIGHT = "Left to right"
    TOP_TO_BOTTOM = "Top to bottom"
    BOTH = "Left to right and top to bottom"

class RotateDirection(str, Enum):
    NONE = "Do not rotate"
    ANGLE = "Enter angle"
    COORDINATES = "Enter coordinates"
    MOUSE = "Use mouse"

class RotationCycle(str, Enum):
    INDIVIDUALLY = "Individually"
    ONCE = "Only Once"

class RotationCoordinateAlignmnet(str, Enum):
    HORIZONTALLY = "Horizontally"
    VERTICALLY = "Vertically"

D_ANGLE = "angle"

"""Rotation measurement category"""
M_ROTATION_CATEGORY = "Rotation"
"""Rotation measurement format (+ image name)"""
M_ROTATION_F = "%s_%%s" % M_ROTATION_CATEGORY

FLIP_ALL = [FlipDirection.NONE, FlipDirection.LEFT_TO_RIGHT, FlipDirection.TOP_TO_BOTTOM, FlipDirection.BOTH]

ROTATE_ALL = [RotateDirection.NONE, RotateDirection.ANGLE, RotateDirection.COORDINATES, RotateDirection.MOUSE]

IO_ALL = [RotationCycle.INDIVIDUALLY, RotationCycle.ONCE]

C_ALL = [RotationCoordinateAlignmnet.HORIZONTALLY, RotationCoordinateAlignmnet.VERTICALLY]
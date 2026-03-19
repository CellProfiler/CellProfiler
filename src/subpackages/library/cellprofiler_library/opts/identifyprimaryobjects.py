from enum import Enum

class UnclumpMethod(str, Enum):
    INTENSITY = "Intensity"
    SHAPE = "Shape"
    NONE = "None"

class WatershedMethod(str, Enum):
    INTENSITY = "Intensity"
    SHAPE = "Shape"
    PROPAGATE = "Propagate"
    NONE = "None"

class FillHolesMethod(str, Enum):
    NEVER = "Never"
    THRESHOLDING = "After both thresholding and declumping"
    DECLUMP = "After declumping only"
FH_ALL = (FillHolesMethod.NEVER.value, FillHolesMethod.THRESHOLDING.value, FillHolesMethod.DECLUMP.value)

class LimitObjects(str, Enum):
    NONE = "Continue"
    ERASE = "Erase"

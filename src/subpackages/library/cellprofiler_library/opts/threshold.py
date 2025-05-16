from enum import Enum

class Scope(str, Enum):
    GLOBAL = "global"
    ADAPTIVE = "adaptive"

class Method(str, Enum):
    OTSU = "otsu"
    MINIMUM_CROSS_ENTROPY = "minimum_cross_entropy"
    ROBUST_BACKGROUND = "robust_background"
    MULTI_OTSU = "multiotsu"
    SAUVOLA = "sauvola"

class Assignment(str, Enum):
    # assign_middle_to_foreground
    FOREGROUND = "foreground"
    BACKGROUND = "background"

class AveragingMethod(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"

class VarianceMethod(str, Enum):
    STANDARD_DEVIATION = "standard_deviation"
    MEDIAN_ABSOLUTE_DEVIATION = "median_absolute_deviation"
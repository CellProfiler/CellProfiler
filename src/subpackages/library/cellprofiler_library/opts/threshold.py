from enum import Enum

class Scope(str, Enum):
    GLOBAL = "global"
    ADAPTIVE = "adaptive"

class OtsuMethod(str, Enum):
    TWO_CLASS = "two_classes"
    THREE_CLASS = "three_classes"

class Method(str, Enum):
    OTSU = "otsu"
    MINIMUM_CROSS_ENTROPY = "minimum_cross_entropy"
    ROBUST_BACKGROUND = "robust_background"
    MULTI_OTSU = "multi_otsu"
    SAUVOLA = "sauvola"
    MAX_INTENSITY_PERCENTAGE = "max_intensity_percentage" # For MeasureColocalization
    MANUAL = "manual" # For IdentifyPrimaryObjects
    MEASUREMENT = "measurement" # For IdentifyPrimaryObjects

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

class OTSUThresholdMethod(str, Enum):
    TWO_CLASS = "two_classes"
    THREE_CLASS = "three_classes"
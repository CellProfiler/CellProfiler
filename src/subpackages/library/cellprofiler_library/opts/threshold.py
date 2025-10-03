from enum import Enum

class Scope(str, Enum):
    GLOBAL = "Global"
    ADAPTIVE = "Adaptive"

class OtsuMethod(str, Enum):
    TWO_CLASS = "Two classes"
    THREE_CLASS = "Three classes"

class Method(str, Enum):
    OTSU = "Otsu"
    MINIMUM_CROSS_ENTROPY = "Minimum Cross-Entropy"
    ROBUST_BACKGROUND = "Robust Background"
    MULTI_OTSU = "Multi-Otsu"
    SAUVOLA = "Sauvola"
    MAX_INTENSITY_PERCENTAGE = "Max Intensity Percentage" # For MeasureColocalization
    MANUAL = "Manual" # For IdentifyPrimaryObjects
    MEASUREMENT = "Measurement" # For IdentifyPrimaryObjects

class Assignment(str, Enum):
    # assign_middle_to_foreground
    FOREGROUND = "Foreground"
    BACKGROUND = "Background"

class AveragingMethod(str, Enum):
    MEAN = "Mean"
    MEDIAN = "Median"
    MODE = "Mode"

class VarianceMethod(str, Enum):
    STANDARD_DEVIATION = "Standard deviation"
    MEDIAN_ABSOLUTE_DEVIATION = "Median absolute deviation"

from enum import Enum

class Target(str, Enum):
    IMAGES = "Across entire image"
    OBJECTS = "Within objects"
    IMAGES_AND_OBJECTS = "Both"

class CostesMethod(str, Enum):
    FAST = "Fast"
    FASTER = "Faster"
    ACCURATE = "Accurate"

class CorrelationMethod(str, Enum):
    LINEAR = "Linear"
    BISECTION = "Bisection"

class MeasurementFormat(str, Enum):
    CORRELATION_FORMAT = "Correlation_Correlation_%s_%s"
    SLOPE_FORMAT = "Correlation_Slope_%s_%s"
    OVERLAP_FORMAT = "Correlation_Overlap_%s_%s"
    K_FORMAT = "Correlation_K_%s_%s"
    KS_FORMAT = "Correlation_KS_%s_%s"
    MANDERS_FORMAT = "Correlation_Manders_%s_%s"
    RWC_FORMAT = "Correlation_RWC_%s_%s"
    COSTES_FORMAT = "Correlation_Costes_%s_%s"

class MeasurementType(str, Enum):
    CORRELATION = "Correlation"
    OVERLAP = "Overlap"
    MANDERS = "Manders"
    RWC = "RWC"
    COSTES = "Costes"
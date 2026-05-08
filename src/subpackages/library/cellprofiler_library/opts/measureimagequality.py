from enum import Enum
from centrosome.threshold import (
    TM_OTSU,
    TM_MOG,
    TM_METHODS,
)
from .threshold import OtsuMethod as ThreshOtsuMethod

"""Root module measurement name"""
C_IMAGE_QUALITY = "ImageQuality"

C_SCALING = "Scaling"

class Feature(str, Enum):
    FOCUS_SCORE = "FocusScore"
    LOCAL_FOCUS_SCORE = "LocalFocusScore"
    CORRELATION = "Correlation"
    POWER_SPECTRUM_SLOPE = "PowerLogLogSlope"
    TOTAL_AREA = "TotalArea"
    TOTAL_VOLUME = "TotalVolume"
    TOTAL_INTENSITY = "TotalIntensity"
    MEAN_INTENSITY = "MeanIntensity"
    MEDIAN_INTENSITY = "MedianIntensity"
    STD_INTENSITY = "StdIntensity"
    MAD_INTENSITY = "MADIntensity"
    MAX_INTENSITY = "MaxIntensity"
    MIN_INTENSITY = "MinIntensity"
    PERCENT_MAXIMAL = "PercentMaximal"
    PERCENT_MINIMAL = "PercentMinimal"
    THRESHOLD = "Threshold"

class TemplateMeasurementFormat(str):
    """A string subclass for measurement formatting."""
    # IQ = Image Quality
    IQ_SCALING = f"{C_IMAGE_QUALITY}_{C_SCALING}_%s"
    IQ_FOCUS_SCORE = f"{C_IMAGE_QUALITY}_{Feature.FOCUS_SCORE.value}_%s"
    IQ_LOCAL_FOCUS_SCORE = f"{C_IMAGE_QUALITY}_{Feature.LOCAL_FOCUS_SCORE.value}_%s"
    IQ_CORR = f"{C_IMAGE_QUALITY}_{Feature.CORRELATION.value}_%s"
    IQ_POWER_SPECTRUM_SLOPE = f"{C_IMAGE_QUALITY}_{Feature.POWER_SPECTRUM_SLOPE.value}_%s"
    IQ_PERCENT_MAXIMAL = f"{C_IMAGE_QUALITY}_{Feature.PERCENT_MAXIMAL.value}_%s"
    IQ_PERCENT_MINIMAL = f"{C_IMAGE_QUALITY}_{Feature.PERCENT_MINIMAL.value}_%s"
    IQ_TOTAL_AREA = f"{C_IMAGE_QUALITY}_{Feature.TOTAL_AREA.value}_%s"
    IQ_TOTAL_VOLUME = f"{C_IMAGE_QUALITY}_{Feature.TOTAL_VOLUME.value}_%s"
    IQ_TOTAL_INTENSITY = f"{C_IMAGE_QUALITY}_{Feature.TOTAL_INTENSITY.value}_%s"
    IQ_MEAN_INTENSITY = f"{C_IMAGE_QUALITY}_{Feature.MEAN_INTENSITY.value}_%s"
    IQ_MEDIAN_INTENSITY = f"{C_IMAGE_QUALITY}_{Feature.MEDIAN_INTENSITY.value}_%s"
    IQ_STD_INTENSITY = f"{C_IMAGE_QUALITY}_{Feature.STD_INTENSITY.value}_%s"
    IQ_MAD_INTENSITY = f"{C_IMAGE_QUALITY}_{Feature.MAD_INTENSITY.value}_%s"
    IQ_MAX_INTENSITY = f"{C_IMAGE_QUALITY}_{Feature.MAX_INTENSITY.value}_%s"
    IQ_MIN_INTENSITY = f"{C_IMAGE_QUALITY}_{Feature.MIN_INTENSITY.value}_%s"
    IQ_THRESHOLD = f"{C_IMAGE_QUALITY}_{Feature.THRESHOLD.value}%s"

class Aggregate(str, Enum):
    MEAN = "Mean"
    MEDIAN = "Median"
    STD = "Std" 
INTENSITY_FEATURES = [
    Feature.TOTAL_INTENSITY,
    Feature.MEAN_INTENSITY,
    Feature.MEDIAN_INTENSITY,
    Feature.STD_INTENSITY,
    Feature.MAD_INTENSITY,
    Feature.MAX_INTENSITY,
    Feature.MIN_INTENSITY,
]
INTENSITY_TEMPLATES = [
    TemplateMeasurementFormat.IQ_TOTAL_INTENSITY,
    TemplateMeasurementFormat.IQ_MEAN_INTENSITY,
    TemplateMeasurementFormat.IQ_MEDIAN_INTENSITY,
    TemplateMeasurementFormat.IQ_STD_INTENSITY,
    TemplateMeasurementFormat.IQ_MAD_INTENSITY,
    TemplateMeasurementFormat.IQ_MAX_INTENSITY,
    TemplateMeasurementFormat.IQ_MIN_INTENSITY,
]
SATURATION_FEATURES = [Feature.PERCENT_MAXIMAL, Feature.PERCENT_MINIMAL]
SATURATION_TEMPLATES = [TemplateMeasurementFormat.IQ_PERCENT_MAXIMAL, TemplateMeasurementFormat.IQ_PERCENT_MINIMAL]
AGGREGATES = [Aggregate.MEAN.value, Aggregate.MEDIAN.value, Aggregate.STD.value]
MEAN_THRESH_ALL_IMAGES = "MeanThresh_AllImages"
MEDIAN_THRESH_ALL_IMAGES = "MedianThresh_AllImages"
STD_THRESH_ALL_IMAGES = "StdThresh_AllImages"

class ScaledThresholdMethod(str, Enum):
    OTSU = TM_OTSU
    MOG = TM_MOG
# could just do `from .threshold import OtsuMethod` at the top
# but we make it explicit here to silence static analysis warnings
OtsuMethod = ThreshOtsuMethod
# similar to above
THRESHOLD_METHODS = TM_METHODS

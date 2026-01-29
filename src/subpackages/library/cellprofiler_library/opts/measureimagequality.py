from enum import Enum

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
SATURATION_FEATURES = [Feature.PERCENT_MAXIMAL, Feature.PERCENT_MINIMAL]
MEAN_THRESH_ALL_IMAGES = "MeanThresh_AllImages"
MEDIAN_THRESH_ALL_IMAGES = "MedianThresh_AllImages"
STD_THRESH_ALL_IMAGES = "StdThresh_AllImages"

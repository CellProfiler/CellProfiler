from enum import Enum


class IntensityMeasurementFormat(str, Enum):
    TOTAL_INTENSITY = "Intensity_TotalIntensity_%s"
    MEAN_INTENSITY = "Intensity_MeanIntensity_%s"
    MEDIAN_INTENSITY = "Intensity_MedianIntensity_%s"
    STD_INTENSITY = "Intensity_StdIntensity_%s"
    MAD_INTENSITY = "Intensity_MADIntensity_%s"
    MAX_INTENSITY = "Intensity_MaxIntensity_%s"
    MIN_INTENSITY = "Intensity_MinIntensity_%s"
    TOTAL_AREA = "Intensity_TotalArea_%s"
    PERCENT_MAXIMAL = "Intensity_PercentMaximal_%s"
    LOWER_QUARTILE = "Intensity_LowerQuartileIntensity_%s"
    UPPER_QUARTILE  = "Intensity_UpperQuartileIntensity_%s"

ALL_MEASUREMENTS = [
    x.value.split("_")[1] for x in IntensityMeasurementFormat
]
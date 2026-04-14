from enum import Enum

C_INTENSITY = "Intensity"

class Feature(str, Enum):
    """Enumeration of raw intensity measurement features."""
    TOTAL_INTENSITY  = "TotalIntensity"
    MEAN_INTENSITY   = "MeanIntensity"
    MEDIAN_INTENSITY = "MedianIntensity"
    STD_INTENSITY    = "StdIntensity"
    MAD_INTENSITY    = "MADIntensity"
    MAX_INTENSITY    = "MaxIntensity"
    MIN_INTENSITY    = "MinIntensity"
    TOTAL_AREA       = "TotalArea"
    PERCENT_MAXIMAL  = "PercentMaximal"
    LOWER_QUARTILE   = "LowerQuartileIntensity"
    UPPER_QUARTILE   = "UpperQuartileIntensity"

class TemplateMeasurementFormat(str):
    """A string subclass for measurement formatting."""
    TOTAL_INTENSITY = f"{C_INTENSITY}_{Feature.TOTAL_INTENSITY.value}_%s"
    MEAN_INTENSITY  = f"{C_INTENSITY}_{Feature.MEAN_INTENSITY.value}_%s"
    MEDIAN_INTENSITY = f"{C_INTENSITY}_{Feature.MEDIAN_INTENSITY.value}_%s"
    STD_INTENSITY   = f"{C_INTENSITY}_{Feature.STD_INTENSITY.value}_%s"
    MAD_INTENSITY   = f"{C_INTENSITY}_{Feature.MAD_INTENSITY.value}_%s"
    MAX_INTENSITY   = f"{C_INTENSITY}_{Feature.MAX_INTENSITY.value}_%s"
    MIN_INTENSITY   = f"{C_INTENSITY}_{Feature.MIN_INTENSITY.value}_%s"
    TOTAL_AREA      = f"{C_INTENSITY}_{Feature.TOTAL_AREA.value}_%s"
    PERCENT_MAXIMAL = f"{C_INTENSITY}_{Feature.PERCENT_MAXIMAL.value}_%s"
    LOWER_QUARTILE  = f"{C_INTENSITY}_{Feature.LOWER_QUARTILE.value}_%s"
    UPPER_QUARTILE  = f"{C_INTENSITY}_{Feature.UPPER_QUARTILE.value}_%s"
    # Intensity_Percentile_{percentile_n}_{measurement}
    INTENSITY_PERCENTILE = f"{C_INTENSITY}_Percentile_%s_%s"

# Iterates over the Enum to get the list of raw strings
ALL_MEASUREMENTS = [x.value for x in Feature]

FORMATED_FEATURE_NAMES = {
    Feature.TOTAL_INTENSITY.value: "Total intensity",
    Feature.MEAN_INTENSITY.value: "Mean intensity",
    Feature.MEDIAN_INTENSITY.value: "Median intensity",
    Feature.STD_INTENSITY.value: "Std intensity",
    Feature.MAD_INTENSITY.value: "MAD intensity",
    Feature.MAX_INTENSITY.value: "Max intensity",
    Feature.MIN_INTENSITY.value: "Min intensity",
    Feature.TOTAL_AREA.value: "Total area",
    Feature.PERCENT_MAXIMAL.value: "Pct maximal",
    Feature.LOWER_QUARTILE.value: "Lower quartile",
    Feature.UPPER_QUARTILE.value: "Upper quartile",
}
# Percentile {percentile_n}
FORMATED_PERCENTILE_TEMPLATE = "Percentile %s"

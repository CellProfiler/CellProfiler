from enum import Enum

INTENSITY = "Intensity"

class IntensityMeasurement(str, Enum):
    INTEGRATED_INTENSITY = "IntegratedIntensity"
    MEAN_INTENSITY = "MeanIntensity"
    STD_INTENSITY = "StdIntensity"
    MIN_INTENSITY = "MinIntensity"
    MAX_INTENSITY = "MaxIntensity"

    INTEGRATED_INTENSITY_EDGE = "IntegratedIntensityEdge"
    MEAN_INTENSITY_EDGE = "MeanIntensityEdge"
    STD_INTENSITY_EDGE = "StdIntensityEdge"
    MIN_INTENSITY_EDGE = "MinIntensityEdge"
    MAX_INTENSITY_EDGE = "MaxIntensityEdge"
    LOWER_QUARTILE_INTENSITY = "LowerQuartileIntensity"
    UPPER_QUARTILE_INTENSITY = "UpperQuartileIntensity"
    MEDIAN_INTENSITY = "MedianIntensity"
    MAD_INTENSITY = "MADIntensity"

    MASS_DISPLACEMENT = "MassDisplacement"
    LOC_CMI_X = "CenterMassIntensity_X"
    LOC_CMI_Y = "CenterMassIntensity_Y"
    LOC_CMI_Z = "CenterMassIntensity_Z"
    LOC_MAX_X = "MaxIntensity_X"
    LOC_MAX_Y = "MaxIntensity_Y"
    LOC_MAX_Z = "MaxIntensity_Z"

ALL_MEASUREMENTS = [
    IntensityMeasurement.INTEGRATED_INTENSITY,
    IntensityMeasurement.MEAN_INTENSITY,
    IntensityMeasurement.STD_INTENSITY,
    IntensityMeasurement.MIN_INTENSITY,
    IntensityMeasurement.MAX_INTENSITY,
    IntensityMeasurement.INTEGRATED_INTENSITY_EDGE,
    IntensityMeasurement.MEAN_INTENSITY_EDGE,
    IntensityMeasurement.STD_INTENSITY_EDGE,
    IntensityMeasurement.MIN_INTENSITY_EDGE,
    IntensityMeasurement.MAX_INTENSITY_EDGE,
    IntensityMeasurement.LOWER_QUARTILE_INTENSITY,
    IntensityMeasurement.UPPER_QUARTILE_INTENSITY,
    IntensityMeasurement.MEDIAN_INTENSITY,
    IntensityMeasurement.MAD_INTENSITY,
    IntensityMeasurement.MASS_DISPLACEMENT,
]

ALL_LOCATION_MEASUREMENTS = [
    IntensityMeasurement.LOC_CMI_X,
    IntensityMeasurement.LOC_CMI_Y,
    IntensityMeasurement.LOC_CMI_Z,
    IntensityMeasurement.LOC_MAX_X,
    IntensityMeasurement.LOC_MAX_Y,
    IntensityMeasurement.LOC_MAX_Z,
]

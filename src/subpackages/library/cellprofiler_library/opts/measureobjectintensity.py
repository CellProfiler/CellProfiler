from enum import Enum

C_INTENSITY = "Intensity"
C_LOCATION = "Location"

class IntensityFeature(str, Enum):
    """Raw strings for intensity and location features."""
    # General Intensity Features
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
    
    # Location Features
    LOC_CMI_X = "CenterMassIntensity_X"
    LOC_CMI_Y = "CenterMassIntensity_Y"
    LOC_CMI_Z = "CenterMassIntensity_Z"
    LOC_MAX_X = "MaxIntensity_X"
    LOC_MAX_Y = "MaxIntensity_Y"
    LOC_MAX_Z = "MaxIntensity_Z"

class TemplateMeasurementFormat(str):
    """Template strings for measurement formatting."""
    INTEGRATED_INTENSITY = f"{C_INTENSITY}_{IntensityFeature.INTEGRATED_INTENSITY.value}_%s"
    MEAN_INTENSITY = f"{C_INTENSITY}_{IntensityFeature.MEAN_INTENSITY.value}_%s"
    STD_INTENSITY = f"{C_INTENSITY}_{IntensityFeature.STD_INTENSITY.value}_%s"
    MIN_INTENSITY = f"{C_INTENSITY}_{IntensityFeature.MIN_INTENSITY.value}_%s"
    MAX_INTENSITY = f"{C_INTENSITY}_{IntensityFeature.MAX_INTENSITY.value}_%s"
    INTEGRATED_INTENSITY_EDGE = f"{C_INTENSITY}_{IntensityFeature.INTEGRATED_INTENSITY_EDGE.value}_%s"
    MEAN_INTENSITY_EDGE = f"{C_INTENSITY}_{IntensityFeature.MEAN_INTENSITY_EDGE.value}_%s"
    STD_INTENSITY_EDGE = f"{C_INTENSITY}_{IntensityFeature.STD_INTENSITY_EDGE.value}_%s"
    MIN_INTENSITY_EDGE = f"{C_INTENSITY}_{IntensityFeature.MIN_INTENSITY_EDGE.value}_%s"
    MAX_INTENSITY_EDGE = f"{C_INTENSITY}_{IntensityFeature.MAX_INTENSITY_EDGE.value}_%s"
    LOWER_QUARTILE_INTENSITY = f"{C_INTENSITY}_{IntensityFeature.LOWER_QUARTILE_INTENSITY.value}_%s"
    UPPER_QUARTILE_INTENSITY = f"{C_INTENSITY}_{IntensityFeature.UPPER_QUARTILE_INTENSITY.value}_%s"
    MEDIAN_INTENSITY = f"{C_INTENSITY}_{IntensityFeature.MEDIAN_INTENSITY.value}_%s"
    MAD_INTENSITY = f"{C_INTENSITY}_{IntensityFeature.MAD_INTENSITY.value}_%s"
    MASS_DISPLACEMENT = f"{C_INTENSITY}_{IntensityFeature.MASS_DISPLACEMENT.value}_%s"
    
    # Location Templates
    LOC_CMI_X = f"{C_LOCATION}_{IntensityFeature.LOC_CMI_X.value}_%s"
    LOC_CMI_Y = f"{C_LOCATION}_{IntensityFeature.LOC_CMI_Y.value}_%s"
    LOC_CMI_Z = f"{C_LOCATION}_{IntensityFeature.LOC_CMI_Z.value}_%s"
    LOC_MAX_X = f"{C_LOCATION}_{IntensityFeature.LOC_MAX_X.value}_%s"
    LOC_MAX_Y = f"{C_LOCATION}_{IntensityFeature.LOC_MAX_Y.value}_%s"
    LOC_MAX_Z = f"{C_LOCATION}_{IntensityFeature.LOC_MAX_Z.value}_%s"

# Generate lists by filtering the Enum members
ALL_MEASUREMENTS = [
    x.value for x in IntensityFeature 
    if not x.name.startswith("LOC_")
]

ALL_LOCATION_MEASUREMENTS = [
    x.value for x in IntensityFeature 
    if x.name.startswith("LOC_")
]
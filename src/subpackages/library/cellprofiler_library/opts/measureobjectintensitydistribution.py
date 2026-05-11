from enum import Enum

"""Root module measurement category"""
C_RADIAL_DISTRIBUTION = "RadialDistribution"


class CenterChoice(str, Enum):
    SELF = "These objects"
    CENTERS_OF_OTHER = "Centers of other objects"
    EDGES_OF_OTHER = "Edges of other objects"


"""Legacy value for the "centers of other" choice in pipelines saved
with variable_revision_number <= 2. Retained for upgrade_settings."""
C_CENTERS_OF_OTHER_V2 = "Other objects"

C_ALL = [
    CenterChoice.SELF.value,
    CenterChoice.CENTERS_OF_OTHER.value,
    CenterChoice.EDGES_OF_OTHER.value,
]


class ZernikeMode(str, Enum):
    NONE = "None"
    MAGNITUDES = "Magnitudes only"
    MAGNITUDES_AND_PHASE = "Magnitudes and phase"


Z_ALL = [
    ZernikeMode.NONE.value,
    ZernikeMode.MAGNITUDES.value,
    ZernikeMode.MAGNITUDES_AND_PHASE.value,
]


class Feature(str, Enum):
    FRAC_AT_D = "FracAtD"
    MEAN_FRAC = "MeanFrac"
    RADIAL_CV = "RadialCV"
    ZERNIKE_MAGNITUDE = "ZernikeMagnitude"
    ZERNIKE_PHASE = "ZernikePhase"


F_ALL = [
    Feature.FRAC_AT_D.value,
    Feature.MEAN_FRAC.value,
    Feature.RADIAL_CV.value,
]


class MeasurementChoice(str, Enum):
    """Human-readable labels used by the heatmap measurement selector."""
    FRAC_AT_D = "Fraction at Distance"
    MEAN_FRAC = "Mean Fraction"
    RADIAL_CV = "Radial CV"


MEASUREMENT_CHOICES = [
    MeasurementChoice.FRAC_AT_D.value,
    MeasurementChoice.MEAN_FRAC.value,
    MeasurementChoice.RADIAL_CV.value,
]

"""Format strings used to build per-bin feature names."""
FF_SCALE = "%dof%d"
FF_OVERFLOW = "Overflow"
FF_GENERIC = "_%s_" + FF_SCALE


class TemplateMeasurementFormat(str):
    """Printf-style templates for fully-qualified feature names."""
    # RD = Radial Distribution
    RD_FRAC_AT_D = "_".join((C_RADIAL_DISTRIBUTION, Feature.FRAC_AT_D.value + FF_GENERIC))
    RD_MEAN_FRAC = "_".join((C_RADIAL_DISTRIBUTION, Feature.MEAN_FRAC.value + FF_GENERIC))
    RD_RADIAL_CV = "_".join((C_RADIAL_DISTRIBUTION, Feature.RADIAL_CV.value + FF_GENERIC))
    RD_OVERFLOW_FRAC_AT_D = "_".join(
        (C_RADIAL_DISTRIBUTION, Feature.FRAC_AT_D.value, "%s", FF_OVERFLOW)
    )
    RD_OVERFLOW_MEAN_FRAC = "_".join(
        (C_RADIAL_DISTRIBUTION, Feature.MEAN_FRAC.value, "%s", FF_OVERFLOW)
    )
    RD_OVERFLOW_RADIAL_CV = "_".join(
        (C_RADIAL_DISTRIBUTION, Feature.RADIAL_CV.value, "%s", FF_OVERFLOW)
    )


MEASUREMENT_ALIASES = {
    MeasurementChoice.FRAC_AT_D.value: TemplateMeasurementFormat.RD_FRAC_AT_D,
    MeasurementChoice.MEAN_FRAC.value: TemplateMeasurementFormat.RD_MEAN_FRAC,
    MeasurementChoice.RADIAL_CV.value: TemplateMeasurementFormat.RD_RADIAL_CV,
}

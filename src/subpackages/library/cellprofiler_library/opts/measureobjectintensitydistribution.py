from enum import Enum

class CenterChoice(str, Enum):
    SELF = "These objects"
    CENTERS_OF_OTHER_V2 = "Other objects"
    CENTERS_OF_OTHER = "Centers of other objects"
    EDGES_OF_OTHER = "Edges of other objects"

C_ALL = [CenterChoice.SELF, CenterChoice.CENTERS_OF_OTHER, CenterChoice.EDGES_OF_OTHER]

class IntensityZernike(str, Enum):
    NONE = "None"
    MAGNITUDES = "Magnitudes only"
    MAGNITUDES_AND_PHASE = "Magnitudes and phase"
Z_ALL = [IntensityZernike.NONE, IntensityZernike.MAGNITUDES, IntensityZernike.MAGNITUDES_AND_PHASE]

M_CATEGORY = "RadialDistribution"

class Feature(str, Enum):
    FRAC_AT_D = "FracAtD"
    MEAN_FRAC = "MeanFrac"
    RADIAL_CV = "RadialCV"
F_ALL = [Feature.FRAC_AT_D, Feature.MEAN_FRAC, Feature.RADIAL_CV]

FF_SCALE = "%dof%d"
FF_GENERIC = "_%s_" + FF_SCALE
class FullFeature(str, Enum): # assuming FF_<ABC> means FullFeature_<ABC>
    FRAC_AT_D = Feature.FRAC_AT_D.value + FF_GENERIC
    MEAN_FRAC = Feature.MEAN_FRAC.value + FF_GENERIC
    RADIAL_CV = Feature.RADIAL_CV.value + FF_GENERIC
    ZERNIKE_MAGNITUDE = "ZernikeMagnitude"
    ZERNIKE_PHASE = "ZernikePhase"
    OVERFLOW = "Overflow"

class MeasurementFeature(str, Enum):
    FRAC_AT_D = "_".join((M_CATEGORY, FullFeature.FRAC_AT_D.value))
    MEAN_FRAC = "_".join((M_CATEGORY, FullFeature.MEAN_FRAC.value))
    RADIAL_CV = "_".join((M_CATEGORY, FullFeature.RADIAL_CV.value))

class OverflowFeature(str, Enum):
    FRAC_AT_D = "_".join((M_CATEGORY, Feature.FRAC_AT_D.value, "%s", FullFeature.OVERFLOW.value))
    MEAN_FRAC = "_".join((M_CATEGORY, Feature.MEAN_FRAC.value, "%s", FullFeature.OVERFLOW.value))
    RADIAL_CV = "_".join((M_CATEGORY, Feature.RADIAL_CV.value, "%s", FullFeature.OVERFLOW.value))


class MeasurementAlias(str, Enum):
    FRAC_AT_D = "Fraction at Distance"
    MEAN_FRAC = "Mean Fraction"
    RADIAL_CV = "Radial CV"
MEASUREMENT_CHOICES = [MeasurementAlias.FRAC_AT_D.value, MeasurementAlias.MEAN_FRAC.value, MeasurementAlias.RADIAL_CV.value]

MEASUREMENT_ALIASES = {
    MeasurementAlias.FRAC_AT_D.value: MeasurementFeature.FRAC_AT_D.value,
    MeasurementAlias.MEAN_FRAC.value: MeasurementFeature.MEAN_FRAC.value,
    MeasurementAlias.RADIAL_CV.value: MeasurementFeature.RADIAL_CV.value,
}

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
class TemplateFullFeature(str): # assuming FF_<ABC> means FullFeature_<ABC>
    FRAC_AT_D = Feature.FRAC_AT_D.value + FF_GENERIC
    MEAN_FRAC = Feature.MEAN_FRAC.value + FF_GENERIC
    RADIAL_CV = Feature.RADIAL_CV.value + FF_GENERIC
    ZERNIKE_MAGNITUDE = "ZernikeMagnitude"
    ZERNIKE_PHASE = "ZernikePhase"
    OVERFLOW = "Overflow"

class TemplateMeasurementFeature(str):
    FRAC_AT_D = "_".join((M_CATEGORY, TemplateFullFeature.FRAC_AT_D))
    MEAN_FRAC = "_".join((M_CATEGORY, TemplateFullFeature.MEAN_FRAC))
    RADIAL_CV = "_".join((M_CATEGORY, TemplateFullFeature.RADIAL_CV))

class TemplateOverflowFeature(str):
    FRAC_AT_D = "_".join((M_CATEGORY, Feature.FRAC_AT_D.value, "%s", TemplateFullFeature.OVERFLOW))
    MEAN_FRAC = "_".join((M_CATEGORY, Feature.MEAN_FRAC.value, "%s", TemplateFullFeature.OVERFLOW))
    RADIAL_CV = "_".join((M_CATEGORY, Feature.RADIAL_CV.value, "%s", TemplateFullFeature.OVERFLOW))


class TemplateZernikeFeature(str):
    MAGNITUDE = "_".join((M_CATEGORY, TemplateFullFeature.ZERNIKE_MAGNITUDE, "%s", "%s", "%s"))
    PHASE = "_".join((M_CATEGORY, TemplateFullFeature.ZERNIKE_PHASE, "%s", "%s", "%s"))


class MeasurementAlias(str, Enum):
    FRAC_AT_D = "Fraction at Distance"
    MEAN_FRAC = "Mean Fraction"
    RADIAL_CV = "Radial CV"
MEASUREMENT_CHOICES = [MeasurementAlias.FRAC_AT_D.value, MeasurementAlias.MEAN_FRAC.value, MeasurementAlias.RADIAL_CV.value]

MEASUREMENT_ALIASES = {
    MeasurementAlias.FRAC_AT_D.value: TemplateMeasurementFeature.FRAC_AT_D,
    MeasurementAlias.MEAN_FRAC.value: TemplateMeasurementFeature.MEAN_FRAC,
    MeasurementAlias.RADIAL_CV.value: TemplateMeasurementFeature.RADIAL_CV,
}

ALL_TEMPLATE_MEASUREMENT_FEATURES = [
    TemplateMeasurementFeature.FRAC_AT_D,
    TemplateMeasurementFeature.MEAN_FRAC,
    TemplateMeasurementFeature.RADIAL_CV,
]

ALL_TEMPLATE_OVERFLOW_FEATURES = [
    TemplateOverflowFeature.FRAC_AT_D,
    TemplateOverflowFeature.MEAN_FRAC,
    TemplateOverflowFeature.RADIAL_CV,
]

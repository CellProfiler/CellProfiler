from enum import Enum
import numpy as np
C_WORM = "Worm"
"""Maximum # of sets of paths considered at any level"""
MAX_CONSIDERED = 50000
"""Maximum # of different paths considered for input"""
MAX_PATHS = 400
class TrainingXMLTag:
    ######################################################
    #
    # Training file XML tags:
    #
    ######################################################

    NAMESPACE = "http://www.cellprofiler.org/linked_files/schemas/UntangleWorms.xsd"
    TRAINING_DATA = "training-data"
    VERSION = "version"
    MIN_AREA = "min-area"
    MAX_AREA = "max-area"
    COST_THRESHOLD = "cost-threshold"
    NUM_CONTROL_POINTS = "num-control-points"
    MEAN_ANGLES = "mean-angles"
    INV_ANGLES_COVARIANCE_MATRIX = "inv-angles-covariance-matrix"
    MAX_SKEL_LENGTH = "max-skel-length"
    MAX_RADIUS = "max-radius"
    MIN_PATH_LENGTH = "min-path-length"
    MAX_PATH_LENGTH = "max-path-length"
    MEDIAN_WORM_AREA = "median-worm-area"
    OVERLAP_WEIGHT = "overlap-weight"
    LEFTOVER_WEIGHT = "leftover-weight"
    RADII_FROM_TRAINING = "radii-from-training"
    TRAINING_SET_SIZE = "training-set-size"
    VALUES = "values"
    VALUE = "value"

class Feature(str):
    LENGTH = "Length"
    ANGLE = "Angle"
    CONTROL_POINT_X = "ControlPointX"
    CONTROL_POINT_Y = "ControlPointY"

######################################################
#
# Features measured
#
######################################################
class TemplateMeasurementFormat(str):
    """The length of the worm skeleton"""
    LENGTH = f"%s_{Feature.LENGTH}" 

    """The angle at each of the control points (Worm_Angle_1 for example)"""
    ANGLE = f"%s_{Feature.ANGLE}_%s"

    """The X coordinate of a control point (Worm_ControlPointX_14 for example)"""
    CONTROL_POINT_X = f"%s_{Feature.CONTROL_POINT_X}_%s"

    """The Y coordinate of a control point (Worm_ControlPointY_14 for example)"""
    CONTROL_POINT_Y = f"%s_{Feature.CONTROL_POINT_Y}_%s"

class OverlapStyle(str, Enum):
    WITH_OVERLAP = "With overlap"
    WITHOUT_OVERLAP = "Without overlap"
    BOTH = "Both"

class Mode(str, Enum):
    TRAIN = "Train"
    UNTANGLE = "Untangle"

class Complexity(str, Enum):
    ALL = "Process all clusters"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very high"
    CUSTOM = "Custom"

ALL_VALUE = np.iinfo(int).max
MEDIUM_VALUE = 200
HIGH_VALUE = 600
VERY_HIGH_VALUE = 1000

complexity_limits = {
    Complexity.ALL: ALL_VALUE,
    Complexity.MEDIUM: MEDIUM_VALUE,
    Complexity.HIGH: HIGH_VALUE,
    Complexity.VERY_HIGH: VERY_HIGH_VALUE,
}
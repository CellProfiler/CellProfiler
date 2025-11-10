from enum import Enum

class SkeletonMeasurements(str, Enum):
    """The trunk count feature"""
    NUMBER_TRUNKS = "NumberTrunks"
    """The branch feature"""
    NUMBER_NON_TRUNK_BRANCHES = "NumberNonTrunkBranches"
    """The endpoint feature"""
    NUMBER_BRANCH_ENDS = "NumberBranchEnds"
    """The neurite length feature"""
    TOTAL_OBJSKELETON_LENGTH = "TotalObjectSkeletonLength"

F_ALL = [
    SkeletonMeasurements.NUMBER_TRUNKS,
    SkeletonMeasurements.NUMBER_NON_TRUNK_BRANCHES,
    SkeletonMeasurements.NUMBER_BRANCH_ENDS,
    SkeletonMeasurements.TOTAL_OBJSKELETON_LENGTH,
]

VF_IMAGE_NUMBER = "image_number"
VF_VERTEX_NUMBER = "vertex_number"

VF_I = "i"
VF_J = "j"
VF_LABELS = "labels"
VF_KIND = "kind"

vertex_file_columns = (
    VF_IMAGE_NUMBER,
    VF_VERTEX_NUMBER,
    VF_I,
    VF_J,
    VF_LABELS,
    VF_KIND,
)

EF_IMAGE_NUMBER = "image_number"

EF_V1 = "v1"
EF_V2 = "v2"
EF_LENGTH = "length"
EF_TOTAL_INTENSITY = "total_intensity"

edge_file_columns = (EF_IMAGE_NUMBER, EF_V1, EF_V2, EF_LENGTH, EF_TOTAL_INTENSITY)

"""The measurement category"""
C_OBJSKELETON = "ObjectSkeleton"
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

"""The measurement category"""
C_OBJSKELETON = "ObjectSkeleton"
from enum import Enum

# Decimation Method for Earth Mover's Distance
class DecimationMethod(str, Enum):
    KMEANS = "K Means"
    SKELETON = "Skeleton"


class Feature(str, Enum):
    F_FACTOR = "Ffactor"
    PRECISION = "Precision"
    RECALL = "Recall"
    TRUE_POS_RATE = "TruePosRate"
    FALSE_POS_RATE = "FalsePosRate"
    FALSE_NEG_RATE = "FalseNegRate"
    TRUE_NEG_RATE = "TrueNegRate"
    RAND_INDEX = "RandIndex"
    ADJUSTED_RAND_INDEX = "AdjustedRandIndex"
    EARTH_MOVERS_DISTANCE = "EarthMoversDistance"

ALL_FEATURES = [
    Feature.F_FACTOR,
    Feature.PRECISION,
    Feature.RECALL,
    Feature.TRUE_POS_RATE,
    Feature.FALSE_POS_RATE,
    Feature.FALSE_NEG_RATE,
    Feature.TRUE_NEG_RATE,
    Feature.RAND_INDEX,
    Feature.ADJUSTED_RAND_INDEX,
    # Feature.EARTH_MOVERS_DISTANCE, # Earth Movers Distance is intentionally not included in ALL_FEATURES
]

C_IMAGE_OVERLAP = "Overlap"
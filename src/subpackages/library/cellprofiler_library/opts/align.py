from enum import Enum

class AlignmentMethod(str, Enum):
    MUTUAL_INFORMATION = "Mutual Information"
    CROSS_CORRELATION = "Normalized Cross Correlation"

class CropMode(str, Enum):
    SAME_SIZE = "Keep size"
    CROP = "Crop to aligned region"
    PAD = "Pad images"

class AdditionalAlignmentChoice(str, Enum):
    SIMILARLY = "Similarly"
    SEPARATELY = "Separately"

M_ALL = (AlignmentMethod.MUTUAL_INFORMATION, AlignmentMethod.CROSS_CORRELATION)

C_ALIGN = "Align"

MEASUREMENT_FORMAT = C_ALIGN + "_%sshift_%s"
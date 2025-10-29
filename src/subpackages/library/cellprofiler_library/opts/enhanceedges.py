from enum import Enum

class EdgeFindingMethod(str, Enum):
    SOBEL = "Sobel"
    PREWITT = "Prewitt"
    ROBERTS = "Roberts"
    LOG = "Log"
    CANNY = "Canny"
    KIRSCH = "Kirsch"

class EdgeDirection(str, Enum):
    ALL = "All"
    HORIZONTAL = "Horizontal"
    VERTICAL = "Vertical"
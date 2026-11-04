from enum import Enum

class SmoothingMethod(str, Enum):
    FIT_POLYNOMIAL = "Fit Polynomial"
    MEDIAN_FILTER = "Median Filter"
    GAUSSIAN_FILTER = "Gaussian Filter"
    SMOOTH_KEEPING_EDGES = "Smooth Keeping Edges"
    CIRCULAR_AVERAGE_FILTER = "Circular Average Filter"
    SM_TO_AVERAGE = "Smooth to Average"
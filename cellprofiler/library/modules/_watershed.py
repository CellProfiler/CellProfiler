from typing import Literal
import numpy
from cellprofiler.library.functions.object_processing import (
    watershed as library_watershed,
)
import skimage

# Simple wrapper for the object_procceing watershed function
def watershed(
    input_image: numpy.ndarray,
    watershed_method: Literal["intensity", "distance", "markers"] = "distance",
    declump_method: Literal["shape", "intensity", None] = "shape",
    local_maxima_method: Literal["local", "regional"] = "local",
    intensity_image: numpy.ndarray = None,
    markers_image: numpy.ndarray = None,
    markers_mask: numpy.ndarray = None,
    max_seeds: int = -1,
    downsample: int = 1,
    min_distance: int = 1,
    min_intensity: float = 0,
    footprint: int = 8,
    connectivity: int = 1,
    compactness: float = 0.0,
    exclude_border: bool = True,
    watershed_line: bool = False,
    gaussian_sigma: float = 1.0,
    structuring_element: Literal[
        "ball", "cube", "diamond", "disk", "octahedron", "square", "star"
    ] = "disk",
    structuring_element_size: int = 1,
    return_seeds: bool = True,
):
    y_data = library_watershed(
        input_image=input_image,
        watershed_method=watershed_method,
        declump_method=declump_method,
        local_maxima_method=local_maxima_method,
        intensity_image=intensity_image,
        markers_image=markers_image,
        markers_mask=markers_mask,
        max_seeds=max_seeds,
        downsample=downsample,
        min_distance=min_distance,
        min_intensity=min_intensity,
        footprint=footprint,
        connectivity=connectivity,
        compactness=compactness,
        exclude_border=exclude_border,
        watershed_line=watershed_line,
        gaussian_sigma=gaussian_sigma,
        structuring_element=structuring_element,
        structuring_element_size=structuring_element_size,
        return_seeds=return_seeds,
    )
    return y_data

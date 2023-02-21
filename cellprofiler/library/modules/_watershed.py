from typing import Literal
from cellprofiler.library.functions.object_processing import watershed as library_watershed
import skimage

# Simple wrapper for the object_procceing watershed function
def watershed(
    input_image,
    watershed_method: Literal["intensity", "distance", "markers"] = "distance",
    declump_method: Literal["shape", "intensity", None] = "shape",
    local_maxima_method: Literal["local", "regional"] = "local",
    intensity_image=None,
    markers_image=None,
    max_seeds: int = -1,
    downsample: int = 1,
    min_distance: int = 1,
    footprint: int = 8,
    connectivity: int = 1,
    compactness: int = 0,
    exclude_border: bool = True,
    watershed_line: bool = False,
    gaussian_sigma: int = 1,
    structuring_element: Literal[
        "ball", "cube", "diamond", "disk", "octahedron", "square", "star"
    ] = "disk",
    structuring_element_size: int = 1,
):
    y_data = library_watershed(
        input_image,
        watershed_method,
        declump_method,
        local_maxima_method,
        intensity_image,
        markers_image,
        max_seeds,
        downsample,
        min_distance,
        footprint,
        connectivity,
        compactness,
        exclude_border,
        watershed_line,
        gaussian_sigma,
        structuring_element,
        structuring_element_size,
    )
    return y_data
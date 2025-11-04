import numpy as np
from numpy.typing import NDArray
from typing import List, Annotated, Optional
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.functions.measurement import measure_image_intensities

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_intensity(
        pixels:         Annotated[NDArray[np.float32], Field(description="Image pixel data")],
        percentiles:    Annotated[Optional[List[int]], Field(description="Percentiles to measure")]=[],
        ):
    if percentiles is None:
        percentiles = []
    statistics, percentile_measures = measure_image_intensities(pixels, percentiles)
    return statistics, percentile_measures

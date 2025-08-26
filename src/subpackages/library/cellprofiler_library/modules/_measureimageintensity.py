import numpy as np
from numpy.typing import NDArray
from typing import List, Annotated
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.functions.measurement import get_intensity_measurements

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_intensity(
        pixels: Annotated[NDArray[np.float32], Field(description="Image pixel data")],
        percentiles: Annotated[List[int], Field(description="Percentiles to measure")]=[],
        ):
    statistics, percentile_measures = get_intensity_measurements(pixels, percentiles)
    return statistics, percentile_measures

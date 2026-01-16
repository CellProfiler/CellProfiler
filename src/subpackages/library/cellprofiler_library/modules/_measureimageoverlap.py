import numpy
from typing import Annotated, Optional, Dict, Any
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.opts.measureimageoverlap import DM, C_IMAGE_OVERLAP
from cellprofiler_library.functions.measurement import (
    measure_image_overlap_statistics,
    compute_earth_movers_distance,
)
from cellprofiler_library.types import ImageBinary, ImageBinaryMask
from cellprofiler_library.measurement_model import LibraryMeasurements

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measureimageoverlap(
    ground_truth_image: Annotated[ImageBinary, Field(description="Ground truth binary image")],
    test_image:         Annotated[ImageBinary, Field(description="Test binary image")],
    test_image_name:    Annotated[str, Field(description="Name of the test image")],
    mask:               Annotated[Optional[ImageBinaryMask], Field(description="Mask image")] = None,
    calculate_emd:      Annotated[bool, Field(description="Calculate Earth Movers Distance")] = False,
    max_distance:       Annotated[int, Field(description="Maximum distance for EMD")] = 250,
    penalize_missing:   Annotated[bool, Field(description="Penalize missing points")] = False,
    decimation_method:  Annotated[DM, Field(description="Decimation method")] = DM.KMEANS,
    max_points:         Annotated[int, Field(description="Maximum number of points")] = 250,
) -> LibraryMeasurements:

    data = measure_image_overlap_statistics(
        ground_truth_image=ground_truth_image, test_image=test_image, mask=mask
    )

    if calculate_emd:
        emd = compute_earth_movers_distance(
            ground_truth_image=ground_truth_image,
            test_image=test_image,
            max_distance=max_distance,
            penalize_missing=penalize_missing,
            decimation_method=decimation_method,
            max_points=max_points,
        )
        data.update({"EarthMoversDistance": emd})
    
    measurements = LibraryMeasurements()
    
    for key, value in data.items():
        if isinstance(value, (int, float, numpy.number)):
            # Scalar measurement
            feature_name = f"{C_IMAGE_OVERLAP}_{key}_{test_image_name}"
            measurements.add_image_measurement(feature_name, float(value))
        else:
            # Display data (arrays)
            measurements.image[key] = value

    return measurements

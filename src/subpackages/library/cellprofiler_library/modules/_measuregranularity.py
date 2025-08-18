import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call, ConfigDict, Field
from typing import Annotated, List
from cellprofiler_library.functions.measurement import get_granularity_measurements, ObjectRecord
from cellprofiler_library.opts.measuregranularity import C_GRANULARITY
from cellprofiler_library.functions.image_processing import apply_grayscale_tophat_filter, downsample_image_and_mask


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_granularity(
        im_pixel_data: Annotated[NDArray[np.float32], Field(description="Pixel values of the image")],
        im_mask: Annotated[NDArray[np.bool_], Field(description="Boolean mask of where the measurements should be made")],
        subsample_size: Annotated[float, Field(description="Subsampling factor for granularity measurements")],
        image_sample_size: Annotated[float, Field(description="Subsampling factor for background reduction")],
        element_size: Annotated[int, Field(description="Radius of structuring element")],
        object_records: Annotated[List[ObjectRecord], Field(description="Object records")],
        granular_spectrum_length: Annotated[int, Field(description="Range of the granular spectrum")],
        dimensions: Annotated[int, Field(description="Dimensionality of the image")] = 2,
        ):
    #
    # Downsample the image and mask
    #
    pixels, mask, new_shape = downsample_image_and_mask(im_pixel_data, im_mask, dimensions, subsample_size)
    
    #
    # Remove background pixels using a greyscale tophat filter
    #
    pixels = apply_grayscale_tophat_filter(pixels, mask, dimensions, image_sample_size, element_size, new_shape)
    
    #
    # Compute measurements
    #
    _measurements_arr, _image_measurements_arr, _statistics = get_granularity_measurements(
        im_pixel_data,
        pixels, 
        mask, 
        new_shape,
        granular_spectrum_length,
        dimensions,
        object_records
    )

    return _measurements_arr, _image_measurements_arr, _statistics
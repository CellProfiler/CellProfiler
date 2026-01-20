import numpy
from typing import Annotated, Tuple, Union
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.types import ImageGrayscale
from cellprofiler_library.opts.measureimageskeleton import TemplateMeasurementFormat
from cellprofiler_library.functions.measurement import branches, endpoints
from cellprofiler_library.measurement_model import LibraryMeasurements

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_skeleton(
        im_pixel_data: Annotated[ImageGrayscale, Field(description="Input image")],
        im_name: Annotated[str, Field(description="Input image name")]="Image1",
        return_visualization_data: Annotated[bool, Field(description="Return data for display")] = False,
    ) -> Union[LibraryMeasurements, Tuple[LibraryMeasurements, numpy.ndarray, numpy.ndarray]]:
    pixels = im_pixel_data > 0
    branch_nodes = branches(pixels)
    endpoint_nodes = endpoints(pixels)
    num_branches = numpy.count_nonzero(branch_nodes)
    num_endpoints = numpy.count_nonzero(endpoint_nodes)
    
    measurements = LibraryMeasurements()
    measurements.add_image_measurement(
        TemplateMeasurementFormat.BRANCHES % im_name,
        num_branches
    )
    measurements.add_image_measurement(
        TemplateMeasurementFormat.ENDPOINTS % im_name,
        num_endpoints
    )
    
    if return_visualization_data:
        return measurements, branch_nodes, endpoint_nodes
    
    return measurements

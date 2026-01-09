import numpy
from typing import Annotated
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.types import ImageGrayscale
from cellprofiler_library.opts.measureimageskeleton import SkeletonMeasurements
from cellprofiler_library.functions.measurement import branches, endpoints

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_skeleton(
        im_pixel_data: Annotated[ImageGrayscale, Field(description="Input image")],
        im_name: Annotated[str, Field(description="Input image name")]="Image1",
    ):
    pixels = im_pixel_data > 0
    branch_nodes = branches(pixels)
    endpoint_nodes = endpoints(pixels)
    num_branches = numpy.count_nonzero(branch_nodes)
    num_endpoints = numpy.count_nonzero(endpoint_nodes)
    measurements = {
        "Image": {
            SkeletonMeasurements.BRANCHES.value.format(im_name): num_branches,
            SkeletonMeasurements.ENDPOINTS.value.format(im_name): num_endpoints,
        }
    }
    return branch_nodes, endpoint_nodes, measurements


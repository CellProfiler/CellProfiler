import numpy
from numpy.typing import NDArray
import skimage.segmentation
from typing import Annotated, Tuple, Union, List, Any
from pydantic import Field, validate_call, ConfigDict, BaseModel
from cellprofiler_library.types import ImageGrayscale
from cellprofiler_library.opts.measureimageskeleton import TemplateMeasurementFormat
from cellprofiler_library.functions.measurement import branches, endpoints
from cellprofiler_library.measurement_model import LibraryMeasurements


ImageSkeletonStatistics = List[Tuple[int, int]]

class ImageSkeletonDisplayData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        populate_by_name=True
    )

    statistics: ImageSkeletonStatistics
    nodes: NDArray[Any]

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_skeleton(
        im_pixel_data: Annotated[ImageGrayscale, Field(description="Input image")],
        im_name: Annotated[str, Field(description="Input image name")]="Image1",
        return_visualization_data: Annotated[bool, Field(description="Return data for display")] = False,
    ) -> Union[LibraryMeasurements, Tuple[LibraryMeasurements, ImageSkeletonDisplayData]]:
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
        a = numpy.copy(branch_nodes).astype(numpy.uint16)
        b = numpy.copy(endpoint_nodes).astype(numpy.uint16)

        a[a == 1] = 1
        b[b == 1] = 2

        nodes: NDArray[Any] = skimage.segmentation.join_segmentations(a, b)

        return measurements, ImageSkeletonDisplayData(
            statistics=[(num_branches, num_endpoints)],
            nodes=nodes
        )
    
    return measurements

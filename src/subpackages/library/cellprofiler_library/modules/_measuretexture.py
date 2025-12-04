import numpy
from numpy.typing import NDArray
from typing import List, Dict, Union, Optional, Annotated
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask, ObjectLabel, ObjectSegmentation
from pydantic import validate_call, Field, ConfigDict
from cellprofiler_library.functions.measurement import get_image_texture_measurements, get_object_texture_measurements


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_texture(
        pixel_data: Annotated[ImageGrayscale, Field(description="Input image to perform texture measurements on")],
        gray_levels: Annotated[int, Field(description="Enter the number of gray levels (ie, total possible values of intensity) you want to measure texture at")], 
        scale: Annotated[int, Field(description="You can specify the scale of texture to be measured, in pixel units; the texture scale is the distance between correlated intensities in the image")], 
        image_name: Annotated[str, Field(description="Name to be assigned in measurements")]
    ) -> List[
        Dict[str, Union[str, numpy.float_]]
    ]:
    return get_image_texture_measurements(pixel_data, gray_levels, scale, image_name)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_texture(
        object_name: Annotated[str, Field(description="Object name to be assigned in measurements")], 
        labels: Annotated[ObjectSegmentation, Field(description="Segmentation labels for object")],
        image_name: Annotated[str, Field(description="Name of the image to assign in measurements")],
        pixel_data: Annotated[ImageGrayscale, Field(description="Image pixel data to measure texture on")],
        mask: Annotated[Optional[ImageGrayscaleMask], Field(description="Image mask if any")],
        gray_levels: Annotated[int, Field(description="Enter the number of gray levels (ie, total possible values of intensity) you want to measure texture at")], 
        unique_labels: Annotated[NDArray[ObjectLabel], Field(description="The unique labels in the object segmentation prior to any masking")], # objects.indices
        scale: Annotated[int, Field(description="You can specify the scale of texture to be measured, in pixel units; the texture scale is the distance between correlated intensities in the image")], 
        volumetric: Annotated[bool, Field(description="Is the input image or objects 3D?")],
    ):
    return get_object_texture_measurements(object_name, labels, image_name, pixel_data, mask, gray_levels, unique_labels, scale, volumetric)
    
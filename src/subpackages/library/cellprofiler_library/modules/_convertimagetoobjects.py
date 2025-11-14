from typing import Annotated, Optional, Union
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.types import ImageGrayscale, ObjectLabelsDense, ImageBinary
from cellprofiler_library.functions.image_processing import image_to_objects

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def convert_image_to_objects(
        data:           Annotated[Union[ImageGrayscale, ImageBinary], Field(description="Image to be converted to Objects")],
        cast_to_bool:   Annotated[bool, Field(description="Convert a grayscale image to binary before converting it to an object")],
        preserve_label: Annotated[bool, Field(description="Preserve original labels of objects")],
        background:     Annotated[int, Field(description="Pixel value of the background")],
        connectivity:   Annotated[Optional[int], Field(description="Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor")]
        ) -> ObjectLabelsDense:
    return image_to_objects(data, cast_to_bool, preserve_label, background, connectivity)

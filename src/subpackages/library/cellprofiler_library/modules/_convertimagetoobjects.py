from typing import Annotated, Any, Optional, Tuple, Callable, Union
from pydantic import Field, validate_call, BeforeValidator, ConfigDict

from cellprofiler_library.types import ImageGrayscale
from ..functions.image_processing import convert_image_to_objects

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def convert_image_to_objects(
        data:           Annotated[ImageGrayscale, Field(description="Image to be converted to Objects")], 
        cast_to_bool:   Annotated[bool, Field(description="Convert a grayscale image to binary before converting it to an object")], 
        preserve_label: Annotated[bool, Field(description="Preserve original labels of objects")], 
        background:     Annotated[int, Field(description="Pixel value of the background")], 
        connectivity:   Annotated[Union[int, None], Field(description="Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor")]
        ) -> Any:
    return convert_image_to_objects(data, cast_to_bool, preserve_label, background, connectivity)
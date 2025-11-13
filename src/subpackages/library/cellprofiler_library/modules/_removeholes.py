from pydantic import validate_call, Field, ConfigDict
from typing import Annotated
from cellprofiler_library.functions.image_processing import fill_holes
from ..types import ImageAny

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def remove_holes(
        image: Annotated[ImageAny, Field(description="Image to fill holes in")],
        diameter: Annotated[float, Field(description="Diameter of holes to fill")],
    ) -> ImageAny:
    return fill_holes(image, diameter)
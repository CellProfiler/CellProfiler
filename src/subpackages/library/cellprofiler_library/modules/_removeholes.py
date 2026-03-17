from pydantic import validate_call, Field, ConfigDict
from typing import Annotated
from cellprofiler_library.functions.image_processing import fill_holes
from ..types import ImageAny
import numpy as np
from numpy.typing import NDArray
from typing import Union

# TODO: #5088 Need to add support for label images (Uint8) for this module. 
# PR #5088 RescaleIntensity adds UInt8 support to ImageAny.
# For now, I will add UInt8 here manually. Ignore the return type error as it will be fixed in #5088.
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def remove_holes(
        image: Annotated[Union[ImageAny, NDArray[np.uint8]], Field(description="Image to fill holes in")],
        diameter: Annotated[float, Field(description="Diameter of holes to fill")],
    ) -> Union[ImageAny, NDArray[np.uint8]]:
    # TODO: #5056 I'm very puzzled as to why the fn call below suddenly broke.
    # I've checked the pre-refactor commit to see if it works there, but it doesn't.
    # Has the underlying dependency changed? skimage.morphology.remove_small_holes
    # Should we upgrade our deps to the latest version?
    return fill_holes(image, diameter)

from pydantic import Field, validate_call, ConfigDict
from typing import Annotated, Any, Iterable, Sequence
import numpy as np
import matplotlib.colors
from ..types import Image2DColor, Image2DGrayscale

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def combine_colortogray(
    image: Annotated[Image2DColor, Field(description="Pixel data of image to threshold")],
    channels: Annotated[Sequence[int], Field(description="Array of integer identifier ")],
    contributions: Annotated[Sequence[float], Field(description="Array of contribution values")],
    ) -> Image2DGrayscale:
    denominator = sum(contributions)
    _channels = np.array(channels, int)
    _contributions = np.array(contributions) / denominator

    output_image = np.sum(
        image[:, :, _channels]
        * _contributions[np.newaxis, np.newaxis, :],
        2
    )
    return output_image

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def split_hsv(
        input_image: Annotated[Image2DColor, Field(description="Pixel data of image to be split")],
) -> Image2DGrayscale:
     output_image = matplotlib.colors.rgb_to_hsv(input_image)
     return output_image
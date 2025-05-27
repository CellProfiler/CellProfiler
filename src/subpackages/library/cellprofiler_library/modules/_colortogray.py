from pydantic import Field
from typing import Annotated, Any, Iterable
import numpy as np
import matplotlib.colors

def combine_colortogray(
    image: Annotated[Any, Field(description="Pixel data of image to threshold")],
    channels: Annotated[Iterable[int], Field(description="Array of integer identifier ")],
    contributions: Annotated[Iterable[float], Field(description="Array of contribution values")],
    ):
    denominator = sum(contributions)
    channels = np.array(channels, int)
    contributions = np.array(contributions) / denominator

    output_image = np.sum(
        image[:, :, channels]
        * contributions[np.newaxis, np.newaxis, :],
        2
    )
    return output_image

def split_hsv(
        input_image: Annotated[Any, Field(description="Pixel data of image to be split")],
):
     output_image = matplotlib.colors.rgb_to_hsv(input_image)
     return output_image
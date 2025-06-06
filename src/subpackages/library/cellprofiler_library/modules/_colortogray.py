from pydantic import Field, validate_call, ConfigDict
from typing import Annotated, Any, Iterable, Sequence, Union, Optional
import numpy as np

from cellprofiler_library.opts.colortogray import ImageChannelType
from ..types import Image2DColor, Image2DGrayscale
from ..functions.image_processing import combine_colortogray, split_hsv, split_rgb, split_multichannel

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def color_to_gray(
          image:              Annotated[Image2DColor, Field(description="Pixel data of image to threshold")],
          image_type:         Annotated[ImageChannelType, Field(description="Image type (RGB, HSV, or Channels)")],
          should_combine:     Annotated[bool, Field(description="Whether to combine or split the image")],
          channels:           Annotated[Optional[Sequence[int]], Field(description="Array of integer identifier for combining")],
          contributions:      Annotated[Optional[Sequence[float]], Field(description="Array of contribution values for combining")],
          ) -> Union[Image2DGrayscale, Sequence[Image2DGrayscale]]:
     if should_combine:
          if channels is None or contributions is None:
               raise ValueError("Must provide channels and contributions when combining")
          return combine_colortogray(image, channels, contributions)
     else:
          return split_colortogray(image, image_type)
     
def split_colortogray(input_image: Image2DColor, image_type:ImageChannelType = ImageChannelType.RGB) -> Sequence[Image2DGrayscale]:
     if image_type == ImageChannelType.RGB:
          return split_rgb(input_image) 
     elif image_type == ImageChannelType.HSV:
          return split_hsv(input_image)
     elif image_type == ImageChannelType.CHANNELS:
          return split_multichannel(input_image)
     else:
          raise ValueError(f"Unsupported image type: {image_type}")

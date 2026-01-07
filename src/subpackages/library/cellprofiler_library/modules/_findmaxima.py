import numpy
import scipy.ndimage
from skimage.feature import peak_local_max
from typing import Optional, Annotated, Union
from pydantic import validate_call, ConfigDict, Field
from cellprofiler_library.types import ImageAny, ObjectSegmentation, ImageBinary
from cellprofiler_library.opts.findmaxima import BackgroundExclusionMode


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def find_maxima(
        x_data:              Annotated[ImageAny, Field(description="Input image to search for maxima")],
        exclude_mode:        Annotated[BackgroundExclusionMode, Field(description="Method for excluding background")],
        min_distance_value:  Annotated[int, Field(description="Minimum distance between maxima")], #Minimum distance between maxima
        label_maxima:        Annotated[bool, Field(description="Individually label maxima?")],
        min_intensity_value: Annotated[Optional[float], Field(description="Specify the minimum intensity of a peak")],
        target_mask:         Annotated[Optional[Union[ImageAny,ObjectSegmentation]], Field(description="The image or objects to search within")], 
    ) -> Union[ImageBinary, ObjectSegmentation]:
    th_abs = min_intensity_value

    if exclude_mode.value == BackgroundExclusionMode.THRESHOLD.value:
        x_data = find_maxima_threshold(x_data)
    elif exclude_mode.value == BackgroundExclusionMode.MASK.value:
        assert target_mask is not None, "Mask image is required for mask mode"
        x_data = apply_target_mask(x_data, target_mask)
    elif exclude_mode.value == BackgroundExclusionMode.OBJECTS.value:
        assert target_mask is not None, "Objects are required for within objects mode"
        x_data = apply_target_mask(x_data, target_mask)
    else:
        raise NotImplementedError("Invalid background method choice")
    
    return find_maxima_in_data(x_data, min_distance_value, label_maxima, th_abs)


def find_maxima_in_data(
        x_data: ImageAny, 
        min_distance_value: int, 
        label_maxima: bool, 
        th_abs: Optional[float]=None
    ) -> Union[ImageBinary, ObjectSegmentation]:
    maxima_coords = peak_local_max(
        x_data,
        min_distance=min_distance_value,
        threshold_abs=th_abs,
    )
    y_data = numpy.zeros(x_data.shape, dtype=bool)
    y_data[tuple(maxima_coords.T)] = True

    if label_maxima:
        y_data = scipy.ndimage.label(y_data)[0]
    return y_data

def find_maxima_threshold(x_data: ImageAny) -> ImageAny:
    """Identity function, just returns the input"""
    return x_data

def apply_target_mask(x_data: ImageAny, masking_image: Union[ImageAny,ObjectSegmentation]) -> ImageAny:
    mask = masking_image.astype(bool)
    x_data[~mask] = 0
    return x_data



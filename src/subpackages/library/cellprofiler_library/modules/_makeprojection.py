from functools import partial
from collections.abc import Callable
from typing import Any, Tuple, Optional, Annotated, Union, cast
from typing_extensions import TypeAlias
from numpy.typing import NDArray
from pydantic import validate_call, ConfigDict, Field, BaseModel
import numpy as np
from cellprofiler_library.types import Image2D, Image2DMask
from ..opts.makeprojection import ProjectionType

STATE_NOT_INITIALIZED = "Invalid state key. Please initialize the state dictionary with a call to set_projection before calling this function"
NORM_IS_ZERO = "Norm is zero. Please check your input images"
REQ_IMG = "Image is required"
AGG_IMG_MISSING = "Aggregate image missing"
AGG_VSUM_MISSING = "Aggregate vsum missing"
AGG_VSQUARED_MISSING = "Aggregate vsquared missing"
AGG_POWER_MASK_MISSING = "Aggregate power mask missing"
AGG_POWER_IMAGE_MISSING = "Aggregate power image missing"
AGG_STACK_NUMBER_MISSING = "Aggregate stack number missing"
AGG_BRIGHT_MAX_MISSING = "Aggregate bright max mssing"
AGG_BRIGHT_MIN_MISSING = "Aggregate bright min mssing"
NORM0_MISSING = "Aggregate norm0 missing"
POWER_FREQUENCY_NOT_PROVIDED = "Frequency must be provided for Power projection"
AGG_IMG_MISTYPE = "Aggregate image must be bool mask"

T_PROJECTION_METHOD_INVALID = "Unknown projection method: %s"


ProjectionAccumulator: TypeAlias = Callable[
    [
        Image2D, # image
        Optional[Image2DMask], # mask
    ],
    "MakeProjectionAccumulator"
]
ProjectionFinalizer: TypeAlias = Callable[
    [],
    Tuple[Image2D, Image2DMask, NDArray[np.int_]]
]
class MakeProjectionAccumulator(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        populate_by_name=True
    )

    accumulate: ProjectionAccumulator
    finalize: ProjectionFinalizer

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def makeprojection(
        # set_projection
        method:     Annotated[ProjectionType, Field(description="The projection method")],
        # set_projection, accumulate_projection
        image:      Annotated[Image2D, Field(description="The pixel data of the image to accumulate")],
        mask:       Annotated[Optional[Image2DMask], Field(description="The mask of the image (True = valid). If None, all pixels are valid")] = None,
        frequency:  Annotated[float, Field(description="Frequency parameter for Power projection")] = 6.0,
    ) -> MakeProjectionAccumulator:
    has_mask = mask is not None
    if not has_mask:
        mask = np.ones(image.shape[:2], dtype=bool)
    agg_image_count = mask.astype(int)

    agg_vsum, agg_vsquared, agg_power_mask, agg_power_image, agg_stack_number, agg_bright_max, agg_bright_min, norm0, agg_image = (None,)*9
    if method == ProjectionType.VARIANCE:
        agg_vsum = image.copy()
        agg_vsum[~mask] = 0
        agg_vsquared = agg_vsum.astype(np.float64) ** 2.0

    elif method == ProjectionType.POWER:
        agg_vsum = image.copy()
        agg_vsum[~mask] = 0
        #
        # e**0 = 1 so the first image is always in the real plane
        #
        agg_power_mask = agg_image_count.astype(np.complex128).copy()
        agg_power_image = image.astype(np.complex128).copy()
        agg_stack_number = np.array(1.)

    elif method == ProjectionType.BRIGHTFIELD:
        agg_bright_max = image.copy()
        agg_bright_min = image.copy()
        norm0 = np.mean(image)

    elif method == ProjectionType.MASK:
        agg_image = mask

    elif method in (ProjectionType.AVERAGE, ProjectionType.SUM, ProjectionType.MAXIMUM, ProjectionType.MINIMUM):
        agg_image = image.copy()
        if has_mask:
            nan_value = 1 if method == ProjectionType.MINIMUM else 0
            agg_image[~mask] = nan_value

    else:
        raise ValueError(T_PROJECTION_METHOD_INVALID % method)

    return MakeProjectionAccumulator(
        accumulate = partial(
            accumulate_projection,
            method=method,
            agg_image_count=agg_image_count,
            agg_vsum=agg_vsum,
            agg_vsquared=agg_vsquared,
            agg_power_mask=agg_power_mask,
            agg_power_image=agg_power_image,
            frequency=frequency,
            agg_stack_number=agg_stack_number,
            agg_bright_max=agg_bright_max,
            agg_bright_min=agg_bright_min,
            norm0=norm0,
            agg_image=agg_image,
        ),
        finalize = partial(
            calculate_final_projection,
            method=method,
            agg_image_count=agg_image_count,
            agg_vsum=agg_vsum,
            agg_vsquared=agg_vsquared,
            agg_power_image=agg_power_image,
            agg_power_mask=agg_power_mask,
            agg_bright_max=agg_bright_max,
            agg_bright_min=agg_bright_min,
            agg_image=agg_image,
        )
    )

def accumulate_projection(
        image:      Annotated[Image2D, Field(description="The pixel data of the image to accumulate")],
        mask:       Annotated[Optional[Image2DMask], Field(description="The mask of the image (True = valid). If None, all pixels are valid")],
        *,
        method:           Annotated[ProjectionType, Field(description="The projection method")],
        agg_image_count:  Annotated[NDArray[np.int_], Field(description="Aggregation of image count")],
        agg_vsum:         Annotated[Optional[Image2D], Field(description="Aggregation of variance (for methods variance, power)")],
        agg_vsquared:     Annotated[Optional[NDArray[np.float64]], Field(description="Aggregation of squared variance (for method variance)")],
        agg_power_mask:   Annotated[Optional[NDArray[np.complex128]], Field(description="Aggregation of power mask (for method power)")],
        agg_power_image:  Annotated[Optional[NDArray[np.complex128]], Field(description="Aggregation of power image (for method power)")],
        frequency:        Annotated[float, Field(description="Frequency parameter for Power projection")],
        agg_stack_number: Annotated[Optional[NDArray[np.float64]], Field(description="Aggregation of stack number (for method power; zero dimensional)")],
        agg_bright_max:   Annotated[Optional[Image2D], Field(description="Aggregation of max brightfield vals (for method brightfield)")],
        agg_bright_min:   Annotated[Optional[Image2D], Field(description="Aggregation of min brightfield vals (for method brightfield)")],
        norm0:            Annotated[Optional[np.floating[Any]], Field(description="Normalization val (for method brightfield)")],
        agg_image:        Annotated[Optional[Union[Image2DMask, Image2D]], Field(description="Aggregation of image or mask (for methods mask, average, sum, maximum, minimum)")],
    ) -> MakeProjectionAccumulator:
    """
    Accumulate an image into the projection state.

    Args:
        image: The pixel data of the image to accumulate.
        mask: The mask of the image (True = valid). If None, all pixels are valid.
        state: The current accumulation state. Empty dict for first image.
        method: The projection method.
        frequency: Frequency parameter for Power projection.

    Returns:
        Updated state dictionary.
    """
    has_mask = mask is not None
    if has_mask:
        agg_image_count += mask.astype(int)
    else:
        agg_image_count += 1
    # Ensure mask exists
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)

    # Initialize if empty

    if method == ProjectionType.AVERAGE or method == ProjectionType.SUM:
        assert agg_image is not None, AGG_IMG_MISSING
        _mut_accumulate_sum(image, mask, agg_image)
    elif method == ProjectionType.MAXIMUM:
        assert agg_image is not None, AGG_IMG_MISSING
        _mut_accumulate_maximum(image, mask, agg_image)
    elif method == ProjectionType.MINIMUM:
        assert agg_image is not None, AGG_IMG_MISSING
        _mut_accumulate_minimum(image, mask, agg_image)
    elif method == ProjectionType.VARIANCE:
        assert agg_vsum is not None, AGG_VSUM_MISSING
        assert agg_vsquared is not None, AGG_VSQUARED_MISSING
        _mut_accumulate_variance(image, mask, agg_vsum, agg_vsquared)
    elif method == ProjectionType.POWER:
        assert frequency is not None, POWER_FREQUENCY_NOT_PROVIDED
        assert agg_vsum is not None, AGG_VSUM_MISSING
        assert agg_power_mask is not None, AGG_POWER_MASK_MISSING
        assert agg_power_image is not None, AGG_POWER_IMAGE_MISSING
        assert agg_stack_number is not None, AGG_STACK_NUMBER_MISSING
        _mut_accumulate_power(image, mask, frequency, agg_vsum, agg_power_mask, agg_power_image, agg_stack_number)
    elif method == ProjectionType.BRIGHTFIELD:
        assert agg_bright_max is not None, AGG_BRIGHT_MAX_MISSING
        assert agg_bright_min is not None, AGG_BRIGHT_MIN_MISSING
        assert norm0 is not None, NORM0_MISSING
        _mut_accumulate_brightfield(image, mask, norm0, agg_bright_max, agg_bright_min)
    elif method == ProjectionType.MASK:
        assert agg_image is not None, AGG_IMG_MISSING
        assert agg_image.dtype == np.bool_, AGG_IMG_MISTYPE
        _mut_accumulate_mask(mask, cast(Image2DMask, agg_image))
    else:
        raise ValueError(T_PROJECTION_METHOD_INVALID % method)

    return MakeProjectionAccumulator(
        accumulate = partial(
            accumulate_projection,
            method=method,
            agg_image_count=agg_image_count,
            agg_vsum=agg_vsum,
            agg_vsquared=agg_vsquared,
            agg_power_mask=agg_power_mask,
            agg_power_image=agg_power_image,
            frequency=frequency,
            agg_stack_number=agg_stack_number,
            agg_bright_max=agg_bright_max,
            agg_bright_min=agg_bright_min,
            norm0=norm0,
            agg_image=agg_image,
        ),
        finalize = partial(
            calculate_final_projection,
            method=method,
            agg_image_count=agg_image_count,
            agg_vsum=agg_vsum,
            agg_vsquared=agg_vsquared,
            agg_power_image=agg_power_image,
            agg_power_mask=agg_power_mask,
            agg_bright_max=agg_bright_max,
            agg_bright_min=agg_bright_min,
            agg_image=agg_image,
        )
    )

def calculate_final_projection(
        *,
        method:           Annotated[ProjectionType, Field(description="The projection method.")],
        agg_image_count:  Annotated[NDArray[np.int_], Field(description="Aggregation of image count")],
        agg_vsum:         Annotated[Optional[Image2D], Field(description="Aggregation of variance (for methods variance, power)")],
        agg_vsquared:     Annotated[Optional[NDArray[np.float64]], Field(description="Aggregation of squared variance (for method variance)")],
        agg_power_image:  Annotated[Optional[NDArray[np.complex128]], Field(description="Aggregation of power image (for method power)")],
        agg_power_mask:   Annotated[Optional[NDArray[np.complex128]], Field(description="Aggregation of power mask (for method power)")],
        agg_bright_max:   Annotated[Optional[Image2D], Field(description="Aggregation of max brightfield vals (for method brightfield)")],
        agg_bright_min:   Annotated[Optional[Image2D], Field(description="Aggregation of min brightfield vals (for method brightfield)")],
        agg_image:        Annotated[Optional[Union[Image2DMask, Image2D]], Field(description="Aggregation of image or mask (for methods mask, average, sum, maximum, minimum)")],
    ) -> Tuple[Image2D, Image2DMask, NDArray[np.int_]]:
    """
    Calculate the final projection image from the state.

    Args:
        state: The accumulation state.
        method: The projection method.

    Returns:
        Tuple of (pixel_data, mask).
    """
    mask_2d = agg_image_count > 0
    final_projection = None
    if method == ProjectionType.AVERAGE:
        assert agg_image is not None, AGG_IMG_MISSING
        final_projection = _finalize_average(agg_image, agg_image_count)
    elif method == ProjectionType.SUM:
        assert agg_image is not None, AGG_IMG_MISSING
        final_projection = _finalize_sum(agg_image, agg_image_count)
    elif method == ProjectionType.MAXIMUM:
        assert agg_image is not None, AGG_IMG_MISSING
        final_projection = _finalize_maximum(agg_image, agg_image_count)
    elif method == ProjectionType.MINIMUM:
        assert agg_image is not None, AGG_IMG_MISSING
        final_projection = _finalize_minimum(agg_image, agg_image_count)
    elif method == ProjectionType.VARIANCE:
        assert agg_vsum is not None, AGG_VSUM_MISSING
        assert agg_vsquared is not None, AGG_VSQUARED_MISSING
        final_projection = _finalize_variance(agg_vsum, agg_vsquared, agg_image_count)
    elif method == ProjectionType.POWER:
        assert agg_vsum is not None, AGG_VSUM_MISSING
        assert agg_power_mask is not None, AGG_POWER_MASK_MISSING
        assert agg_power_image is not None, AGG_POWER_IMAGE_MISSING
        final_projection = _finalize_power(agg_image_count, agg_power_image, agg_power_mask, agg_vsum)
    elif method == ProjectionType.BRIGHTFIELD:
        assert agg_bright_max is not None, AGG_BRIGHT_MAX_MISSING
        assert agg_bright_min is not None, AGG_BRIGHT_MIN_MISSING
        final_projection = _finalize_brightfield(agg_image_count, agg_bright_max, agg_bright_min)
    elif method == ProjectionType.MASK:
        assert agg_image is not None, AGG_IMG_MISSING
        final_projection = _finalize_mask(agg_image)
    else:
        raise ValueError(T_PROJECTION_METHOD_INVALID % method)

    return final_projection, mask_2d, agg_image_count

# --- Helper functions ---

def _mut_accumulate_sum(image: Image2D, mask: Image2DMask, agg_image: Union[Image2DMask, Image2D]):
    """This function is called by both sum and average projection methods"""
    agg_image[mask] += image[mask]

def _finalize_sum(agg_image: Union[Image2DMask, Image2D], agg_image_count: NDArray[np.int_]) -> Image2D:
    mask = agg_image_count > 0

    if np.any(~mask):
        cached_image = agg_image.copy()
        cached_image[~mask] = 0
    else:
        cached_image = agg_image

    return cached_image

def _finalize_average(agg_image: Union[Image2DMask, Image2D], agg_image_count: NDArray[np.int_]) -> Image2D:
    # Handle multi-channel image count broadcasting
    if agg_image.ndim == 3 and agg_image_count.ndim == 2:
        agg_image_count = np.dstack([agg_image_count] * agg_image.shape[2])

    mask = agg_image_count > 0

    # Avoid divide by zero
    cached_image = np.zeros_like(agg_image)
    valid = agg_image_count > 0
    cached_image[valid] = agg_image[valid] / agg_image_count[valid]

    if cached_image.ndim == 3 and mask.ndim == 2:
        cached_image[~mask, :] = 0
    else:
        cached_image[~mask] = 0

    return cached_image

def _mut_accumulate_maximum(image: Image2D, mask: Image2DMask, agg_image: Union[Image2DMask, Image2D]):
    agg_image[mask] = np.maximum(agg_image[mask], image[mask])

def _finalize_maximum(agg_image: Union[Image2DMask, Image2D], agg_image_count: NDArray[np.int_]) -> Image2D:
    # Same finalization logic as SumProvider except it uses the max accumulated image
    return _finalize_sum(agg_image, agg_image_count)

def _mut_accumulate_minimum(image: Image2D, mask: Image2DMask, agg_image: Union[Image2DMask, Image2D]):
    agg_image[mask] = np.minimum(agg_image[mask], image[mask])

def _finalize_minimum(agg_image: Union[Image2DMask, Image2D], agg_image_count: NDArray[np.int_]) -> Image2D:
    return _finalize_sum(agg_image, agg_image_count)

def _mut_accumulate_variance(image: Image2D, mask: Image2DMask, agg_vsum: Image2D, agg_vsquared: NDArray[np.float64]):
    agg_vsum[mask] += image[mask]
    agg_vsquared[mask] += image[mask].astype(np.float64) ** 2

def _finalize_variance(agg_vsum: Image2D, agg_vsquared: NDArray[np.float64], agg_image_count: NDArray[np.int_]) -> Image2D:
    if agg_vsquared.ndim == 3 and agg_image_count.ndim == 2:
        agg_image_count = np.dstack([agg_image_count] * agg_vsquared.shape[2])

    mask = agg_image_count > 0

    cached_image = np.zeros(agg_vsquared.shape, np.float32)

    # Calculate variance: E[x^2] - (E[x])^2

    valid = mask # logic alias
    cached_image[valid] = agg_vsquared[valid] / agg_image_count[valid]
    cached_image[valid] -= (agg_vsum[valid] ** 2) / (agg_image_count[valid] ** 2)

    cached_image[~mask] = 0

    return cached_image

def _mut_accumulate_power(
        image: Image2D,
        mask: Image2DMask,
        frequency: float,
        agg_vsum: Image2D,
        agg_power_mask: NDArray[np.complex128],
        agg_power_image: NDArray[np.complex128],
        agg_stack_number: NDArray[np.float64],
    ):
    multiplier = np.exp(2j * np.pi * agg_stack_number / frequency)
    agg_stack_number += 1

    agg_vsum[mask] += image[mask]
    agg_power_image[mask] += multiplier * image[mask]
    agg_power_mask[mask] += multiplier

def _finalize_power(
        agg_image_count: NDArray[np.int_],
        agg_power_image: NDArray[np.complex128],
        agg_power_mask: NDArray[np.complex128],
        agg_vsum: Image2D,
    ) -> Image2D:
    if agg_power_image.ndim == 3 and agg_image_count.ndim == 2:
        agg_image_count = np.dstack([agg_image_count] * agg_power_image.shape[2])

    mask = agg_image_count > 0

    cached_image = np.zeros(agg_image_count.shape, np.complex128)
    cached_image[mask] = agg_power_image[mask]
    cached_image[mask] -= (agg_vsum[mask] * agg_power_mask[mask] / agg_image_count[mask])

    # |z|^2 = z * conj(z)
    cached_image = (cached_image * np.conj(cached_image)).real.astype(np.float32)
    cached_image[~mask] = 0

    return cached_image

def _mut_accumulate_brightfield(image: Image2D, mask: Image2DMask, norm0: np.floating[Any], agg_bright_max: Image2D, agg_bright_min: Image2D):
    norm = np.mean(image)
    assert norm != 0, NORM_IS_ZERO
    pixel_data = image * norm0 / norm

    max_mask = (agg_bright_max < pixel_data) & mask
    min_mask = (agg_bright_min > pixel_data) & mask

    agg_bright_min[min_mask] = pixel_data[min_mask]
    agg_bright_max[max_mask] = pixel_data[max_mask]
    agg_bright_min[max_mask] = agg_bright_max[max_mask]

def _finalize_brightfield(agg_image_count: NDArray[np.int_], agg_bright_max: Image2D, agg_bright_min: Image2D) -> Image2D:
    if agg_bright_max.ndim == 3 and agg_image_count.ndim == 2:
        agg_image_count = np.dstack([agg_image_count] * agg_bright_max.shape[2])

    mask = agg_image_count > 0

    cached_image = np.zeros(agg_image_count.shape, np.float32)
    cached_image[mask] = agg_bright_max[mask] - agg_bright_min[mask]
    cached_image[~mask] = 0

    return cached_image

def _mut_accumulate_mask(mask: Image2DMask, agg_image: Image2DMask):
    agg_image &= mask

def _finalize_mask(agg_image: Union[Image2DMask, Image2D]) -> Image2D:
    return agg_image

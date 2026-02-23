from pydantic import validate_call, ConfigDict, Field
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, Annotated
from cellprofiler_library.types import Image2D, Image2DMask
from ..opts.makeprojection import ProjectionType
from ..opts.makeprojection import StateKey

STATE_NOT_INITIALIZED = "Invalid state key. Please initialize the state dictionary with a call to set_projection before calling this function"
POWER_FREQUENCY_NOT_PROVIDED = "Frequency must be provided for Power projection"
PROJECTION_METHOD_INVALID = "Unknown projection method: %s"
NORM_IS_ZERO = "Norm is zero. Please check your input images"

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def accumulate_projection(
    image:      Annotated[Image2D, Field(description="The pixel data of the image to accumulate")],
    mask:       Annotated[Optional[Image2DMask], Field(description="The mask of the image (True = valid). If None, all pixels are valid")],
    state:      Annotated[Dict[str, Any], Field(description="The current accumulation state. Empty dict for first image")],
    method:     Annotated[ProjectionType, Field(description="The projection method")],
    frequency:  Annotated[Optional[float], Field(description="Frequency parameter for Power projection")] = 6.0
) -> Dict[str, Any]:
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
    assert StateKey.IMAGE_COUNT.value in state, STATE_NOT_INITIALIZED
    has_mask = mask is not None
    if has_mask:
        state[StateKey.IMAGE_COUNT.value] += mask.astype(int)
    else:
        state[StateKey.IMAGE_COUNT.value] += 1
    # Ensure mask exists
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)
        
    # Initialize if empty

    if method == ProjectionType.AVERAGE or method == ProjectionType.SUM:
        _accumulate_sum(image, mask, state)
    elif method == ProjectionType.MAXIMUM:
        _accumulate_maximum(image, mask, state)
    elif method == ProjectionType.MINIMUM:
        _accumulate_minimum(image, mask, state)
    elif method == ProjectionType.VARIANCE:
        _accumulate_variance(image, mask, state)
    elif method == ProjectionType.POWER:
        assert frequency is not None, POWER_FREQUENCY_NOT_PROVIDED
        _accumulate_power(image, mask, state, frequency)
    elif method == ProjectionType.BRIGHTFIELD:
        _accumulate_brightfield(image, mask, state)
    elif method == ProjectionType.MASK:
        _accumulate_mask(image, mask, state)
    else:
        raise ValueError(f"Unknown projection method: {method}")
        
    return state

def set_projection(
        image:  Annotated[Image2D, Field(description="The pixel data of the image to accumulate.")],
        mask:   Annotated[Optional[Image2DMask], Field(description="The mask of the image (True = valid). If None, all pixels are valid")],
        state:  Annotated[Dict[str, Any], Field(description="The current accumulation state. Empty dict for first image")],
        method: Annotated[ProjectionType, Field(description="The projection method")],
    ) -> Dict[str, Any]:
    has_mask = mask is not None
    if not has_mask:
        mask = np.ones(image.shape[:2], dtype=bool)
    state[StateKey.IMAGE_COUNT.value] = mask.astype(int)
    
    if method == ProjectionType.VARIANCE:
        state[StateKey.VSUM.value] = image.copy()
        state[StateKey.VSUM.value][~mask] = 0
        state[StateKey.VSQUARED.value] = state[StateKey.VSUM.value].astype(np.float64) ** 2.0

    elif method == ProjectionType.POWER:
        state[StateKey.VSUM.value] = image.copy()
        state[StateKey.VSUM.value][~mask] = 0
        #
        # e**0 = 1 so the first image is always in the real plane
        #
        state[StateKey.POWER_MASK.value] = state[StateKey.IMAGE_COUNT.value].astype(np.complex128).copy()
        state[StateKey.POWER_IMAGE.value] = image.astype(np.complex128).copy()
        state[StateKey.STACK_NUMBER.value] = 1

    elif method == ProjectionType.BRIGHTFIELD:
        state[StateKey.BRIGHT_MAX.value] = image.copy()
        state[StateKey.BRIGHT_MIN.value] = image.copy()
        state[StateKey.NORM0.value] = np.mean(image)

    elif method == ProjectionType.MASK:
        state[StateKey.IMAGE.value] = mask

    elif method in (ProjectionType.AVERAGE, ProjectionType.SUM, ProjectionType.MAXIMUM, ProjectionType.MINIMUM):
        state[StateKey.IMAGE.value] = image.copy()
        if has_mask:
            nan_value = 1 if method == ProjectionType.MINIMUM else 0
            state[StateKey.IMAGE.value][~mask] = nan_value

    else:
        raise ValueError(PROJECTION_METHOD_INVALID % method)
    
    return state

    

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def calculate_final_projection(
    state: Annotated[Dict[str, Any], Field(description="The accumulation state.")],
    method: Annotated[ProjectionType, Field(description="The projection method.")]
) -> Tuple[Image2D, Image2DMask]:
    """
    Calculate the final projection image from the state.
    
    Args:
        state: The accumulation state.
        method: The projection method.
        
    Returns:
        Tuple of (pixel_data, mask).
    """
    
    image_count = state[StateKey.IMAGE_COUNT.value]
    mask_2d = image_count > 0
    final_projection = None
    if method == ProjectionType.AVERAGE:
        final_projection = _finalize_average(state)
    elif method == ProjectionType.SUM:
        final_projection = _finalize_sum(state)
    elif method == ProjectionType.MAXIMUM:
        final_projection = _finalize_maximum(state)
    elif method == ProjectionType.MINIMUM:
        final_projection = _finalize_minimum(state)
    elif method == ProjectionType.VARIANCE:
        final_projection = _finalize_variance(state)
    elif method == ProjectionType.POWER:
        final_projection = _finalize_power(state)
    elif method == ProjectionType.BRIGHTFIELD:
        final_projection = _finalize_brightfield(state)
    elif method == ProjectionType.MASK:
        final_projection = _finalize_mask(state)
    else:
        raise ValueError(PROJECTION_METHOD_INVALID % method)

    return final_projection, mask_2d

# --- Helper functions ---

def _accumulate_sum(image: Image2D, mask: Image2DMask, state: Dict[str, Any]):
    """This function is called by both sum and average projection methods"""
    assert StateKey.IMAGE.value in state, STATE_NOT_INITIALIZED
    state[StateKey.IMAGE.value][mask] += image[mask]


def _finalize_sum(state: Dict[str, Any]) -> Image2D:
    image = state[StateKey.IMAGE.value]
    image_count = state[StateKey.IMAGE_COUNT.value]
    
    mask = image_count > 0
    
    if np.any(~mask):
        cached_image = image.copy()
        cached_image[~mask] = 0
    else:
        cached_image = image
        
    return cached_image

def _finalize_average(state: Dict[str, Any]) -> Image2D:
    image = state[StateKey.IMAGE.value]
    image_count = state[StateKey.IMAGE_COUNT.value]
    # Handle multi-channel image count broadcasting
    if image.ndim == 3 and image_count.ndim == 2:
        image_count = np.dstack([image_count] * image.shape[2])
        
    mask = image_count > 0
    
    # Avoid divide by zero
    cached_image = np.zeros_like(image)
    valid = image_count > 0
    cached_image[valid] = image[valid] / image_count[valid]
    
    if cached_image.ndim == 3 and mask.ndim == 2:
        cached_image[~mask, :] = 0
    else:
        cached_image[~mask] = 0
        
    return cached_image

def _accumulate_maximum(image: Image2D, mask: Image2DMask, state: Dict[str, Any]):
    assert StateKey.IMAGE.value in state, STATE_NOT_INITIALIZED
    current_image = state[StateKey.IMAGE.value]
    state[StateKey.IMAGE.value][mask] = np.maximum(current_image[mask], image[mask])

def _finalize_maximum(state: Dict[str, Any]) -> Image2D:
    # Same finalization logic as SumProvider except it uses the max accumulated image
    return _finalize_sum(state)

def _accumulate_minimum(image: Image2D, mask: Image2DMask, state: Dict[str, Any]):
    assert StateKey.IMAGE.value in state, STATE_NOT_INITIALIZED
    current_image = state[StateKey.IMAGE.value]
    state[StateKey.IMAGE.value][mask] = np.minimum(current_image[mask], image[mask])

def _finalize_minimum(state: Dict[str, Any]) -> Image2D:
    return _finalize_sum(state)

def _accumulate_variance(image: Image2D, mask: Image2DMask, state: Dict[str, Any]):
    assert StateKey.VSUM.value in state, STATE_NOT_INITIALIZED
    assert StateKey.VSQUARED.value in state, STATE_NOT_INITIALIZED
    state[StateKey.VSUM.value][mask] += image[mask]
    state[StateKey.VSQUARED.value][mask] += image[mask].astype(np.float64) ** 2

def _finalize_variance(state: Dict[str, Any]) -> Image2D:
    vsum = state[StateKey.VSUM.value]
    vsquared = state[StateKey.VSQUARED.value]
    image_count = state[StateKey.IMAGE_COUNT.value]
    
    if vsquared.ndim == 3 and image_count.ndim == 2:
        image_count = np.dstack([image_count] * vsquared.shape[2])
        
    mask = image_count > 0
    
    cached_image = np.zeros(vsquared.shape, np.float32)
    
    # Calculate variance: E[x^2] - (E[x])^2
    
    valid = mask # logic alias
    cached_image[valid] = vsquared[valid] / image_count[valid]
    cached_image[valid] -= (vsum[valid] ** 2) / (image_count[valid] ** 2)
    
    cached_image[~mask] = 0
    
    return cached_image

def _accumulate_power(image: Image2D, mask: Image2DMask, state: Dict[str, Any], frequency: float):     
    assert StateKey.VSUM.value in state, STATE_NOT_INITIALIZED
    assert StateKey.POWER_MASK.value in state, STATE_NOT_INITIALIZED
    assert StateKey.POWER_IMAGE.value in state, STATE_NOT_INITIALIZED
    assert StateKey.STACK_NUMBER.value in state, STATE_NOT_INITIALIZED

    stack_number = state[StateKey.STACK_NUMBER.value]
    multiplier = np.exp(2j * np.pi * float(stack_number) / frequency)
    state[StateKey.STACK_NUMBER.value] += 1
    
    state[StateKey.VSUM.value][mask] += image[mask]
    state[StateKey.POWER_IMAGE.value][mask] += multiplier * image[mask]
    state[StateKey.POWER_MASK.value][mask] += multiplier

def _finalize_power(state: Dict[str, Any]) -> Image2D:
    image_count = state[StateKey.IMAGE_COUNT.value]
    power_image = state[StateKey.POWER_IMAGE.value]
    vsum = state[StateKey.VSUM.value]
    power_mask = state[StateKey.POWER_MASK.value]
    
    if power_image.ndim == 3 and image_count.ndim == 2:
        image_count = np.dstack([image_count] * power_image.shape[2])
        
    mask = image_count > 0
    
    cached_image = np.zeros(image_count.shape, np.complex128)
    cached_image[mask] = power_image[mask]
    cached_image[mask] -= (vsum[mask] * power_mask[mask] / image_count[mask])
    
    # |z|^2 = z * conj(z)
    cached_image = (cached_image * np.conj(cached_image)).real.astype(np.float32)
    cached_image[~mask] = 0
    
    return cached_image

def _accumulate_brightfield(image: Image2D, mask: Image2DMask, state: Dict[str, Any]):
    assert StateKey.BRIGHT_MAX.value in state, STATE_NOT_INITIALIZED
    assert StateKey.BRIGHT_MIN.value in state, STATE_NOT_INITIALIZED
    assert StateKey.NORM0.value in state, STATE_NOT_INITIALIZED

    norm0 = state[StateKey.NORM0.value]
    bright_max = state[StateKey.BRIGHT_MAX.value]
    bright_min = state[StateKey.BRIGHT_MIN.value]
    
    norm = np.mean(image)
    assert norm != 0, NORM_IS_ZERO
    pixel_data = image * norm0 / norm
    
    max_mask = (bright_max < pixel_data) & mask
    min_mask = (bright_min > pixel_data) & mask
    
    bright_min[min_mask] = pixel_data[min_mask]
    bright_max[max_mask] = pixel_data[max_mask]
    bright_min[max_mask] = bright_max[max_mask]
    state[StateKey.BRIGHT_MAX.value] = bright_max
    state[StateKey.BRIGHT_MIN.value] = bright_min  

def _finalize_brightfield(state: Dict[str, Any]) -> Image2D:
    image_count = state[StateKey.IMAGE_COUNT.value]
    bright_max = state[StateKey.BRIGHT_MAX.value]
    bright_min = state[StateKey.BRIGHT_MIN.value]
    
    if bright_max.ndim == 3 and image_count.ndim == 2:
        image_count = np.dstack([image_count] * bright_max.shape[2])
        
    mask = image_count > 0
    
    cached_image = np.zeros(image_count.shape, np.float32)
    cached_image[mask] = bright_max[mask] - bright_min[mask]
    cached_image[~mask] = 0
    
    return cached_image

def _accumulate_mask(image: Image2D, mask: Image2DMask, state: Dict[str, Any]):
    assert StateKey.IMAGE.value in state, STATE_NOT_INITIALIZED
    state[StateKey.IMAGE.value] = state[StateKey.IMAGE.value] & mask

def _finalize_mask(state: Dict[str, Any]) -> Image2D:
    final_mask = state[StateKey.IMAGE.value]
    return final_mask


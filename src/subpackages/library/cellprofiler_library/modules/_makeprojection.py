from pydantic import validate_call, ConfigDict
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from ..opts.makeprojection import ProjectionType
from ..opts.makeprojection import StateKey


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def accumulate_projection(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    state: Dict[str, Any],
    method: ProjectionType,
    frequency: float = 6.0
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
    # Ensure mask exists
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)
        
    # Initialize if empty
    is_first = len(state) == 0

    if method == ProjectionType.AVERAGE or method == ProjectionType.SUM:
        _accumulate_sum(image, mask, state, is_first)
    elif method == ProjectionType.MAXIMUM:
        _accumulate_maximum(image, mask, state, is_first)
    elif method == ProjectionType.MINIMUM:
        _accumulate_minimum(image, mask, state, is_first)
    elif method == ProjectionType.VARIANCE:
        _accumulate_variance(image, mask, state, is_first)
    elif method == ProjectionType.POWER:
        _accumulate_power(image, mask, state, is_first, frequency)
    elif method == ProjectionType.BRIGHTFIELD:
        _accumulate_brightfield(image, mask, state, is_first)
    elif method == ProjectionType.MASK:
        _accumulate_mask(image, mask, state, is_first)
    else:
        raise ValueError(f"Unknown projection method: {method}")
        
    return state

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def calculate_final_projection(
    state: Dict[str, Any],
    method: ProjectionType
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the final projection image from the state.
    
    Args:
        state: The accumulation state.
        method: The projection method.
        
    Returns:
        Tuple of (pixel_data, mask).
    """
    if method == ProjectionType.AVERAGE:
        return _finalize_average(state)
    elif method == ProjectionType.SUM:
        return _finalize_sum(state)
    elif method == ProjectionType.MAXIMUM:
        return _finalize_maximum(state)
    elif method == ProjectionType.MINIMUM:
        return _finalize_minimum(state)
    elif method == ProjectionType.VARIANCE:
        return _finalize_variance(state)
    elif method == ProjectionType.POWER:
        return _finalize_power(state)
    elif method == ProjectionType.BRIGHTFIELD:
        return _finalize_brightfield(state)
    elif method == ProjectionType.MASK:
        return _finalize_mask(state)
    else:
        raise ValueError(f"Unknown projection method: {method}")

# --- Helper functions ---

def _wrap_result(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to ensure mask compatibility and return tuple"""
    # Original logic: _wrap_image
    # If mask is 3D, take first slice
    final_mask = mask
    if mask.ndim == 3:
        final_mask = mask[:, :, 0]
    
    # Original logic checks numpy.all(mask) but here we return mask always
    return image, final_mask

def _accumulate_sum(image: np.ndarray, mask: np.ndarray, state: Dict[str, Any], is_first: bool):
    if is_first:
        state[StateKey.IMAGE_COUNT.value] = mask.astype(int)
        
        # _set_image_impl for Sum
        img_copy = image.copy()
        img_copy[~mask] = 0
        state[StateKey.IMAGE.value] = img_copy
    else:
        state[StateKey.IMAGE_COUNT.value] += mask.astype(int)
        
        state[StateKey.IMAGE.value][mask] += image[mask]

def _finalize_sum(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    image = state[StateKey.IMAGE.value]
    image_count = state[StateKey.IMAGE_COUNT.value]
    
    mask = image_count > 0
    
    if np.any(~mask):
        cached_image = image.copy()
        cached_image[~mask] = 0
    else:
        cached_image = image
        
    return _wrap_result(cached_image, mask)

def _finalize_average(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
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
        
    return _wrap_result(cached_image, mask)

def _accumulate_maximum(image: np.ndarray, mask: np.ndarray, state: Dict[str, Any], is_first: bool):
    if is_first:
        state[StateKey.IMAGE_COUNT.value] = mask.astype(int)
        
        img_copy = image.copy()
        img_copy[~mask] = 0
        state[StateKey.IMAGE.value] = img_copy
    else:
        state[StateKey.IMAGE_COUNT.value] += mask.astype(int)
        
        # _accumulate_image_impl
        current_image = state[StateKey.IMAGE.value]
        state[StateKey.IMAGE.value][mask] = np.maximum(current_image[mask], image[mask])

def _finalize_maximum(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    # Same finalization logic as SumProvider except it uses the max accumulated image
    return _finalize_sum(state)

def _accumulate_minimum(image: np.ndarray, mask: np.ndarray, state: Dict[str, Any], is_first: bool):
    if is_first:
        state[StateKey.IMAGE_COUNT.value] = mask.astype(int)
        
        img_copy = image.copy()
        # Set all masked pixels to 1 so that they are not included in the minimum
        img_copy[~mask] = 1 
        state[StateKey.IMAGE.value] = img_copy
    else:
        state[StateKey.IMAGE_COUNT.value] += mask.astype(int)
        
        current_image = state[StateKey.IMAGE.value]
        state[StateKey.IMAGE.value][mask] = np.minimum(current_image[mask], image[mask])

def _finalize_minimum(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    return _finalize_sum(state)

def _accumulate_variance(image: np.ndarray, mask: np.ndarray, state: Dict[str, Any], is_first: bool):
    if is_first:
        state[StateKey.IMAGE_COUNT.value] = mask.astype(int)
        
        vsum = image.copy()
        vsum[~mask] = 0
        state[StateKey.VSUM.value] = vsum
        state[StateKey.VSQUARED.value] = vsum.astype(np.float64) ** 2.0
    else:
        state[StateKey.IMAGE_COUNT.value] += mask.astype(int)
        
        state[StateKey.VSUM.value][mask] += image[mask]
        state[StateKey.VSQUARED.value][mask] += image[mask].astype(np.float64) ** 2

def _finalize_variance(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
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
    
    return _wrap_result(cached_image, mask)

def _accumulate_power(image: np.ndarray, mask: np.ndarray, state: Dict[str, Any], is_first: bool, frequency: float):
    if is_first:
        state[StateKey.IMAGE_COUNT.value] = mask.astype(int)
        
        vsum = image.copy()
        vsum[~mask] = 0
        state[StateKey.VSUM.value] = vsum
        
        state[StateKey.POWER_MASK.value] = state[StateKey.IMAGE_COUNT.value].astype(np.complex128).copy()
        state[StateKey.POWER_IMAGE.value] = image.astype(np.complex128).copy()
        state[StateKey.STACK_NUMBER.value] = 1
    else:
        state[StateKey.IMAGE_COUNT.value] += mask.astype(int)
        
        stack_number = state[StateKey.STACK_NUMBER.value]
        multiplier = np.exp(2j * np.pi * float(stack_number) / frequency)
        state[StateKey.STACK_NUMBER.value] += 1
        
        state[StateKey.VSUM.value][mask] += image[mask]
        state[StateKey.POWER_IMAGE.value][mask] += multiplier * image[mask]
        state[StateKey.POWER_MASK.value][mask] += multiplier

def _finalize_power(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
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
    
    return _wrap_result(cached_image, mask)

def _accumulate_brightfield(image: np.ndarray, mask: np.ndarray, state: Dict[str, Any], is_first: bool):
    if is_first:
        state[StateKey.IMAGE_COUNT.value] = mask.astype(int)
        
        state[StateKey.BRIGHT_MAX.value] = image.copy()
        state[StateKey.BRIGHT_MIN.value] = image.copy()
        state[StateKey.NORM0.value] = np.mean(image)
    else:
        state[StateKey.IMAGE_COUNT.value] += mask.astype(int)
        
        norm0 = state[StateKey.NORM0.value]
        bright_max = state[StateKey.BRIGHT_MAX.value]
        bright_min = state[StateKey.BRIGHT_MIN.value]
        
        norm = np.mean(image)
        assert norm != 0, "norm is zero"
        pixel_data = image * norm0 / norm
        
        max_mask = (bright_max < pixel_data) & mask
        min_mask = (bright_min > pixel_data) & mask
        
        bright_min[min_mask] = pixel_data[min_mask]
        bright_max[max_mask] = pixel_data[max_mask]
        # Original: self._bright_min[max_mask] = self._bright_max[max_mask]
        bright_min[max_mask] = bright_max[max_mask]

def _finalize_brightfield(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    image_count = state[StateKey.IMAGE_COUNT.value]
    bright_max = state[StateKey.BRIGHT_MAX.value]
    bright_min = state[StateKey.BRIGHT_MIN.value]
    
    if bright_max.ndim == 3 and image_count.ndim == 2:
        image_count = np.dstack([image_count] * bright_max.shape[2])
        
    mask = image_count > 0
    
    cached_image = np.zeros(image_count.shape, np.float32)
    cached_image[mask] = bright_max[mask] - bright_min[mask]
    cached_image[~mask] = 0
    
    return _wrap_result(cached_image, mask)

def _accumulate_mask(image: np.ndarray, mask: np.ndarray, state: Dict[str, Any], is_first: bool):
    # For MaskProvider, "image" is actually the mask.
    # So we are accumulating MASKS, not pixel data.
    
    if is_first:
        state[StateKey.IMAGE_COUNT.value] = mask.astype(int)
        state[StateKey.IMAGE.value] = mask # Accumulating the mask into StateKey.IMAGE.value
    else:
        state[StateKey.IMAGE.value] = state[StateKey.IMAGE.value] & mask

def _finalize_mask(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    final_mask = state[StateKey.IMAGE.value]
    # MaskProvider returns Image(self._image), so pixel data IS the mask.    
    return final_mask, np.ones(final_mask.shape, dtype=bool)


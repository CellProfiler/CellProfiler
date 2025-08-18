import logging
from typing import Annotated, Optional, Union, List, Dict, Any

import numpy
from numpy.typing import NDArray
from pydantic import validate_call, Field, ConfigDict

from ..opts.morph import MorphFunction, RepeatMethod
from ..functions.image_processing import (
    apply_branchpoints,
    apply_bridge, 
    apply_clean,
    apply_convex_hull,
    apply_diag,
    apply_distance,
    apply_endpoints,
    apply_fill,
    apply_hbreak,
    apply_majority,
    apply_openlines,
    apply_remove,
    apply_shrink,
    apply_skelpe,
    apply_spur,
    apply_thicken,
    apply_thin,
    apply_vbreak,
)
from ..types import ImageGrayscale, ImageGrayscaleMask

LOGGER = logging.getLogger(__name__)


def morph_operation(
        pixel_data: Annotated[Union[ImageGrayscale, ImageGrayscaleMask], Field(description="Image array supporting binary, integer, or float types")],
        mask: Annotated[Optional[ImageGrayscaleMask], Field(description="Optional boolean mask array")],
        function_name: Annotated[MorphFunction, Field(description="Morphological operation to perform")],
        repeat_count: Annotated[int, Field(description="Number of times to repeat the operation", ge=1)],
        custom_repeats: Annotated[int, Field(description="Custom repeat value for specific operations", ge=1)],
        rescale_values: Annotated[bool, Field(description="Whether to rescale distance values from 0 to 1")],
        ) -> NDArray[numpy.float64]:
    """Apply a morphological operation to the image, routing to appropriate function from functions layer."""
    count = repeat_count

    is_binary = pixel_data.dtype.kind == "b"

    if (
        function_name
        in (
            MorphFunction.BRANCHPOINTS,
            MorphFunction.BRIDGE,
            MorphFunction.CLEAN,
            MorphFunction.DIAG,
            MorphFunction.CONVEX_HULL,
            MorphFunction.DISTANCE,
            MorphFunction.ENDPOINTS,
            MorphFunction.FILL,
            MorphFunction.HBREAK,
            MorphFunction.MAJORITY,
            MorphFunction.REMOVE,
            MorphFunction.SHRINK,
            MorphFunction.SKELPE,
            MorphFunction.SPUR,
            MorphFunction.THICKEN,
            MorphFunction.THIN,
            MorphFunction.VBREAK,
        )
        and not is_binary
    ):
        # Apply a very crude threshold to the image for binary algorithms
        LOGGER.warning(
            "Warning: converting image to binary for %s\n" % function_name
        )
        pixel_data = pixel_data != 0

    # Route to appropriate function from functions layer
    if function_name == MorphFunction.BRANCHPOINTS:
        return apply_branchpoints(pixel_data, mask)
    elif function_name == MorphFunction.BRIDGE:
        return apply_bridge(pixel_data, mask, count)
    elif function_name == MorphFunction.CLEAN:
        return apply_clean(pixel_data, mask, count)
    elif function_name == MorphFunction.CONVEX_HULL:
        return apply_convex_hull(pixel_data, mask)
    elif function_name == MorphFunction.DIAG:
        return apply_diag(pixel_data, mask, count)
    elif function_name == MorphFunction.DISTANCE:
        return apply_distance(pixel_data, rescale_values)
    elif function_name == MorphFunction.ENDPOINTS:
        return apply_endpoints(pixel_data, mask)
    elif function_name == MorphFunction.FILL:
        return apply_fill(pixel_data, mask, count)
    elif function_name == MorphFunction.HBREAK:
        return apply_hbreak(pixel_data, mask, count)
    elif function_name == MorphFunction.MAJORITY:
        return apply_majority(pixel_data, mask, count)
    elif function_name == MorphFunction.OPENLINES:
        return apply_openlines(pixel_data, mask, custom_repeats)
    elif function_name == MorphFunction.REMOVE:
        return apply_remove(pixel_data, mask, count)
    elif function_name == MorphFunction.SHRINK:
        return apply_shrink(pixel_data, count)
    elif function_name == MorphFunction.SKELPE:
        return apply_skelpe(pixel_data, mask)
    elif function_name == MorphFunction.SPUR:
        return apply_spur(pixel_data, mask, count)
    elif function_name == MorphFunction.THICKEN:
        return apply_thicken(pixel_data, mask, count)
    elif function_name == MorphFunction.THIN:
        return apply_thin(pixel_data, mask, count)
    elif function_name == MorphFunction.VBREAK:
        return apply_vbreak(pixel_data, mask)
    else:
        raise NotImplementedError(
            "Unimplemented morphological function: %s" % function_name
        )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def apply_morphological_operations(
        pixel_data: Annotated[Union[ImageGrayscale, ImageGrayscaleMask], Field(description="Image array supporting binary, integer, or float types")],
        mask: Annotated[Optional[ImageGrayscaleMask], Field(description="Optional boolean mask array")] = None,
        operations_list: Annotated[List[Dict[str, Any]], Field(description="List of morphological operations to apply sequentially")] = [],
        ) -> NDArray[numpy.float64]:
    """Apply a sequence of morphological operations to the image.
    
    Args:
        pixel_data: Input image pixel data to process
        mask: Optional image mask to constrain operations
        operations_list: List of operation dictionaries, each containing:
            - function_name: Name of morphological operation (MorphFunction enum)
            - repeat_count: Number of times to repeat the operation
            - custom_repeats: Custom repeat value for specific operations
            - rescale_values: Whether to rescale distance values from 0 to 1
    
    Returns:
        Processed pixel data array with applied morphological operations
    """
    result_pixel_data = pixel_data
    
    for operation in operations_list:
        result_pixel_data = morph_operation(
            result_pixel_data,
            mask,
            operation["function_name"],
            operation["repeat_count"],
            operation["custom_repeats"],
            operation["rescale_values"]
        )
    
    return result_pixel_data

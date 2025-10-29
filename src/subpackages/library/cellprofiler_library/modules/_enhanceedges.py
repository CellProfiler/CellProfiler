import warnings

import numpy
import centrosome
from pydantic import Field, validate_call, ConfigDict
from typing import Annotated, Optional
from ..functions.image_processing import (
    enhance_edges_sobel,
    enhance_edges_log,
    enhance_edges_prewitt,
    enhance_edges_canny,
)
from ..opts.enhanceedges import EdgeFindingMethod, EdgeDirection
from ..types import Image2DGrayscale, Image2DGrayscaleMask

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def enhanceedges(
    image:                          Annotated[Image2DGrayscale, Field(description="Input image")],
    mask:                           Annotated[Optional[Image2DGrayscaleMask], Field(description="Mask of the input image")] = None,
    method:                         Annotated[EdgeFindingMethod, Field(description="Edge finding method")] = EdgeFindingMethod.SOBEL,
    automatic_threshold:            Annotated[bool, Field(description="Automatically calculate the threshold?")] = True,
    direction:                      Annotated[EdgeDirection, Field(description="Select edge direction to enhance")] = EdgeDirection.ALL,
    sigma:                          Annotated[int, Field(description="Gaussian's sigma value")] = 10,
    manual_threshold:               Annotated[float, Field(description="Absolute threshold")] = 0.2,
    threshold_adjustment_factor:    Annotated[float, Field(description="Threshold adjustment factor")] = 1.0,
    automatic_low_threshold:        Annotated[bool, Field(description="Automatically calculate the low / soft threshold cutoff for the Canny method?")] = True,
    low_threshold:                  Annotated[float, Field(description="Low threshold value")] = 0.1,
) -> Image2DGrayscale:
    """EnhanceEdges module

    Parameters
    ----------
    image : numpy.array
        Input image
    mask : numpy.array, optional
        Boolean mask, by default None
    method : str, optional
        Enhance edges algorithm to apply to the input image, by default "sobel"
    direction : str, optional
        Applicable to only the Sobel and Prewitt algorithms, by default "all"
    sigma : int, optional
        Applicable to only the Canny and Laplacian of Gaussian algorithms, by default 10. Only considered if automatic_gaussian is False.
    automatic_threshold : bool, optional
        Applicable only to the Canny algorithm, by default True
    manual_threshold : float, optional
        Applicable only to the Canny algorithm, by default 0.2
    threshold_adjustment_factor : float, optional
        Applicable only to the Canny algorithm, by default 1.0
    automatic_low_threshold : bool, optional
        Applicable only to the Canny algorithm, by default True
    low_threshold : float, optional
        Applicable only to the Canny algorithm, by default 0.1

    Returns
    -------
    numpy.array
        Image with enhanced edges
    """

    if not 0 <= low_threshold <= 1:
        warnings.warn(
            f"""low_threshold value of {low_threshold} is outside
            of the [0-1] CellProfiler default."""
        )

    if mask is None:
        mask = numpy.ones(image.shape, bool)

    if method == EdgeFindingMethod.SOBEL:
        output_pixels = enhance_edges_sobel(image, mask, direction)
    elif method == EdgeFindingMethod.LOG:
        output_pixels = enhance_edges_log(image, mask, sigma)
    elif method == EdgeFindingMethod.PREWITT:
        output_pixels = enhance_edges_prewitt(image, mask, direction)
    elif method == EdgeFindingMethod.CANNY:
        output_pixels = enhance_edges_canny(
            image,
            mask,
            auto_threshold=automatic_threshold,
            auto_low_threshold=automatic_low_threshold,
            sigma=sigma,
            low_threshold=low_threshold,
            manual_threshold=manual_threshold,
            threshold_adjustment_factor=threshold_adjustment_factor,
        )
    elif method == EdgeFindingMethod.ROBERTS:
        output_pixels = centrosome.filter.roberts(image, mask)
    elif method == EdgeFindingMethod.KIRSCH:
        output_pixels = centrosome.kirsch.kirsch(image)
    else:
        raise NotImplementedError(f"{method} edge detection method is not implemented.")

    return output_pixels

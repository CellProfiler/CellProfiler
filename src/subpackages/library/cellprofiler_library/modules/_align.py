import numpy as np

from typing import Tuple, Optional, List, Union, Annotated
from pydantic import Field, validate_call, ConfigDict

from cellprofiler_library.opts.align import CropMode, AlignmentMethod
from cellprofiler_library.types import Image2D, Image2DMask, ImageBinary
from cellprofiler_library.functions.image_processing import (
    align_cross_correlation,
    align_mutual_information,
    offset_slice,
)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def align_images(
        image1_pixels: Image2D, 
        image2_pixels: Image2D, 
        image1_mask: Image2DMask, 
        image2_mask: Image2DMask, 
        alignment_method: AlignmentMethod
    ) -> Tuple[
        int,
        int
    ]:
    """Align the second image with the first
    Calculate the alignment offset that must be added to indexes in the
    first image to arrive at indexes in the second image.

    Returns the x,y (not i,j) offsets.
    """
    if alignment_method == AlignmentMethod.CROSS_CORRELATION.value:
        return align_cross_correlation(image1_pixels, image2_pixels)
    else:
        return align_mutual_information(
            image1_pixels, image2_pixels, image1_mask, image2_mask
        )
    

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def adjust_offsets(
        offsets: Annotated[List[Tuple[int, int]], Field(description="Offsets to be adjusted")],
        shapes: Annotated[List[Tuple[int, int]], Field(description="Shapes of images")],
        crop_mode: Annotated[CropMode, Field(description="The crop mode determines how the output images are either cropped or padded after alignment")]
    ) -> Tuple[
        List[Tuple[int, int]],
        List[Tuple[int, int]]
    ]:
    """Adjust the offsets and shapes for output

    workspace - workspace passed to "run"

    offsets - i,j offsets for each image

    shapes - shapes of the input images

    names - pairs of input / output names

    Based on the crop mode, adjust the offsets and shapes to optimize
    the cropping.
    """
    offsets = np.array(offsets)
    shapes = np.array(shapes)
    if crop_mode == CropMode.CROP.value:
        # modify the offsets so that all are negative
        max_offset = np.max(offsets, 0)
        offsets = offsets - max_offset[np.newaxis, :]
        #
        # Reduce each shape by the amount chopped off
        #
        shapes += offsets
        #
        # Pick the smallest in each of the dimensions and repeat for all
        #
        shape = np.min(shapes, 0)
        shapes = np.tile(shape, len(shapes))
        shapes.shape = offsets.shape
    elif crop_mode == CropMode.PAD.value:
        #
        # modify the offsets so that they are all positive
        #
        min_offset = np.min(offsets, 0)
        offsets = offsets - min_offset[np.newaxis, :]
        #
        # Expand each shape by the top-left padding
        #
        shapes += offsets
        #
        # Pick the largest in each of the dimensions and repeat for all
        #
        shape = np.max(shapes, 0)
        shapes = np.tile(shape, len(shapes))
        shapes.shape = offsets.shape
    return offsets.tolist(), shapes.tolist()


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def apply_alignment(
        pixel_data: Annotated[Image2D, Field(description="Pixel data to be aligned")],
        image_mask: Annotated[Image2DMask, Field(description="Mask of the image to be aligned")],
        off_x: Annotated[int, Field(description="Offset of the resultant image relative to the original")],
        off_y: Annotated[int, Field(description="Offset of the resultant image relative to the original")],
        shape: Annotated[Tuple[int, int], Field(description="Shape of the resultant image")],
    ) -> Tuple[
        Image2D, 
        Image2DMask, 
        Optional[Union[Image2DMask, ImageBinary]],
    ]:
    if pixel_data.ndim == 2:
        output_shape = (shape[0], shape[1], 1)
        planes = [pixel_data]
    else:
        output_shape = (shape[0], shape[1], pixel_data.shape[2])
        planes = [pixel_data[:, :, i] for i in range(pixel_data.shape[2])]
    output_pixels = np.zeros(output_shape, pixel_data.dtype)
    for i, plane in enumerate(planes):
        #
        # Copy the input to the output
        #
        p1, p2 = offset_slice(plane, output_pixels[:, :, i], off_y, off_x)
        p2[:, :] = p1[:, :]
    if pixel_data.ndim == 2:
        output_pixels.shape = output_pixels.shape[:2]
    output_mask = np.zeros(shape, bool)
    p1, p2 = offset_slice(image_mask, output_mask, off_y, off_x)
    p2[:, :] = p1[:, :]
    if np.all(output_mask):
        output_mask = None
    crop_mask = np.zeros(pixel_data.shape, bool)
    p1, p2 = offset_slice(crop_mask, output_pixels, off_y, off_x)
    p1[:, :] = True
    if np.all(crop_mask):
        crop_mask = None
    return output_pixels, output_mask, crop_mask

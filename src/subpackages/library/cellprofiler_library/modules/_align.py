import numpy as np

from typing import Tuple, Optional, List, Union, Annotated
from pydantic import Field, validate_call, ConfigDict, BaseModel

from cellprofiler_library.opts.align import CropMode, AlignmentMethod, AdditionalAlignmentChoice, MEASUREMENT_FORMAT
from cellprofiler_library.types import Image2D, Image2DMask, ImageBinary
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.functions.image_processing import (
    align_cross_correlation,
    align_mutual_information,
    offset_slice,
)

ImageInfo = List[Tuple[
    str, # input image name
    Image2D, # input image pixels
    str, # output image name
    Image2D, # output image pixels
    int, # offset X
    int, # offset Y
    Tuple[int, int], # image shape
]]

class AlignDisplayData(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        populate_by_name=True
    )

    image_info: ImageInfo

AlignReturnData = Tuple[
    List[Image2D],
    List[Image2DMask],
    List[Optional[Union[Image2DMask, ImageBinary]]],
    LibraryMeasurements
]

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def align_images(
        primary_image: Annotated[Image2D, Field(description="Primary image")],
        primary_image_mask: Annotated[Image2DMask, Field(description="Primary image mask")],
        secondary_image: Annotated[Image2D, Field(description="Secondary image")],
        secondary_image_mask: Annotated[Image2DMask, Field(description="Secondary image mask")],
        alignment_method: Annotated[AlignmentMethod, Field(description="Alignment method")],
        crop_mode: Annotated[CropMode, Field(description="Crop mode")],
        additional_images: Annotated[List[Image2D], Field(description="List of additonal images to align")] = [],
        additional_image_masks: Annotated[List[Image2DMask], Field(description="List of image masks for additonal images")] = [],
        additional_image_alignments: Annotated[List[AdditionalAlignmentChoice], Field(description="List of alignment types for additonal images")] = [],
        input_image_names: Annotated[Optional[List[str]], Field(description="List of input image names for visualization (if None name will be generated)")] = None,
        output_image_names: Annotated[Optional[List[str]], Field(description="List of output image names for visualization (if None name will be generated)")] = None,
        return_visualization_data: Annotated[bool, Field(description="Return GT_pixels and ID_pixels for visualization")] = False,
) -> Union[
        Tuple[AlignReturnData, AlignDisplayData],
        AlignReturnData
    ]:
        assert len(additional_images) == len(additional_image_masks), "Must have same number of image masks as images"
        assert len(additional_images) == len(additional_image_alignments), "Must have same number of image alignments as images"
        assert input_image_names is None or len(input_image_names) == (len(additional_images) + 2), "Must have same number of input image names as images"
        assert output_image_names is None or len(output_image_names) == (len(additional_images) + 2), "Must have same number of output image names as images"

        off_x, off_y = align_pair(
            primary_image,
            secondary_image,
            primary_image_mask,
            secondary_image_mask,
            alignment_method
        )
        offsets = [(0,0), (off_y, off_x)]

        for i in range(len(additional_images)):
            if additional_image_alignments[i] == AdditionalAlignmentChoice.SIMILARLY.value:
                offsets.append((off_y, off_x))
            elif additional_image_alignments[i] == AdditionalAlignmentChoice.SEPARATELY.value:
                a_off_x, a_off_y = align_pair(
                    primary_image,
                    additional_images[i],
                    primary_image_mask,
                    additional_image_masks[i],
                    alignment_method
                )
                offsets.append((a_off_y, a_off_x))

        shapes = [primary_image.shape[:2], secondary_image.shape[:2]] + [img.shape[:2] for img in additional_images]
        offsets, shapes = adjust_offsets(offsets, shapes, crop_mode)

        output_images: List[Image2D] = []
        output_image_masks: List[Image2DMask] = []
        crop_masks: List[Optional[Union[Image2DMask, ImageBinary]]] = []

        measurements = LibraryMeasurements()
        image_info: ImageInfo = []

        for i in range(len(offsets)):
            if i  == 0:
                input_image = primary_image
                input_image_mask = primary_image_mask
                if return_visualization_data:
                    input_image_name = input_image_names[i] if input_image_names else "Primary Image"
                    output_image_name = output_image_names[i] if output_image_names else "Primary Image Aligned"
            elif i  == 1:
                input_image = secondary_image
                input_image_mask = secondary_image_mask
                if return_visualization_data:
                    input_image_name = input_image_names[i] if input_image_names else "Secondary Image"
                    output_image_name = output_image_names[i] if output_image_names else "Secondary Image Aligned"
            else:
                input_image = additional_images[i-2]
                input_image_mask = additional_image_masks[i-2]
                if return_visualization_data:
                    input_image_name = input_image_names[i] if input_image_names else f"Additional Image {i+1}"
                    output_image_name = output_image_names[i] if output_image_names else f"Additional Image {i+1} Aligned"

            output_image, output_mask, crop_mask = apply_alignment(
                input_image,
                input_image_mask,
                offsets[i][1],
                offsets[i][0],
                shapes[i],
            )

            output_images.append(output_image)
            output_image_masks.append(output_mask)
            crop_masks.append(crop_mask)
            output_image_name = output_image_names[i] if output_image_names is not None else f"Image{i+1}"

            for axis, value in (("X", -offsets[i][1]), ("Y", -offsets[i][0])):
                measurements.add_image_measurement(MEASUREMENT_FORMAT % (axis, output_image_name), value)

            if return_visualization_data:
                image_info.append((
                    input_image_name,
                    input_image,
                    output_image_name,
                    output_image,
                    offsets[i][1],
                    offsets[i][0],
                    shapes[i]
                ))

        if return_visualization_data:
            return (output_images, output_image_masks, crop_masks, measurements), AlignDisplayData(image_info=image_info)
        return output_images, output_image_masks, crop_masks, measurements

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def align_pair(
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
    else: # alignment_method == AlignmentMethod.MUTUAL_INFORMATION.value:
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

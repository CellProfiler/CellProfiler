import numpy
from numpy.typing import NDArray
from typing import List, Tuple, Optional
from cellprofiler_library.opts.graytocolor import Scheme
from cellprofiler_library.types import Image2DGrayscale, Image2DColor

def gray_to_rgb(
        image_arr: List[Optional[Image2DGrayscale]],
        adjustment_factor_array: List[float],
        intensities: List[Tuple[float, ...]]=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        wants_rescale: bool=True,
        ):
    assert(len(image_arr) == len(adjustment_factor_array))
    assert(len(image_arr) == len(intensities))

    parent_image = None
    rgb_pixel_data = None
    for pixel_data, adjustment_factor, intensity_triplet in zip(image_arr, adjustment_factor_array, intensities):
        if pixel_data is None:
            continue
        multiplier = numpy.array(intensity_triplet) * adjustment_factor
        if wants_rescale:
            pixel_data = pixel_data /numpy.max(pixel_data)
        if parent_image is not None:
            if parent_image.shape != pixel_data.shape:
                raise ValueError(
                    "The input images have different sizes (%s vs %s)"
                    % (
                        parent_image.shape, 
                        pixel_data.shape,
                    )
                )
            rgb_pixel_data += numpy.dstack([pixel_data] * 3) * multiplier
        else:
            parent_image = pixel_data
            rgb_pixel_data = numpy.dstack([pixel_data] * 3) * multiplier
    return rgb_pixel_data


def gray_to_composite_color(
        pixel_data_arr: List[Optional[Image2DGrayscale]],
        scheme_choice: Scheme,
        wants_rescale: bool,
        color_array: List[Tuple[int, int, int]],
        weight_array: List[float],
):
    source_channels = pixel_data_arr
    parent_image = pixel_data_arr[0]
    for idx, pd in enumerate(source_channels):
        if pd is None:
            continue
        if pd.shape != source_channels[0].shape:
            raise ValueError(
                "The input images have different sizes (%s vs %s)"
                % (
                    source_channels[0].shape,
                    pd.shape,
                )
            )
    if scheme_choice == Scheme.STACK:
        rgb_pixel_data = numpy.dstack(source_channels)
    else:
        colors: List[NDArray[numpy.float32]] = []
        pixel_data = parent_image
        if wants_rescale:
            pixel_data = pixel_data / numpy.max(pixel_data)
        for color_tuple, weight in zip(color_array, weight_array):
            color = (weight * numpy.array(color_tuple).astype(pixel_data.dtype) / 255)
            colors += [color[numpy.newaxis, numpy.newaxis, :]]
        rgb_pixel_data = pixel_data[:, :, numpy.newaxis] * colors[0]
        for image, color in zip(source_channels[1:], colors[1:]):
            if wants_rescale:
                image = image / numpy.max(image)
            rgb_pixel_data = rgb_pixel_data + image[:, :, numpy.newaxis] * color

    # if scheme_choice != Scheme.STACK and wants_rescale:
    #     # If we rescaled, clip values that went out of range after multiplication
    #     rgb_pixel_data[rgb_pixel_data > 1] = 1
    return rgb_pixel_data
    

def gray_to_stacked_color(
    image_arr: List[Image2DGrayscale]
):
    source_channels = image_arr
    rgb_pixel_data = numpy.dstack(source_channels)
    return rgb_pixel_data


class ColorSchemeSettings(object):
    """Collect all of the details for one color in one place"""

    def __init__(
        self,
        image_name_setting: str,
        adjustment_setting: float,
        red_intensity: float,
        green_intensity: float,
        blue_intensity: float,
    ):
        """Initialize with settings and multipliers

        image_name_setting - names the image to use for the color
        adjustment_setting - weights the image
        red_intensity - indicates how much it contributes to the red channel
        green_intensity - indicates how much it contributes to the green channel
        blue_intensity - indicates how much it contributes to the blue channel
        """
        self.image_name = image_name_setting
        self.adjustment_factor = adjustment_setting
        self.red_intensity = red_intensity
        self.green_intensity = green_intensity
        self.blue_intensity = blue_intensity

    @property
    def intensities(self) -> NDArray[numpy.float32]:
        """The intensities in RGB order as a numpy array"""
        return numpy.array(
            (self.red_intensity, self.green_intensity, self.blue_intensity)
        )

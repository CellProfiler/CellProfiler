import numpy
from typing import Annotated, Optional, Tuple, Callable, Dict, Union
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.opts.convertobjectstoimage import ImageMode
from cellprofiler_library.types import ImageBinary, ImageColor, ImageGrayscale, ObjectLabelSet, ImageAny, ImageInt
from cellprofiler_library.functions.object_processing import image_mode_black_and_white, image_mode_grayscale, image_mode_color, image_mode_uint16

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def convert_objects_to_image(
        image_mode: Annotated[ImageMode, Field(description="Color format to be used for conversion")],
        objects_labels : Annotated[ObjectLabelSet, Field(description="Labels of the objects")],
        objects_shape : Annotated[Tuple[int, ...], Field(description="Shape of the objects")],
        colormap_value : Annotated[Optional[str], Field(description="Colormap to be used for conversion")] = None
        ) -> ImageAny:

    alpha = numpy.zeros(objects_shape, numpy.int32)

    converter_fn_map = {
        ImageMode.BINARY: image_mode_black_and_white,
        ImageMode.GRAYSCALE: image_mode_grayscale,
        ImageMode.COLOR: image_mode_color,
        ImageMode.UINT16: image_mode_uint16,
    }

    pixel_data_init_map: Dict[
        ImageMode,
        Callable[[], Union[ImageGrayscale, ImageBinary, ImageColor, ImageInt]]
    ] = {
        ImageMode.BINARY: lambda: numpy.zeros(objects_shape, bool),
        ImageMode.GRAYSCALE: lambda: numpy.zeros(objects_shape),
        ImageMode.COLOR: lambda: numpy.zeros(objects_shape + (3,)),
        ImageMode.UINT16: lambda: numpy.zeros(objects_shape, numpy.int32),
    }
    pixel_data = pixel_data_init_map.get(image_mode, lambda: numpy.zeros(objects_shape + (3,)))()
    for labels, _ in objects_labels:
        mask = labels != 0
        if numpy.all(~mask):
            continue
        pixel_data, alpha = converter_fn_map[image_mode](pixel_data, mask, alpha, labels, colormap_value)
    mask = alpha > 0
    if image_mode == ImageMode.COLOR:
        pixel_data[mask, :] = pixel_data[mask, :] / alpha[mask][:, numpy.newaxis]
    elif image_mode != ImageMode.BINARY:
        pixel_data[mask] = pixel_data[mask] / alpha[mask]
    return pixel_data

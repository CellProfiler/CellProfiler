import numpy
import matplotlib.cm
import centrosome.cpmorphology
from typing import Annotated, Any, Optional, Tuple, Callable
from pydantic import Field, validate_call, BeforeValidator, ConfigDict
from cellprofiler_library.opts.convertobjectstoimage import ImageMode
from ..types import ObjectLabelSet
from ..functions.object_processing import image_mode_black_and_white, image_mode_grayscale, image_mode_color, image_mode_uint16

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def convert_objects_to_image(
        image_mode: Annotated[ImageMode, Field(description="Color format to be used for conversion")],
        objects_labels : Annotated[ObjectLabelSet, Field(description="Labels of the objects")],
        objects_shape : Annotated[tuple, Field(description="Shape of the objects")],
        colormap_value : Annotated[Optional[str], Field(description="Colormap to be used for conversion")] = None
        ):
    
    alpha = numpy.zeros(objects_shape)

    converter_fn_map = {
        ImageMode.BINARY: image_mode_black_and_white,
        ImageMode.GRAYSCALE: image_mode_grayscale,
        ImageMode.COLOR: image_mode_color,
        ImageMode.UINT16: image_mode_uint16,
    }

    pixel_data_init_map = {
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


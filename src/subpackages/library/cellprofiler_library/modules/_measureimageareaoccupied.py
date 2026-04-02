import skimage.measure
import numpy as np
from typing import Annotated, Optional, Tuple, List
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.types import ImageBinary, ObjectSegmentation, ImageAny
from cellprofiler_library.functions.measurement import measure_area_occupied, measure_total_area, measure_perimeter, measure_object_perimeter, measure_objects_area_occupied, measure_objects_total_area
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.opts.measureimageareaoccupied import TemplateMeasurementFormat


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_area_perimeter(
    im_pixel_data:  Annotated[ImageBinary, Field(description="Binary image pixel data")],
    image_name:     Annotated[str, Field(description="Name of the image")],
    im_volumetric:  Annotated[bool, Field(description="Image is volumetric")],
    im_spacing:     Annotated[Optional[Tuple[float, ...]], Field(description="Image spacing")] = None,
    pipeline_volumetric: Annotated[bool, Field(description="Pipeline is volumetric")] = False
    ) -> Tuple[
        LibraryMeasurements,
        List[List[str]]
    ]:

    area_occupied = measure_area_occupied(im_pixel_data)
    perimeter = measure_perimeter(im_pixel_data, im_volumetric, im_spacing)  if area_occupied > 0 else np.float64(0.0)
    total_area = measure_total_area(im_pixel_data)
 
    area_format = TemplateMeasurementFormat.VOLUME_OCCUPIED_FORMAT if pipeline_volumetric else TemplateMeasurementFormat.AREA_OCCUPIED_FORMAT
    perimeter_format = TemplateMeasurementFormat.SURFACE_AREA_FORMAT if pipeline_volumetric else TemplateMeasurementFormat.PERIMETER_FORMAT
    total_area_format = TemplateMeasurementFormat.TOTAL_VOLUME_FORMAT if pipeline_volumetric else TemplateMeasurementFormat.TOTAL_AREA_FORMAT

    # Format: AreaOccupied_Feature_ImageName
    measurements = LibraryMeasurements()
    measurements.add_image_measurement(area_format % image_name, area_occupied)
    measurements.add_image_measurement(perimeter_format % image_name, perimeter)
    measurements.add_image_measurement(total_area_format % image_name, total_area)

    return measurements, [[image_name, str(area_occupied), str(perimeter), str(total_area),]]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_objects_area_perimeter(
    label_image:    Annotated[ObjectSegmentation, Field(description="Object labels to measure")],
    object_name:    Annotated[str, Field(description="Name of the objects")],
    mask:           Annotated[Optional[ImageAny], Field(description="Mask of the image")] = None,
    volumetric:     Annotated[bool, Field(description="True if the objects are volumetric")] = False,
    spacing:        Annotated[Optional[Tuple[float, ...]], Field(description="Image spacing")] = None,
    pipeline_volumetric: Annotated[bool, Field(description="Pipeline is volumetric")] = False
    ) -> Tuple[
        LibraryMeasurements,
        List[List[str]]
    ]:
    if mask is not None:
        label_image[~mask] = 0
    regionprops = skimage.measure.regionprops(label_image)

    total_area = measure_objects_total_area(label_image, mask)
    area_occupied = measure_objects_area_occupied(None, regionprops=regionprops)
    perimeter = measure_object_perimeter(label_image, regionprops=regionprops, volumetric=volumetric, spacing=spacing) if area_occupied > 0 else np.float64(0.0)

    area_format = TemplateMeasurementFormat.VOLUME_OCCUPIED_FORMAT if pipeline_volumetric else TemplateMeasurementFormat.AREA_OCCUPIED_FORMAT
    perimeter_format = TemplateMeasurementFormat.SURFACE_AREA_FORMAT if pipeline_volumetric else TemplateMeasurementFormat.PERIMETER_FORMAT
    total_area_format = TemplateMeasurementFormat.TOTAL_VOLUME_FORMAT if pipeline_volumetric else TemplateMeasurementFormat.TOTAL_AREA_FORMAT

    # Format: AreaOccupied_Feature_ObjectName
    measurements = LibraryMeasurements()
    measurements.add_image_measurement(area_format % object_name, area_occupied)
    measurements.add_image_measurement(perimeter_format % object_name, perimeter)
    measurements.add_image_measurement(total_area_format % object_name,total_area)

    return measurements, [[object_name, str(area_occupied), str(perimeter), str(total_area),]]

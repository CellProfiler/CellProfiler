import skimage.measure
import numpy as np
from typing import Annotated, Optional, Tuple
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.types import ImageBinary, ObjectSegmentation, ImageAny
from cellprofiler_library.functions.measurement import measure_area_occupied, measure_total_area, measure_perimeter, measure_object_perimeter, measure_objects_area_occupied, measure_objects_total_area
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.opts.measureimageareaoccupied import MeasurementType, C_AREA_OCCUPIED


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_image_area_perimeter(
    im_pixel_data:  Annotated[ImageBinary, Field(description="Binary image pixel data")],
    image_name:     Annotated[str, Field(description="Name of the image")],
    im_volumetric:  Annotated[bool, Field(description="Image is volumetric")],
    im_spacing:     Annotated[Optional[Tuple[float, ...]], Field(description="Image spacing")] = None
    ) -> LibraryMeasurements:

    area_occupied = measure_area_occupied(im_pixel_data)
    perimeter = measure_perimeter(im_pixel_data, im_volumetric, im_spacing)  if area_occupied > 0 else np.float64(0.0)
    total_area = measure_total_area(im_pixel_data)
 
    measurements = LibraryMeasurements()

    feature_area_occupied = MeasurementType.VOLUME_OCCUPIED.value if im_volumetric else MeasurementType.AREA_OCCUPIED.value
    feature_perimeter = MeasurementType.SURFACE_AREA.value if im_volumetric else MeasurementType.PERIMETER.value
    feature_total_area = MeasurementType.TOTAL_VOLUME.value if im_volumetric else MeasurementType.TOTAL_AREA.value

    # Format: AreaOccupied_Feature_ImageName
    measurements.add_image_measurement(
        f"{C_AREA_OCCUPIED}_{feature_area_occupied}_{image_name}",
        area_occupied
    )
    measurements.add_image_measurement(
        f"{C_AREA_OCCUPIED}_{feature_perimeter}_{image_name}",
        perimeter
    )
    measurements.add_image_measurement(
        f"{C_AREA_OCCUPIED}_{feature_total_area}_{image_name}",
        total_area
    )

    return measurements


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_objects_area_perimeter(
    label_image:    Annotated[ObjectSegmentation, Field(description="Object labels to measure")],
    object_name:    Annotated[str, Field(description="Name of the objects")],
    mask:           Annotated[Optional[ImageAny], Field(description="Mask of the image")] = None,
    volumetric:     Annotated[bool, Field(description="True if the image is volumetric")] = False,
    spacing:        Annotated[Optional[Tuple[float, ...]], Field(description="Image spacing")] = None
    ) -> LibraryMeasurements:
    if mask is not None:
        label_image[~mask] = 0
    regionprops = skimage.measure.regionprops(label_image)

    total_area = measure_objects_total_area(label_image, mask)
    area_occupied = measure_objects_area_occupied(None, regionprops=regionprops)
    perimeter = measure_object_perimeter(label_image, regionprops=regionprops, volumetric=volumetric, spacing=spacing) if area_occupied > 0 else np.float64(0.0)

    measurements = LibraryMeasurements()

    feature_area_occupied = MeasurementType.VOLUME_OCCUPIED.value if volumetric else MeasurementType.AREA_OCCUPIED.value
    feature_perimeter = MeasurementType.SURFACE_AREA.value if volumetric else MeasurementType.PERIMETER.value
    feature_total_area = MeasurementType.TOTAL_VOLUME.value if volumetric else MeasurementType.TOTAL_AREA.value

    # Format: AreaOccupied_Feature_ObjectName
    measurements.add_image_measurement(
        f"{C_AREA_OCCUPIED}_{feature_area_occupied}_{object_name}",
        area_occupied
    )
    measurements.add_image_measurement(
        f"{C_AREA_OCCUPIED}_{feature_perimeter}_{object_name}",
        perimeter
    )
    measurements.add_image_measurement(
        f"{C_AREA_OCCUPIED}_{feature_total_area}_{object_name}",
        total_area
    )

    return measurements

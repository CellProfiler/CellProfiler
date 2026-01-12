import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Annotated, Optional, Dict, Union, Any
from pydantic import validate_call, ConfigDict, BeforeValidator, Field
from cellprofiler_library.functions.measurement import measure_correlation_and_slope_from_objects, measure_manders_coefficient_from_objects, measure_rwc_coefficient_from_objects, measure_overlap_coefficient_from_objects, measure_costes_coefficient_from_objects, get_thresholded_images_and_counts, measure_correlation_and_slope, measure_manders_coefficient, measure_rwc_coefficient, measure_overlap_coefficient, measure_costes_coefficient
from cellprofiler_library.opts.measurecolocalization import TemplateMeasurementFormat, MeasurementType
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask, Pixel, ObjectLabel, ObjectSegmentation, ImageAny, ImageAnyMask
from cellprofiler_library.opts.measurecolocalization import CostesMethod
from cellprofiler_library.functions.image_processing import crop_image_similarly
from cellprofiler_library.functions.object_processing import size_similarly, object_crop_image_similarly

def crop_image_pair_similarly(
        im1_pixel_data:     Annotated[ImageGrayscale, Field(description="First image pixel data")],
        im2_pixel_data:     Annotated[ImageGrayscale, Field(description="Second image pixel data")],
        im1_mask:           Annotated[ImageGrayscaleMask, Field(description="First image mask")],
        im2_mask:           Annotated[ImageGrayscaleMask, Field(description="Second image mask")],
        im1_crop_mask:      Annotated[Optional[ImageGrayscaleMask], Field(description="Mask used to obtain the first image")]=None,
        im2_crop_mask:      Annotated[Optional[ImageGrayscaleMask], Field(description="Mask used to obtain the second image")]=None,
    ) -> Tuple[ImageGrayscale, ImageGrayscale, ImageGrayscaleMask]:
    
    im1_pixel_count = np.prod(im1_pixel_data.shape)
    im2_pixel_count = np.prod(im2_pixel_data.shape)

    if im1_pixel_count < im2_pixel_count:
        # im2_pixel_data = im1.crop_image_similarly(im2.pixel_data)
        im2_pixel_data = crop_image_similarly(im1_pixel_data, im2_pixel_data, im1_crop_mask)
        im2_mask = crop_image_similarly(im1_pixel_data, im2_mask, im1_crop_mask)
    elif im2_pixel_count < im1_pixel_count:
        im1_pixel_data = crop_image_similarly(im2_pixel_data, im1_pixel_data, im2_crop_mask)
        im1_mask = crop_image_similarly(im2_pixel_data, im1_mask, im2_crop_mask)
    mask = (
        im1_mask
        & im2_mask
        & (~np.isnan(im1_pixel_data))
        & (~np.isnan(im2_pixel_data))
    )
    return im1_pixel_data, im2_pixel_data, mask

def crop_image_pair_and_object_similarly(
        im1_pixel_data:     Annotated[ImageGrayscale, Field(description="First image pixel data")],
        im2_pixel_data:     Annotated[ImageGrayscale, Field(description="Second image pixel data")],
        obj_segmented:      Annotated[ObjectSegmentation, Field(description="Object segmentations")],
        im1_mask:           Annotated[ImageGrayscaleMask, Field(description="First image mask")],
        im2_mask:           Annotated[ImageGrayscaleMask, Field(description="Second image mask")],
        obj_parent_image:   Annotated[Optional[ImageAny], Field(description="Object parent image")],
        obj_parent_crop_mask: Annotated[Optional[ImageAnyMask], Field(description="Object parent image crop mask")]=None,
        im1_crop_mask:      Annotated[Optional[ImageGrayscaleMask], Field(description="Mask used to obtain the first image")]=None,
        im2_crop_mask:      Annotated[Optional[ImageGrayscaleMask], Field(description="Mask used to obtain the second image")]=None,
) -> Tuple[
    NDArray[Pixel],
    NDArray[Pixel],
    ObjectSegmentation,
    ImageGrayscaleMask,
    NDArray[Pixel],
    NDArray[Pixel]
]:
    try:
        im1_pixels = object_crop_image_similarly(obj_segmented, im1_pixel_data, obj_parent_image, obj_parent_crop_mask) # first check image shape and segmented shape, if new, crop image similarly wiht parent_image. To do that, verify parent image shape and the image shape, if neq, use parent image crop mask as well as the parent image shape 
        _im1_mask = object_crop_image_similarly(obj_segmented, im1_mask, obj_parent_image, obj_parent_crop_mask)
    except ValueError:
        im1_pixels, m1 = size_similarly(obj_segmented, im1_pixel_data)
        _im1_mask, m1 = size_similarly(obj_segmented, im1_mask)
        _im1_mask[~m1] = False
    try:
        im2_pixels = object_crop_image_similarly(obj_segmented, im2_pixel_data, obj_parent_image, obj_parent_crop_mask)
        _im2_mask = object_crop_image_similarly(obj_segmented, im2_mask, obj_parent_image, obj_parent_crop_mask)
    except ValueError:
        im2_pixels, m1 = size_similarly(obj_segmented, im2_pixel_data)
        _im2_mask, m1 = size_similarly(obj_segmented, im2_mask)
        _im2_mask[~m1] = False
    mask = (obj_segmented > 0) & _im1_mask & _im2_mask
    im1_pixels = im1_pixels[mask]
    im2_pixels = im2_pixels[mask]
    obj_segmented = obj_segmented[mask]
    #
    # Code below is used for the Costes' automated thresholding
    #
    # im1_mask = im1_mask
    im1_pixel_count = np.product(im1_pixel_data.shape)
    im2_pixel_data = im2_pixel_data
    # im2_mask = im2_mask
    im2_pixel_count = np.product(im2_pixel_data.shape)
    #
    # Crop the larger image similarly to the smaller one
    #
    if im1_pixel_count < im2_pixel_count:
        im2_pixel_data = crop_image_similarly(im1_pixel_data, im2_pixel_data, im1_crop_mask)
        im2_mask = crop_image_similarly(im1_pixel_data, im2_mask, im1_crop_mask)
    elif im2_pixel_count < im1_pixel_count:
        im1_pixel_data = crop_image_similarly(im2_pixel_data, im1_pixel_data, im2_crop_mask)
        im1_mask = crop_image_similarly(im2_pixel_data, im1_mask, im2_crop_mask)
    mask = (
        im1_mask
        & im2_mask
        & (~np.isnan(im1_pixel_data))
        & (~np.isnan(im2_pixel_data))
    )
    #
    # fi and si are used to obtain the costes threshold values for their respective images
    #
    im1_costes_pixels = None
    im2_costes_pixels = None
    if mask is not None and np.any(mask):
        im1_costes_pixels = im1_pixel_data[mask]
        im2_costes_pixels = im2_pixel_data[mask]

    return im1_pixels, im2_pixels, obj_segmented, mask, im1_costes_pixels, im2_costes_pixels

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def run_image_pair_images(
    im1_pixel_data:     Annotated[ImageGrayscale, Field(description="First image pixel data")],
    im2_pixel_data:     Annotated[ImageGrayscale, Field(description="Second image pixel data")],
    im1_name:           Annotated[str, Field(description="First image name")] = "Name of the first image. This will be used to name the measurement columns",
    im2_name:           Annotated[str, Field(description="Second image name")] = "Name of the second image. This will be used to name the measurement columns",
    mask:               Annotated[Optional[ImageGrayscaleMask], Field(description="Mask of the pixels to be used for the measurements")] = None, 
    im1_thr_percentage: Annotated[float, Field(description="Threshold value for the first image"), BeforeValidator(np.float64)]=100, 
    im2_thr_percentage: Annotated[float, Field(description="Threshold value for the second image"), BeforeValidator(np.float64)]=100, 
    measurement_types:  Annotated[List[MeasurementType], Field(description="List of measurement types to be calculated")] = [MeasurementType.CORRELATION, MeasurementType.MANDERS, MeasurementType.RWC, MeasurementType.OVERLAP, MeasurementType.COSTES],
    im1_scale:          Annotated[Optional[float], Field(description="")] = None,
    im2_scale:          Annotated[Optional[float], Field(description="")] = None,
    costes_method:      Annotated[Optional[CostesMethod], Field(description="")] = CostesMethod.FAST,
    ) -> Annotated[
        Tuple[
            Dict[str, Dict[str, Union[float, int, str, np.float64]]],
            List[Tuple[str, str, str, str, str]]
        ], Field(description="List of measurement results and a dictionary of measurements with precise values")]:
    """Calculate the correlation between the pixels of two images"""


    summary: List[Tuple[str, str, str, str, str]] = []
    measurements: Dict[str, Union[float, int, str, np.float64]] = {}
    corr =      np.float64(np.NaN)
    slope =     np.float64(np.NaN)
    C1 =        np.float64(np.NaN)
    C2 =        np.float64(np.NaN)
    M1 =        np.float64(np.NaN)
    M2 =        np.float64(np.NaN)
    RWC1 =      np.float64(np.NaN)
    RWC2 =      np.float64(np.NaN)
    overlap =   np.float64(np.NaN)
    K1 =        np.float64(np.NaN)
    K2 =        np.float64(np.NaN)
    if mask is not None and np.any(mask):
        im1_pixels = im1_pixel_data[mask]
        im2_pixels = im2_pixel_data[mask]

        if MeasurementType.CORRELATION in measurement_types:
            corr, slope = measure_correlation_and_slope(im1_pixels, im2_pixels)
            summary += [
                (im1_name, im2_name,"-","Correlation","%.3f" % corr),
                (im1_name, im2_name, "-", "Slope", "%.3f" % slope),
            ]

        if set(measurement_types).intersection({MeasurementType.MANDERS, MeasurementType.RWC, MeasurementType.OVERLAP, MeasurementType.COSTES}) != set():
            # Threshold as percentage of maximum intensity in each channel
            im1_thr_sum, im2_thr_sum, thr_mask_intersection = get_thresholded_images_and_counts(im1_pixels, im2_pixels, im1_thr_percentage, im2_thr_percentage)
            # take the first element as only a single value is expected for whole image threshold sum
            im1_thr_sum = im1_thr_sum[0]
            im2_thr_sum = im2_thr_sum[0]

            if MeasurementType.MANDERS in measurement_types:
                M1, M2 = measure_manders_coefficient(im1_pixels, im2_pixels, im1_thr_sum, im2_thr_sum, thr_mask_intersection)
                summary += [
                    (im1_name, im2_name, "-", "Manders Coefficient", "%.3f" % M1),
                    (im2_name, im1_name, "-", "Manders Coefficient", "%.3f" % M2),
                ]


            if MeasurementType.RWC in measurement_types:
                RWC1, RWC2 = measure_rwc_coefficient(im1_pixels, im2_pixels, im1_thr_sum, im2_thr_sum, thr_mask_intersection)
                summary += [
                    (im1_name, im2_name, "-", "RWC Coefficient", "%.3f" % RWC1),
                    (im2_name, im1_name, "-", "RWC Coefficient", "%.3f" % RWC2),
                ]

            if MeasurementType.OVERLAP in measurement_types:
                overlap, K1, K2 = measure_overlap_coefficient(im1_pixels, im2_pixels, thr_mask_intersection)
                summary += [
                    (im1_name, im2_name, "-", "Overlap Coefficient", "%.3f" % overlap),
                ]

            if MeasurementType.COSTES in measurement_types:
                assert costes_method is not None, "costes_method must be one of {}".format([e.value for e in CostesMethod])
                assert costes_method in [e.value for e in CostesMethod], "costes_method must be one of {}".format([e.value for e in CostesMethod])
                
                C1, C2 = measure_costes_coefficient(im1_pixels, im2_pixels, im1_scale, im2_scale, costes_method=costes_method)
                
                summary += [
                    (im1_name, im2_name, "-", "Manders Coefficient (Costes)", "%.3f" % C1),
                    (im2_name, im1_name, "-", "Manders Coefficient (Costes)", "%.3f" % C2),
                ]

    #
    # Add the measurements
    #
    corr_measurement = TemplateMeasurementFormat.CORRELATION_FORMAT % (im1_name, im2_name)
    slope_measurement = TemplateMeasurementFormat.SLOPE_FORMAT % (im1_name, im2_name)
    overlap_measurement = TemplateMeasurementFormat.OVERLAP_FORMAT % (im1_name, im2_name)

    k_measurement_1 = TemplateMeasurementFormat.K_FORMAT % (im1_name, im2_name)
    k_measurement_2 = TemplateMeasurementFormat.K_FORMAT % (im2_name, im1_name)
    
    manders_measurement_1 = TemplateMeasurementFormat.MANDERS_FORMAT % (im1_name, im2_name)
    manders_measurement_2 = TemplateMeasurementFormat.MANDERS_FORMAT % (im2_name, im1_name)

    rwc_measurement_1 = TemplateMeasurementFormat.RWC_FORMAT % (im1_name, im2_name)
    rwc_measurement_2 = TemplateMeasurementFormat.RWC_FORMAT % (im2_name, im1_name)

    costes_measurement_1 = TemplateMeasurementFormat.COSTES_FORMAT % (im1_name, im2_name)
    costes_measurement_2 = TemplateMeasurementFormat.COSTES_FORMAT % (im2_name, im1_name)
    
    if MeasurementType.CORRELATION in measurement_types:
        measurements[corr_measurement] = corr
        measurements[slope_measurement] = slope
    if MeasurementType.OVERLAP in measurement_types:        
        measurements[overlap_measurement] = overlap
        measurements[k_measurement_1] = K1
        measurements[k_measurement_2] = K2

    if MeasurementType.MANDERS in measurement_types:
        measurements[manders_measurement_1] = M1
        measurements[manders_measurement_2] = M2
    if MeasurementType.RWC in measurement_types:
        measurements[rwc_measurement_1] = RWC1
        measurements[rwc_measurement_2] = RWC2
    if MeasurementType.COSTES in measurement_types:
        measurements[costes_measurement_1] = C1
        measurements[costes_measurement_2] = C2

    return {"Image": measurements}, summary


def __get_object_result_array(col_order_list: Tuple[str, str, str], measurement_name: str, measurement_array: NDArray[np.float64]) -> List[Tuple[str, str, str, str, str]]:
    summary: List[Tuple[str, str, str, str, str]] = []
    summary += [
        (*col_order_list, f"Mean {measurement_name}", "%.3f" % np.mean(measurement_array)),
        (*col_order_list, f"Median {measurement_name}", "%.3f" % np.median(measurement_array)),
        (*col_order_list, f"Min {measurement_name}", "%.3f" % np.min(measurement_array)),
        (*col_order_list, f"Max {measurement_name}", "%.3f" % np.max(measurement_array)),
    ]
    return summary

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def run_image_pair_objects(
    im1_pixels:         Annotated[NDArray[Pixel], Field(description="First image pixels")],
    im2_pixels:         Annotated[NDArray[Pixel], Field(description="Second image pixels")],
    labels:             Annotated[NDArray[ObjectLabel], Field(description="Labels")],
    object_count:       Annotated[int, Field(description="Object count")],
    im1_name:           Annotated[str, Field(description="First image name")] = "First image",
    im2_name:           Annotated[str, Field(description="Second image name")] = "Second image",
    object_name:        Annotated[str, Field(description="Object name")] = "Objects",
    mask:               Annotated[Optional[ImageGrayscaleMask], Field(description="Mask")] = None, 
    im1_thr_percentage: Annotated[float, Field(description="First image threshold value"), BeforeValidator(np.float64)]=100, 
    im2_thr_percentage: Annotated[float, Field(description="Second image threshold value"), BeforeValidator(np.float64)]=100, 
    im1_costes_pixels:  Annotated[Optional[NDArray[Pixel]], Field(description="First image pixel data for costes")]=None,
    im2_costes_pixels:  Annotated[Optional[NDArray[Pixel]], Field(description="Second image pixel data for costes")]=None,
    measurement_types:  Annotated[List[MeasurementType], Field(description="List of measurement types to be calculated")]=[MeasurementType.CORRELATION, MeasurementType.MANDERS, MeasurementType.RWC, MeasurementType.OVERLAP, MeasurementType.COSTES],
    im1_scale:          Annotated[Optional[Union[float, int]], Field(description="First image scale for costes thresholding")]=None,
    im2_scale:          Annotated[Optional[Union[float, int]], Field(description="Second image scale for costes thresholding")]=None,
    costes_method:      Annotated[Optional[CostesMethod], Field(description="Costes method for costes thresholding")]=CostesMethod.FAST
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        List[Tuple[str, str, str, str, str]]
        ]:
    if MeasurementType.COSTES in measurement_types:
        assert costes_method is not None, "Costes requires a costes method"
    """Calculate per-object correlations between intensities in two images"""
    summary = []

    n_objects = object_count
    # Handle case when both images for the correlation are completely masked out
    corr = np.zeros((0,))
    overlap = np.zeros((0,))
    K1 = np.zeros((0,))
    K2 = np.zeros((0,))
    M1 = np.zeros((0,))
    M2 = np.zeros((0,))
    RWC1 = np.zeros((0,))
    RWC2 = np.zeros((0,))
    C1 = np.zeros((0,))
    C2 = np.zeros((0,))

    if n_objects == 0:
        corr = np.zeros((0,))
        overlap = np.zeros((0,))
        K1 = np.zeros((0,))
        K2 = np.zeros((0,))
        M1 = np.zeros((0,))
        M2 = np.zeros((0,))
        RWC1 = np.zeros((0,))
        RWC2 = np.zeros((0,))
        C1 = np.zeros((0,))
        C2 = np.zeros((0,))
    elif mask is not None and np.where(mask)[0].__len__() == 0:
        corr = np.zeros((n_objects,))
        corr[:] = np.NaN
        overlap = K1 = K2 = M1 = M2 = RWC1 = RWC2 = C1 = C2 = corr
    else:
        lrange = np.arange(n_objects, dtype=np.int32) + 1

        if MeasurementType.CORRELATION in measurement_types:
            corr = measure_correlation_and_slope_from_objects(im1_pixels, im2_pixels, labels, lrange)
            col_order_1 = (im1_name, im2_name, object_name)
            summary += __get_object_result_array(col_order_1, "Correlation coeff", corr)

        if set(measurement_types).intersection({MeasurementType.MANDERS, MeasurementType.RWC, MeasurementType.OVERLAP}) != set():
            # Get channel-specific thresholds from thresholds array
            im1_threshold = im1_thr_percentage
            im2_threshold = im2_thr_percentage
            (
                im1_thr_sum, 
                im2_thr_sum, 
                thr_mask_intersection,
            ) = get_thresholded_images_and_counts (im1_pixels, im2_pixels, im1_threshold, im2_threshold, labels)

            if MeasurementType.MANDERS in measurement_types:
                M1, M2 = measure_manders_coefficient_from_objects(im1_pixels, im2_pixels, im1_thr_sum, im2_thr_sum, thr_mask_intersection,labels, lrange)
                summary += __get_object_result_array((im1_name, im2_name, object_name), "Manders coeff", M1)
                summary += __get_object_result_array((im2_name, im1_name, object_name), "Manders coeff", M2)


            if MeasurementType.RWC in measurement_types:
                RWC1, RWC2 = measure_rwc_coefficient_from_objects(im1_pixels, im2_pixels, im1_thr_sum, im2_thr_sum, thr_mask_intersection,labels, lrange)
                summary += __get_object_result_array((im1_name, im2_name, object_name), "RWC coeff", RWC1)
                summary += __get_object_result_array((im2_name, im1_name, object_name), "RWC coeff", RWC2)


            if MeasurementType.OVERLAP in measurement_types:
                overlap, K1, K2 = measure_overlap_coefficient_from_objects(im1_pixels, im2_pixels, thr_mask_intersection,labels, lrange)
                summary += __get_object_result_array((im1_name, im2_name, object_name), "Overlap coeff", overlap)


        if MeasurementType.COSTES in measurement_types:
            C1, C2 = measure_costes_coefficient_from_objects(
                im1_pixels = im1_pixels, 
                im2_pixels = im2_pixels, 
                im1_costes_pixels = im1_costes_pixels, 
                im2_costes_pixels = im2_costes_pixels, 
                labels = labels, 
                lrange = lrange, 
                im1_scale = im1_scale, 
                im2_scale = im2_scale, 
                costes_method = costes_method,
            )
            summary += __get_object_result_array((im1_name, im2_name, object_name), "Manders coeff (Costes)", C1)
            summary += __get_object_result_array((im2_name, im1_name, object_name), "Manders coeff (Costes)", C2)


    measurements: Dict[str, NDArray[np.float64]] = {}
    if MeasurementType.CORRELATION in measurement_types:
        measurement = TemplateMeasurementFormat.CORRELATION_FORMAT % (im1_name, im2_name)
        measurements[measurement] = corr
    if MeasurementType.MANDERS in measurement_types:
        manders_measurement_1 = TemplateMeasurementFormat.MANDERS_FORMAT % (im1_name, im2_name)
        manders_measurement_2 = TemplateMeasurementFormat.MANDERS_FORMAT % (im2_name, im1_name)
        
        measurements[manders_measurement_1] = M1
        measurements[manders_measurement_2] = M2
    
    if MeasurementType.RWC in measurement_types:
        rwc_measurement_1 = TemplateMeasurementFormat.RWC_FORMAT % (im1_name, im2_name)
        rwc_measurement_2 = TemplateMeasurementFormat.RWC_FORMAT % (im2_name, im1_name)

        measurements[rwc_measurement_1] = RWC1
        measurements[rwc_measurement_2] = RWC2
    
    if MeasurementType.OVERLAP in measurement_types:
        overlap_measurement = TemplateMeasurementFormat.OVERLAP_FORMAT % (im1_name, im2_name)
        k_measurement_1 = TemplateMeasurementFormat.K_FORMAT % (im1_name, im2_name)
        k_measurement_2 = TemplateMeasurementFormat.K_FORMAT % (im2_name, im1_name)
        
        measurements[overlap_measurement] = overlap
        measurements[k_measurement_1] = K1
        measurements[k_measurement_2] = K2
    
    if MeasurementType.COSTES in measurement_types:
        costes_measurement_1 = TemplateMeasurementFormat.COSTES_FORMAT % (im1_name, im2_name)
        costes_measurement_2 = TemplateMeasurementFormat.COSTES_FORMAT % (im2_name, im1_name)
        
        measurements[costes_measurement_1] = C1
        measurements[costes_measurement_2] = C2

    if n_objects == 0:
        col_order_1 = (im1_name, im2_name, object_name)
        summary +=  [
            (*col_order_1, "Mean correlation", "-"),
            (*col_order_1, "Median correlation", "-"),
            (*col_order_1, "Min correlation", "-"),
            (*col_order_1, "Max correlation", "-"),
        ]
        
    #
    # Calculate aggregate statistics
    #
    image_measurements: Dict[str, float] = {}
    for name, array in measurements.items():
        if array.size > 0:
            image_measurements[f"Mean_{name}"] = np.mean(array)
            image_measurements[f"Median_{name}"] = np.median(array)
            image_measurements[f"Min_{name}"] = np.min(array)
            image_measurements[f"Max_{name}"] = np.max(array)
            image_measurements[f"StDev_{name}"] = np.std(array)
        else:
             image_measurements[f"Mean_{name}"] = np.nan
             image_measurements[f"Median_{name}"] = np.nan
             image_measurements[f"Min_{name}"] = np.nan
             image_measurements[f"Max_{name}"] = np.nan
             image_measurements[f"StDev_{name}"] = np.nan

    return {
        "Object": {object_name: measurements},
        "Image": image_measurements
    }, summary

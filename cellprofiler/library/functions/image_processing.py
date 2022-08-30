# from multiprocessing.sharedctypes import Value
import skimage.color
import skimage.morphology
import numpy
import centrosome.threshold
from cellprofiler.library.functions.library_utils import Threshold

def rgb_to_greyscale(image):
    if image.shape[-1] == 4:
        output = skimage.color.rgba2rgb(image)
        return skimage.color.rgb2gray(output)
    else:
        return skimage.color.rgb2gray(image)

def medial_axis(image):
    if image.ndim > 2 and image.shape[-1] in (3, 4):
        raise ValueError("Convert image to grayscale or use medialaxis module")
    if image.ndim > 2 and image.shape[-1] not in (3, 4):
        raise ValueError("Process 3D images plane-wise or the medialaxis module")
    return skimage.morphology.medial_axis(image)

def get_threshold_robust_background(image,
                                    lower_outlier_fraction=0.05,
                                    upper_outlier_fraction=0.05,
                                    averaging_method="Mean",
                                    variance_method="Standard deviation",
                                    number_of_deviations=2,
                                    ):
    """Calculate threshold based on mean & standard deviation.
        The threshold is calculated by trimming the top and bottom 5% of
        pixels off the image, then calculating the mean and standard deviation
        of the remaining image. The threshold is then set at 2 (empirical
        value) standard deviations above the mean.

        
        lower_outlier_fraction - after ordering the pixels by intensity, remove
            the pixels from 0 to len(image) * lower_outlier_fraction from
            the threshold calculation (default = 0.05).
        upper_outlier_fraction - remove the pixels from
            len(image) * (1 - upper_outlier_fraction) to len(image) from
            consideration (default = 0.05).
        averaging_method - Determines how the intensity midpoint is determined 
            after discarding outliers. (default "Mean". Options: "Mean", "Median",
            "Mode").
        variance_method - Method to calculate variance (default = 
            "Standard deviation". Options: "Standard deviation", 
            "Median absolute deviation")
        number_of_deviations - Following calculation of the standard deviation 
            or MAD, multiply this number and add to the average to get the final
            threshold (default = 2)
        average_fn - function used to calculate the average intensity (e.g.
            np.mean, np.median or some sort of mode function). Default = np.mean
        variance_fn - function used to calculate the amount of variance.
                        Default = np.sd
    """

    if averaging_method.casefold() == "mean":
        average_fn = numpy.mean
    elif averaging_method.casefold() == "median":
        average_fn = numpy.median
    elif averaging_method.casefold() == "mode":
        average_fn = centrosome.threshold.binned_mode
    else:
        raise ValueError(f"{averaging_method} not in 'Mean', 'Median', 'Mode'")

    if variance_method.casefold() == "standard deviation":
        variance_fn = numpy.std
    elif variance_method.casefold() == "median absolute deviation":
        variance_fn = centrosome.threshold.mad
    else:
        raise ValueError(f"{variance_method} not in 'standard deviation', 'median absolute deviation'")

    flat_image = image.flatten()
    n_pixels = len(flat_image)
    if n_pixels < 3:
        return 0

    flat_image.sort()
    if flat_image[0] == flat_image[-1]:
        return flat_image[0]
    low_chop = int(round(n_pixels * lower_outlier_fraction))
    hi_chop = n_pixels - int(round(n_pixels * upper_outlier_fraction))
    im = flat_image if low_chop == 0 else flat_image[low_chop:hi_chop]
    mean = average_fn(im)
    sd = variance_fn(im)
    return mean + sd * number_of_deviations

def get_threshold(
        image, 
        mask=None,
        threshold_operation="Minimum Cross-Entropy",
        two_class_otsu="Two classes",
        assign_middle_to_foreground="Foreground",
        log_transform=False,
        manual_threshold=0,
        thresholding_measurement=None,
        threshold_scope="Global",
        threshold_correction_factor=1,
        threshold_range_min=0,
        threshold_range_max=1,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method="Mean",
        variance_method="Standard deviation",
        number_of_deviations=2,
        volumetric=False,
        automatic=False,
        ):
    """
    Get the threshold for an image with the provided settings
    """
    threshold = Threshold(
        threshold_operation,
        two_class_otsu,
        assign_middle_to_foreground,
        log_transform,
        manual_threshold,
        thresholding_measurement,
        threshold_scope,
        threshold_correction_factor,
        threshold_range_min,
        threshold_range_max,
        adaptive_window_size,
        lower_outlier_fraction,
        upper_outlier_fraction,
        averaging_method,
        variance_method,
        number_of_deviations,
        volumetric
    )
    # No mask provided, create a dummy mask
    if mask is not None:
        mask = numpy.ones(image.shape, dtype=bool)

    need_transform = (
            not automatic and
            threshold_operation.casefold() in ("minimum cross-entropy", "otsu") and
            log_transform
    )

    if need_transform:
        image, conversion_dict = centrosome.threshold.log_transform(image)

    if threshold_operation.casefold() == "manual":
        return manual_threshold, manual_threshold, None

    elif threshold_operation.casefold() == "measurement":
        # Thresholds are stored as single element arrays.  Cast to float to extract the value.
        orig_threshold = float(thresholding_measurement)

        return threshold._correct_global_threshold(orig_threshold), orig_threshold, None

    elif threshold_scope.casefold() == "global" or automatic:
        th_guide = None
        th_original = threshold.get_global_threshold(image, mask, automatic=automatic)

    elif threshold_scope.casefold() == "adaptive":
        th_guide = threshold.get_global_threshold(image, mask)
        th_original = threshold.get_local_threshold(image, mask)
    else:
        raise ValueError("Invalid thresholding settings")

    if need_transform:
        if image.min() < 0 or image.max() > 1:
            raise ValueError("For log transform, image pixel values must be between [0, 1]")
        th_original = centrosome.threshold.inverse_log_transform(th_original, conversion_dict)
        if th_guide is not None:
            th_guide = centrosome.threshold.inverse_log_transform(th_guide, conversion_dict)

    if threshold_scope.casefold() == "global" or automatic:
        th_corrected = threshold._correct_global_threshold(th_original)
    else:
        th_guide = threshold._correct_global_threshold(th_guide)
        th_corrected = threshold._correct_local_threshold(th_original, th_guide)

    return th_corrected, th_original, th_guide

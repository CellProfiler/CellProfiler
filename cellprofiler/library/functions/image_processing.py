# from multiprocessing.sharedctypes import Value
import skimage.color
import skimage.morphology
import numpy
import centrosome.threshold
import scipy


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
        raise ValueError("Process 3D images plane-wise or use the medialaxis module")
    return skimage.morphology.medial_axis(image)


def get_threshold_robust_background(
    image,
    lower_outlier_fraction=0.05,
    upper_outlier_fraction=0.05,
    averaging_method="mean",
    variance_method="standard_deviation",
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

    if variance_method.casefold() == "standard_deviation":
        variance_fn = numpy.std
    elif variance_method.casefold() == "median_absolute_deviation":
        variance_fn = centrosome.threshold.mad
    else:
        raise ValueError(
            f"{variance_method} not in 'standard_deviation', 'median_absolute_deviation'"
        )

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


def get_adaptive_threshold(
    image,
    mask=None,
    threshold_method="otsu",
    window_size=50,
    threshold_min=0,
    threshold_max=1,
    threshold_correction_factor=1,
    assign_middle_to_foreground="foreground",
    global_limits=[0.7, 1.5],
    log_transform=False,
    volumetric=False,
    **kwargs,
):

    if mask is not None:
        # Apply mask and preserve image shape
        image = numpy.where(mask, image, False)

    if volumetric:
        # Array to store threshold values
        thresh_out = numpy.zeros(image.shape)
        for z in range(image.shape[2]):
            thresh_out[:, :, z] = get_adaptive_threshold(
                image[:, :, z],
                mask=None,  # Mask has already been applied
                threshold_method=threshold_method,
                window_size=window_size,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_correction_factor=threshold_correction_factor,
                assign_middle_to_foreground=assign_middle_to_foreground,
                global_limits=global_limits,
                log_transform=log_transform,
                volumetric=False,  # Processing a single plane, so volumetric=False
                **kwargs,
            )
        return thresh_out

    if log_transform:
        image, conversion_dict = centrosome.threshold.log_transform(image)
    bin_wanted = 0 if assign_middle_to_foreground.casefold() == "foreground" else 1

    thresh_out = None

    if len(image) == 0 or numpy.all(image == numpy.nan):
        thresh_out = numpy.zeros_like(image)
    elif numpy.all(image == image.ravel()[0]):
        thresh_out = numpy.full_like(image, image.ravel()[0])
    # Define the threshold method to be run in each adaptive window
    elif threshold_method.casefold() == "otsu":
        threshold_fn = skimage.filters.threshold_otsu
    elif threshold_method.casefold() == "multiotsu":
        threshold_fn = skimage.filters.threshold_multiotsu
        # If nbins not set in kwargs, use default 128
        kwargs["nbins"] = kwargs["nbins"] if "nbins" in kwargs else 128
    elif threshold_method.casefold() == "minimum_cross_entropy":
        tol = max(numpy.min(numpy.diff(numpy.unique(image))) / 2, 0.5 / 65536)
        kwargs["tolerance"] = tol
        threshold_fn = skimage.filters.threshold_li
    elif threshold_method.casefold() == "robust_background":
        threshold_fn = get_threshold_robust_background
        kwargs["lower_outlier_fraction"] = (
            kwargs["lower_outlier_fraction"]
            if "lower_outlier_fraction" in kwargs
            else 0.05
        )
        kwargs["upper_outlier_fraction"] = (
            kwargs["upper_outlier_fraction"]
            if "upper_outlier_fraction" in kwargs
            else 0.05
        )
        kwargs["averaging_method"] = (
            kwargs["averaging_method"] if "averaging_method" in kwargs else "mean"
        )
        kwargs["variance_method"] = (
            kwargs["variance_method"]
            if "variance_method" in kwargs
            else "standard_deviation"
        )
        kwargs["number_of_deviations"] = (
            kwargs["number_of_deviations"] if "number_of_deviations" in kwargs else 2
        )
    elif threshold_method.casefold() == "sauvola":
        if window_size % 2 == 0:
            window_size += 1
        thresh_out = skimage.filters.threshold_sauvola(image, window_size)
    else:
        raise NotImplementedError(f"Threshold method {threshold_method} not supported.")

    if thresh_out is None:
        image_size = numpy.array(image.shape[:2], dtype=int)
        nblocks = image_size // window_size
        if any(n < 2 for n in nblocks):
            raise ValueError(
                "Adaptive window cannot exceed 50%% of an image dimension.\n"
                "Window of %dpx is too large for a %sx%s image"
                % (window_size, image_size[1], image_size[0])
            )
        #
        # Use a floating point block size to apportion the roundoff
        # roughly equally to each block
        #
        increment = numpy.array(image_size, dtype=float) / numpy.array(
            nblocks, dtype=float
        )
        #
        # Put the answer here
        #
        thresh_out = numpy.zeros(image_size, image.dtype)
        #
        # Loop once per block, computing the "global" threshold within the
        # block.
        #
        block_threshold = numpy.zeros([nblocks[0], nblocks[1]])
        for i in range(nblocks[0]):
            i0 = int(i * increment[0])
            i1 = int((i + 1) * increment[0])
            for j in range(nblocks[1]):
                j0 = int(j * increment[1])
                j1 = int((j + 1) * increment[1])
                block = image[i0:i1, j0:j1]
                block = block[~numpy.logical_not(block)]
                if len(block) == 0:
                    threshold_out = 0.0
                elif numpy.all(block == block[0]):
                    # Don't compute blocks with only 1 value.
                    threshold_out = block[0]
                elif threshold_method == "multiotsu" and len(numpy.unique(block)) < 3:
                    # Region within window has only 2 values.
                    # Can't run 3-class otsu on only 2 values.
                    threshold_out = skimage.filters.threshold_otsu(block)
                else:
                    try:
                        threshold_out = threshold_fn(block, **kwargs)
                    except ValueError:
                        # Drop nbins kwarg when multi-otsu fails. See issue #6324 scikit-image
                        threshold_out = threshold_fn(block)
                if isinstance(threshold_out, numpy.ndarray):
                    # Select correct bin if running multiotsu
                    threshold_out = threshold_out[bin_wanted]
                block_threshold[i, j] = threshold_out
        #
        # Use a cubic spline to blend the thresholds across the image to avoid image artifacts
        #
        spline_order = min(3, numpy.min(nblocks) - 1)
        xStart = int(increment[0] / 2)
        xEnd = int((nblocks[0] - 0.5) * increment[0])
        yStart = int(increment[1] / 2)
        yEnd = int((nblocks[1] - 0.5) * increment[1])
        xtStart = 0.5
        xtEnd = image.shape[0] - 0.5
        ytStart = 0.5
        ytEnd = image.shape[1] - 0.5
        block_x_coords = numpy.linspace(xStart, xEnd, nblocks[0])
        block_y_coords = numpy.linspace(yStart, yEnd, nblocks[1])
        adaptive_interpolation = scipy.interpolate.RectBivariateSpline(
            block_x_coords,
            block_y_coords,
            block_threshold,
            bbox=(xtStart, xtEnd, ytStart, ytEnd),
            kx=spline_order,
            ky=spline_order,
        )
        thresh_out_x_coords = numpy.linspace(
            0.5, int(nblocks[0] * increment[0]) - 0.5, thresh_out.shape[0]
        )
        thresh_out_y_coords = numpy.linspace(
            0.5, int(nblocks[1] * increment[1]) - 0.5, thresh_out.shape[1]
        )
        # Smooth out the "blocky" adaptive threshold
        thresh_out = adaptive_interpolation(thresh_out_x_coords, thresh_out_y_coords)

    # Get global threshold
    global_threshold = get_global_threshold(
        image,
        mask,
        threshold_method,
        threshold_min,
        threshold_max,
        threshold_correction_factor,
        assign_middle_to_foreground,
        log_transform=log_transform,
    )

    if log_transform:
        # Revert the log transformation
        thresh_out = centrosome.threshold.inverse_log_transform(
            thresh_out, conversion_dict
        )
        global_threshold = centrosome.threshold.inverse_log_transform(
            global_threshold, conversion_dict
        )

    # Apply threshold_correction
    thresh_out *= threshold_correction_factor

    t_min = max(threshold_min, global_threshold * global_limits[0])
    t_max = min(threshold_max, global_threshold * global_limits[1])
    thresh_out[thresh_out < t_min] = t_min
    thresh_out[thresh_out > t_max] = t_max
    return thresh_out


def get_global_threshold(
    image,
    mask=None,
    threshold_method="otsu",
    threshold_min=0,
    threshold_max=1,
    threshold_correction_factor=1,
    assign_middle_to_foreground="foreground",
    log_transform=False,
    **kwargs,
):

    if log_transform:
        image, conversion_dict = centrosome.threshold.log_transform(image)

    if mask is not None:
        # Apply mask and discard masked pixels
        image = image[mask]

    # Shortcuts - Check if image array is empty or all pixels are the same value.
    if len(image) == 0:
        threshold = 0.0
    elif numpy.all(image == image.ravel()[0]):
        # All pixels are the same value
        threshold = image.ravel()[0]

    elif threshold_method.casefold() in ("minimum_cross_entropy", "sauvola"):
        tol = max(numpy.min(numpy.diff(numpy.unique(image))) / 2, 0.5 / 65536)
        threshold = skimage.filters.threshold_li(image, tolerance=tol)
    elif threshold_method.casefold() == "robust_background":
        threshold = get_threshold_robust_background(image, **kwargs)
    elif threshold_method.casefold() == "otsu":
        threshold = skimage.filters.threshold_otsu(image)
    elif threshold_method.casefold() == "multiotsu":
        bin_wanted = 0 if assign_middle_to_foreground.casefold() == "foreground" else 1
        kwargs["nbins"] = kwargs["nbins"] if "nbins" in kwargs else 128
        threshold = skimage.filters.threshold_multiotsu(image, **kwargs)
        threshold = threshold[bin_wanted]
    else:
        raise NotImplementedError(f"Threshold method {threshold_method} not supported.")

    if log_transform:
        threshold = centrosome.threshold.inverse_log_transform(
            threshold, conversion_dict
        )

    threshold *= threshold_correction_factor
    threshold = min(max(threshold, threshold_min), threshold_max)
    return threshold


def apply_threshold(image, threshold, mask=None, smoothing=0):
    if mask is None:
        # Create a fake mask if one isn't provided
        mask = numpy.full(image.shape, True)
    if smoothing == 0:
        return (image >= threshold) & mask
    else:
        # Convert from a scale into a sigma. What I've done here
        # is to structure the Gaussian so that 1/2 of the smoothed
        # intensity is contributed from within the smoothing diameter
        # and 1/2 is contributed from outside.
        sigma = smoothing / 0.6744 / 2.0

    blurred_image = centrosome.smooth.smooth_with_function_and_mask(
        image,
        lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
        mask,
    )
    return (blurred_image >= threshold) & mask

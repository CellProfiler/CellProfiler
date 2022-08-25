# from multiprocessing.sharedctypes import Value
import skimage.color
import skimage.morphology
import numpy
import centrosome.threshold

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



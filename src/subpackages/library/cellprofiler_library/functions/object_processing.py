
import centrosome.cpmorphology
import numpy
import scipy.ndimage
import skimage.morphology
import cellprofiler.utilities.morphology
import mahotas
import matplotlib.cm
from numpy.typing import NDArray
from typing import Optional, Literal, Tuple
from cellprofiler_library.types import ImageAnyMask, ObjectLabel, ImageColor, ImageGrayscale, ImageBinary, ObjectSegmentation, StructuringElement

def shrink_to_point(labels, fill):
    """
    Remove all pixels but one from filled objects.
    If `fill` = False, thin objects with holes to loops.
    """

    if fill:
        labels=centrosome.cpmorphology.fill_labeled_holes(labels)
    return centrosome.cpmorphology.binary_shrink(labels)

def shrink_defined_pixels(labels, fill, iterations):
    """
    Remove pixels around the perimeter of an object unless
    doing so would change the object’s Euler number `iterations` times. 
    Processing stops automatically when there are no more pixels to
    remove.
    """

    if fill:
        labels=centrosome.cpmorphology.fill_labeled_holes(labels)
    return centrosome.cpmorphology.binary_shrink(
                labels, iterations=iterations
            )

def add_dividing_lines(labels):
    """
    Remove pixels from an object that are adjacent to
    another object’s pixels unless doing so would change the object’s
    Euler number
    """

    adjacent_mask = centrosome.cpmorphology.adjacent(labels)

    thinnable_mask = centrosome.cpmorphology.binary_shrink(labels, 1) != 0

    out_labels = labels.copy()

    out_labels[adjacent_mask & ~thinnable_mask] = 0

    return out_labels

def skeletonize(labels):
    """
    Erode each object to its skeleton.
    """
    return centrosome.cpmorphology.skeletonize_labels(labels)

def despur(labels, iterations):
    """
    Remove or reduce the length of spurs in a skeletonized
    image. The algorithm reduces spur size by `iterations` pixels.
    """
    return centrosome.cpmorphology.spur(
                labels, iterations=iterations
            )

def expand(labels, distance):
    """
    Expand labels by a specified distance.
    """

    background = labels == 0

    distances, (i, j) = scipy.ndimage.distance_transform_edt(
        background, return_indices=True
    )

    out_labels = labels.copy()

    mask = background & (distances <= distance)

    out_labels[mask] = labels[i[mask], j[mask]]

    return out_labels

def expand_until_touching(labels):
    """
    Expand objects, assigning every pixel in the
    image to an object. Background pixels are assigned to the nearest
    object.
    """
    distance = numpy.max(labels.shape)
    return expand(labels, distance)

def expand_defined_pixels(labels, iterations):
    """
    Expand each object by adding background pixels
    adjacent to the image `iterations` times. Processing stops 
    automatically if there are no more background pixels.
    """
    return expand(labels, iterations)

def merge_objects(labels_x, labels_y, dimensions):
    """
    Make overlapping objects combine into a single object, taking 
    on the label of the object from the initial set.

    If an object overlaps multiple objects, each pixel of the added 
    object will be assigned to the closest object from the initial 
    set. This is primarily useful when the same objects appear in 
    both sets.
    """
    output = numpy.zeros_like(labels_x)
    labels_y[labels_y > 0] += labels_x.max()
    indices_x = numpy.unique(labels_x)
    indices_x = indices_x[indices_x > 0]
    indices_y = numpy.unique(labels_y)
    indices_y = indices_y[indices_y > 0]
    # Resolve non-conflicting labels first
    undisputed = numpy.logical_xor(labels_x > 0, labels_y > 0)
    undisputed_x = numpy.setdiff1d(indices_x, labels_x[~undisputed])
    mask = numpy.isin(labels_x, undisputed_x)
    output = numpy.where(mask, labels_x, output)
    labels_x[mask] = 0
    undisputed_y = numpy.setdiff1d(indices_y, labels_y[~undisputed])
    mask = numpy.isin(labels_y, undisputed_y)
    output = numpy.where(mask, labels_y, output)
    labels_y[mask] = 0
    to_segment = numpy.logical_or(labels_x > 0, labels_y > 0)
    if dimensions == 2: 
        distances, (i, j) = scipy.ndimage.distance_transform_edt(
            labels_x == 0, return_indices=True
        )
        output[to_segment] = labels_x[i[to_segment], j[to_segment]]
    if dimensions == 3:
        distances, (i, j, v) = scipy.ndimage.distance_transform_edt(
            labels_x == 0, return_indices=True
        )
        output[to_segment] = labels_x[i[to_segment], j[to_segment], v[to_segment]]
    
    return output

def preserve_objects(labels_x, labels_y):
    """
    Preserve the initial object set. Any overlapping regions from 
    the second set will be ignored in favour of the object from 
    the initial set. 
    """
    labels_y[labels_y > 0] += labels_x.max()
    return numpy.where(labels_x > 0, labels_x, labels_y)

def discard_objects(labels_x, labels_y):
    """
    Discard objects that overlap with objects in the initial set
    """
    output = numpy.zeros_like(labels_x)
    indices_x = numpy.unique(labels_x)
    indices_x = indices_x[indices_x > 0]
    indices_y = numpy.unique(labels_y)
    indices_y = indices_y[indices_y > 0]
    # Resolve non-conflicting labels first
    undisputed = numpy.logical_xor(labels_x > 0, labels_y > 0)
    undisputed_x = numpy.setdiff1d(indices_x, labels_x[~undisputed])
    mask = numpy.isin(labels_x, undisputed_x)
    output = numpy.where(mask, labels_x, output)
    labels_x[mask] = 0
    undisputed_y = numpy.setdiff1d(indices_y, labels_y[~undisputed])
    mask = numpy.isin(labels_y, undisputed_y)
    output = numpy.where(mask, labels_y, output)
    labels_y[mask] = 0

    return numpy.where(labels_x > 0, labels_x, output)

def segment_objects(labels_x, labels_y, dimensions):
    """
    Combine object sets and re-draw segmentation for overlapping
    objects.
    """
    output = numpy.zeros_like(labels_x)
    labels_y[labels_y > 0] += labels_x.max()
    indices_x = numpy.unique(labels_x)
    indices_x = indices_x[indices_x > 0]
    indices_y = numpy.unique(labels_y)
    indices_y = indices_y[indices_y > 0]
    # Resolve non-conflicting labels first
    undisputed = numpy.logical_xor(labels_x > 0, labels_y > 0)
    undisputed_x = numpy.setdiff1d(indices_x, labels_x[~undisputed])
    mask = numpy.isin(labels_x, undisputed_x)
    output = numpy.where(mask, labels_x, output)
    labels_x[mask] = 0
    undisputed_y = numpy.setdiff1d(indices_y, labels_y[~undisputed])
    mask = numpy.isin(labels_y, undisputed_y)
    output = numpy.where(mask, labels_y, output)
    labels_y[mask] = 0

    to_segment = numpy.logical_or(labels_x > 0, labels_y > 0)
    disputed = numpy.logical_and(labels_x > 0, labels_y > 0)
    seeds = numpy.add(labels_x, labels_y)
    # Find objects which will be completely removed due to 100% overlap.
    will_be_lost = numpy.setdiff1d(labels_x[disputed], labels_x[~disputed])
    # Check whether this was because an identical object is in both arrays.
    for label in will_be_lost:
        x_mask = labels_x == label
        y_lab = numpy.unique(labels_y[x_mask])
        if not y_lab or len(y_lab) > 1:
            # Labels are not identical
            continue
        else:
            # Get mask of object on y, check if identical to x
            y_mask = labels_y == y_lab[0]
            if numpy.array_equal(x_mask, y_mask):
                # Label is identical
                output[x_mask] = label
                to_segment[x_mask] = False
    seeds[disputed] = 0
    if dimensions == 2:
        distances, (i, j) = scipy.ndimage.distance_transform_edt(
            seeds == 0, return_indices=True
        )
        output[to_segment] = seeds[i[to_segment], j[to_segment]]
    elif dimensions == 3:
        distances, (i, j, v) = scipy.ndimage.distance_transform_edt(
            seeds == 0, return_indices=True
        )
        output[to_segment] = seeds[i[to_segment], j[to_segment], v[to_segment]]

    return output

def watershed(
    input_image: numpy.ndarray,
    mask: numpy.ndarray = None,
    watershed_method: Literal["distance", "markers", "intensity"] = "distance",
    declump_method: Literal["shape", "intensity", "none"] = "shape",
    seed_method: Literal["local", "regional"] = "local",
    intensity_image: numpy.ndarray = None,
    markers_image: numpy.ndarray = None,
    max_seeds: int = -1,
    downsample: int = 1,
    min_distance: int = 1,
    min_intensity: float = 0,
    footprint: int = 8,
    connectivity: int = 1,
    compactness: float = 0.0,
    exclude_border: bool = False,
    watershed_line: bool = False,
    gaussian_sigma: float = 0.0,
    structuring_element: Literal[
        "ball", "cube", "diamond", "disk", "octahedron", "square", "star"
    ] = "disk",
    structuring_element_size: int = 1,
    return_seeds: bool = False,
):
    # Check inputs
    if not numpy.array_equal(input_image, input_image.astype(bool)):
        raise ValueError("Watershed expects a thresholded image as input")
    if (
        watershed_method.casefold() == "intensity" or declump_method.casefold() == "intensity"
    ) and intensity_image is None:
        raise ValueError(f"Intensity-based methods require an intensity image")

    if watershed_method.casefold() == "markers" and markers_image is None:
        raise ValueError("Markers watershed method require a markers image")

    # No declumping, so just label the binary input image
    if declump_method.casefold() == "none":
        if mask is not None:
            input_image[~mask] = 0
        watershed_image = scipy.ndimage.label(input_image)[0]
        if return_seeds:
            return watershed_image, numpy.zeros_like(watershed_image, bool)
        else:
            return watershed_image

    # Create and check structuring element for seed dilation
    strel = getattr(skimage.morphology, structuring_element.casefold())(
        structuring_element_size
    )

    if strel.ndim != input_image.ndim:
        raise ValueError(
            "Structuring element does not match object dimensions: "
            "{} != {}".format(strel.ndim, input_image.ndim)
        )

    if input_image.ndim == 3:
        maxima_footprint = numpy.ones((footprint, footprint, footprint))
    else:
        maxima_footprint = numpy.ones((footprint, footprint))

    # Downsample input image
    if downsample > 1:
        input_shape = input_image.shape
        if input_image.ndim > 2:
            # Only scale x and y
            factors = (1, downsample, downsample)
        else:
            factors = (downsample, downsample)

        input_image = skimage.transform.downscale_local_mean(input_image, factors)
        # Resize optional images
        if intensity_image is not None:
            intensity_image = skimage.transform.downscale_local_mean(
                intensity_image, factors
            )
        if markers_image is not None:
            markers_image = skimage.transform.downscale_local_mean(
                markers_image, factors
            )
        if mask is not None:
            mask = skimage.transform.downscale_local_mean(mask, factors)

    # Only calculate the distance transform if required for shape-based declumping
    # or distance-based seed generation
    if declump_method.casefold() == "shape" or watershed_method.casefold() == "distance":
        smoothed_input_image = skimage.filters.gaussian(
            input_image, sigma=gaussian_sigma
        )
        # Calculate distance transform
        distance = scipy.ndimage.distance_transform_edt(smoothed_input_image)

    # Generate alternative input to the watershed based on declumping
    if declump_method.casefold() == "shape":
        # Invert the distance transform of the input image.
        # The peaks of the distance tranform become the troughs and
        # this image is given as input to watershed
        watershed_input_image = -distance
        # Move to positive realm
        watershed_input_image = watershed_input_image - watershed_input_image.min()
    elif declump_method.casefold() == "intensity":
        # Convert pixel intensity peaks to troughs and
        # use this as the image input in watershed
        watershed_input_image = 1 - intensity_image
    else:
        raise ValueError(f"declump_method {declump_method} is not supported.")

    # Determine image from which to calculate seeds
    if watershed_method.casefold() == "distance":
        seed_image = distance
    elif watershed_method.casefold() == "intensity":
        seed_image = intensity_image
    elif watershed_method.casefold() == "markers":
        # The user has provided their own seeds/markers
        seeds = markers_image
        seeds = skimage.morphology.binary_dilation(seeds, strel)
    else:
        raise NotImplementedError(
            f"watershed method {watershed_method} is not supported"
        )

    if not watershed_method.casefold() == "markers":
        # Generate seeds
        if seed_method.casefold() == "local":
            seed_coords = skimage.feature.peak_local_max(
                seed_image,
                min_distance=min_distance,
                threshold_rel=min_intensity,
                footprint=maxima_footprint,
                num_peaks=max_seeds if max_seeds != -1 else numpy.inf,
                exclude_border=False
            )
            seeds = numpy.zeros(seed_image.shape, dtype=bool)
            seeds[tuple(seed_coords.T)] = True
            seeds = skimage.morphology.binary_dilation(seeds, strel)
            seeds = scipy.ndimage.label(seeds)[0]

        elif seed_method.casefold() == "regional":
            seeds = mahotas.regmax(seed_image, maxima_footprint)
            seeds = skimage.morphology.binary_dilation(seeds, strel)
            seeds = scipy.ndimage.label(seeds)[0]
        else:
            raise NotImplementedError(
                f"seed_method {seed_method} is not supported."
            )

    # Run watershed
    watershed_image = skimage.segmentation.watershed(
        watershed_input_image,
        markers=seeds,
        mask=mask if mask is not None else input_image != 0,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
    )

    # Reverse downsampling
    if downsample > 1:
        watershed_image = skimage.transform.resize(
            watershed_image, input_shape, mode="edge", order=0, preserve_range=True
        )
        watershed_image = numpy.rint(watershed_image).astype(numpy.uint16)

    if exclude_border:
        watershed_image = skimage.segmentation.clear_border(watershed_image)

    if return_seeds:
        # Reverse seed downsampling
        if downsample > 1:
            seeds = skimage.transform.resize(
                seeds, input_shape, mode="edge", order=0, preserve_range=True
            )
            seeds = numpy.rint(seeds).astype(numpy.uint16)
        return watershed_image, seeds
    else:
        return watershed_image

def fill_object_holes(labels, diameter, planewise=False):
    array = labels.copy()
    # Calculate radius from diameter
    radius = diameter / 2.0

    # Check if grayscale, RGB or operation is being performed planewise
    if labels.ndim == 2 or labels.shape[-1] in (3, 4) or planewise:
        # 2D circle area will be calculated
        factor = radius ** 2
    else:
        # Calculate the volume of a sphere
        factor = (4.0 / 3.0) * (radius ** 3)

    min_obj_size = numpy.pi * factor

    if planewise and labels.ndim != 2 and labels.shape[-1] not in (3, 4):
        for plane in array:
            for obj in numpy.unique(plane):
                if obj == 0:
                    continue
                filled_mask = skimage.morphology.remove_small_holes(
                    plane == obj, min_obj_size
                )
                plane[filled_mask] = obj
        return array
    else:
        for obj in numpy.unique(array):
            if obj == 0:
                continue
            filled_mask = skimage.morphology.remove_small_holes(
                array == obj, min_obj_size
            )
            array[filled_mask] = obj
    return array

def fill_convex_hulls(labels):
    data = skimage.measure.regionprops(labels)
    output = numpy.zeros_like(labels)
    for prop in data:
        label = prop["label"]
        bbox = prop["bbox"]
        cmask = prop["convex_image"]
        if len(bbox) <= 4:
            output[bbox[0] : bbox[2], bbox[1] : bbox[3]][cmask] = label
        else:
            output[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]][
                cmask
            ] = label
    return output


# Rename to get_maxima_from_foreground?
def get_maxima(
    image,
    labeled_image=None,
    maxima_mask=None, # This should be renamed to footprint
    image_resize_factor=1.0
):
    """_summary_

    Parameters
    ----------
    image : _type_
        _description_
    labeled_image : _type_
        Binary threshold of the input image
    """
    if image_resize_factor < 1.0:
        shape = numpy.array(image.shape) * image_resize_factor
        i_j = (
            numpy.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
            / image_resize_factor
        )
        resized_image = scipy.ndimage.map_coordinates(image, i_j)
        resized_labels = scipy.ndimage.map_coordinates(
            labeled_image, i_j, order=0
        ).astype(labeled_image.dtype)
    else:
        resized_image = image
        resized_labels = labeled_image
    #
    # find local maxima
    #
    if maxima_mask is not None:
        binary_maxima_image = centrosome.cpmorphology.is_local_maximum(
            resized_image, resized_labels, maxima_mask
        )
        binary_maxima_image[resized_image <= 0] = 0
    else:
        binary_maxima_image = (resized_image > 0) & (labeled_image > 0)
    if image_resize_factor < 1.0:
        inverse_resize_factor = float(image.shape[0]) / float(
            binary_maxima_image.shape[0]
        )
        i_j = (
            numpy.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(float)
            / inverse_resize_factor
        )
        binary_maxima_image = (
            scipy.ndimage.map_coordinates(binary_maxima_image.astype(float), i_j)
            > 0.5
        )
        assert binary_maxima_image.shape[0] == image.shape[0]
        assert binary_maxima_image.shape[1] == image.shape[1]

    # Erode blobs of touching maxima to a single point
    shrunk_image = centrosome.cpmorphology.binary_shrink(binary_maxima_image)
    return shrunk_image


def smooth_image(image, mask=None, filter_size=None, min_obj_size=10):
    """Apply the smoothing filter to the image"""

    if mask is None:
        mask = numpy.ones_like(image, dtype=bool)

    if filter_size is None:
        filter_size = 2.35 * min_obj_size / 3.5

    if filter_size == 0:
        return image
    sigma = filter_size / 2.35
    #
    # We not only want to smooth using a Gaussian, but we want to limit
    # the spread of the smoothing to 2 SD, partly to make things happen
    # locally, partly to make things run faster, partly to try to match
    # the Matlab behavior.
    #
    filter_size = max(int(float(filter_size) / 2.0), 1)
    f = (
        1
        / numpy.sqrt(2.0 * numpy.pi)
        / sigma
        * numpy.exp(
            -0.5 * numpy.arange(-filter_size, filter_size + 1) ** 2 / sigma ** 2
        )
    )

    def fgaussian(image):
        output = scipy.ndimage.convolve1d(image, f, axis=0, mode="constant")
        return scipy.ndimage.convolve1d(output, f, axis=1, mode="constant")

    #
    # Use the trick where you similarly convolve an array of ones to find
    # out the edge effects, then divide to correct the edge effects
    #
    edge_array = fgaussian(mask.astype(float))
    masked_image = image.copy()
    masked_image[~mask] = 0
    smoothed_image = fgaussian(masked_image)
    masked_image[mask] = smoothed_image[mask] / edge_array[mask]
    return masked_image


def filter_on_size(labeled_image, min_size, max_size, return_only_small=False):
    """ Filter the labeled image based on the size range

    labeled_image - pixel image labels
    object_count - # of objects in the labeled image
    returns the labeled image, and the labeled image with the
    small objects removed
    """
    labeled_image = labeled_image.copy()

    object_count = len(numpy.unique(labeled_image))
    # Check if there are no labelled objects
    if object_count == 1:
        small_removed_labels = labeled_image.copy()
    else:
        areas = scipy.ndimage.measurements.sum(
            numpy.ones(labeled_image.shape),
            labeled_image,
            numpy.array(list(range(0, object_count)), dtype=numpy.int32),
        )
        areas = numpy.array(areas, dtype=int)
        min_allowed_area = (
            numpy.pi * (min_size * min_size) / 4
        )
        max_allowed_area = (
            numpy.pi * (max_size * max_size) / 4
        )
        # area_image has the area of the object at every pixel within the object
        area_image = areas[labeled_image]
        labeled_image[area_image < min_allowed_area] = 0
        if return_only_small:
            small_removed_labels = labeled_image.copy()
            labeled_image[area_image > max_allowed_area] = 0
            return labeled_image, small_removed_labels
        else:
            labeled_image[area_image > max_allowed_area] = 0
            return labeled_image


def filter_on_border(labeled_image, mask=None):
    """Filter out objects touching the border

    In addition, if the image has a mask, filter out objects
    touching the border of the mask.
    """
    labeled_image = labeled_image.copy()

    border_labels = list(labeled_image[0, :])
    border_labels.extend(labeled_image[:, 0])
    border_labels.extend(labeled_image[labeled_image.shape[0] - 1, :])
    border_labels.extend(labeled_image[:, labeled_image.shape[1] - 1])
    border_labels = numpy.array(border_labels)
    #
    # the following histogram has a value > 0 for any object
    # with a border pixel
    #
    histogram = scipy.sparse.coo_matrix(
        (
            numpy.ones(border_labels.shape),
            (border_labels, numpy.zeros(border_labels.shape)),
        ),
        shape=(numpy.max(labeled_image) + 1, 1),
    ).todense()
    histogram = numpy.array(histogram).flatten()
    if any(histogram[1:] > 0):
        histogram_image = histogram[labeled_image]
        labeled_image[histogram_image > 0] = 0
    elif mask is not None:
        # The assumption here is that, if nothing touches the border,
        # the mask is a large, elliptical mask that tells you where the
        # well is. That's the way the old Matlab code works and it's duplicated here
        #
        # The operation below gets the mask pixels that are on the border of the mask
        # The erosion turns all pixels touching an edge to zero. The not of this
        # is the border + formerly masked-out pixels.
        mask_border = numpy.logical_not(
            scipy.ndimage.binary_erosion(image.mask)
        )
        mask_border = numpy.logical_and(mask_border, image.mask)
        border_labels = labeled_image[mask_border]
        border_labels = border_labels.flatten()
        histogram = scipy.sparse.coo_matrix(
            (
                numpy.ones(border_labels.shape),
                (border_labels, numpy.zeros(border_labels.shape)),
            ),
            shape=(numpy.max(labeled_image) + 1, 1),
        ).todense()
        histogram = numpy.array(histogram).flatten()
        if any(histogram[1:] > 0):
            histogram_image = histogram[labeled_image]
            labeled_image[histogram_image > 0] = 0
    return labeled_image

    def separate_neighboring_objects(
        image,
        mask=None,
        unclump_method: Literal["intensity", "shape", "none"] = "intensity",
        watershed_method: Literal["intensity", "shape", "propagate", "none"] = "intensity",
        fill_holes: Literal["never", "thresholding", "declumping"] = "thresholding",
        filter_size=None,
        min_size=10, 
        max_size=40,
        low_res_maxima=False,
        maxima_suppression_size=7,
        automatic_suppression=False,
        ):

        # Expects a thresholded image
        if not numpy.array_equal(input_image, input_image.astype(bool)):
            raise ValueError("separate_neighboring_objects expects a thresholded image as input")

        # Label the thresholded image
        labeled_image = sicpy.ndimage.label(
            image, numpy.ones((3, 3), bool)
        )

        if unclump_method.casefold() == "none" and watershed_method.casefold() == "none":
            return labeled_image
        
        blurred_image = smooth_image(image, mask)

        # For image resizing, the min_size must be larger than 10
        if min_size > 10 and low_res_maxima:
            image_resize_factor = 10.0 / float(min_size)
            if automatic_suppression:
                maxima_suppression_size = 7
            else:
                maxima_suppression_size = (
                    maxima_suppression_size * image_resize_factor + 0.5
                )
        else:
            image_resize_factor = 1.0
            if automatic_suppression:
                maxima_suppression_size = min_size / 1.5
            else:
                maxima_suppression_size = maxima_suppression_size
        
        maxima_mask = centrosome.cpmorphology.strel_disk(
            max(1, maxima_suppression_size - 0.5)
        )

        distance_transformed_image = None

        if unclump_method.casefold() == "intensity":
            # Remove dim maxima
            maxima_image = get_maxima(
                blurred_image,
                labeled_image,
                maxima_mask,
                image_resize_factor
                )

#############################################################
# ConvertObjectsToImage
#############################################################

def image_mode_black_and_white(
        pixel_data:     ImageBinary,
        mask:           ImageAnyMask, 
        alpha:          NDArray[numpy.int32],
        labels:         Optional[NDArray[ObjectLabel]] = None,
        colormap_value: Optional[str] = None
        ) -> Tuple[ImageBinary, NDArray[numpy.int32]]:
    pixel_data[mask] = True
    alpha[mask] = 1
    return pixel_data.astype(numpy.bool_), alpha

def image_mode_grayscale(
        pixel_data:     ImageGrayscale,
        mask:           ImageAnyMask,
        alpha:          NDArray[numpy.int32],
        labels:         NDArray[ObjectLabel],
        colormap_value: Optional[str] = None
        ) -> Tuple[ImageGrayscale, NDArray[numpy.int32]]:
    pixel_data[mask] = labels[mask].astype(float) / numpy.max(labels)
    alpha[mask] = 1
    return pixel_data.astype(numpy.float32), alpha

def image_mode_color(
        pixel_data:     ImageColor,
        mask:           ImageAnyMask,
        alpha:          NDArray[numpy.int32],
        labels:         NDArray[ObjectLabel],
        colormap_value: str
        ) -> Tuple[ImageColor, NDArray[numpy.int32]]:
    if colormap_value == "colorcube":
        # Colorcube missing from matplotlib
        cm_name = "gist_rainbow"
    elif colormap_value == "lines":
        # Lines missing from matplotlib and not much like it,
        # Pretty boring palette anyway, hence
        cm_name = "Pastel1"
    elif colormap_value == "white":
        # White missing from matplotlib, it's just a colormap
        # of all completely white... not even different kinds of
        # white. And, isn't white just a uniform sampling of
        # frequencies from the spectrum?
        cm_name = "Spectral"
    else:
        cm_name = colormap_value

    cm = matplotlib.cm.get_cmap(cm_name)

    mapper = matplotlib.cm.ScalarMappable(cmap=cm)

    if labels.ndim == 3:
        for index, plane in enumerate(mask):
            pixel_data[index, plane, :] = mapper.to_rgba(
                centrosome.cpmorphology.distance_color_labels(labels[index])
            )[plane, :3]
    else:
        pixel_data[mask, :] += mapper.to_rgba(
            centrosome.cpmorphology.distance_color_labels(labels)
        )[mask, :3]

    alpha[mask] += 1
    return pixel_data.astype(numpy.float32), alpha

def image_mode_uint16(
        pixel_data:     NDArray[numpy.int32],
        mask:           ImageAnyMask,
        alpha:          NDArray[numpy.int32],
        labels:         NDArray[ObjectLabel],
        colormap_value: Optional[str] = None
        ) -> Tuple[NDArray[numpy.int32], NDArray[numpy.int32]]:
    pixel_data[mask] = labels[mask]
    alpha[mask] = 1
    return pixel_data, alpha


################################################################################
# Morphological Operations Helpers
################################################################################

def morphological_gradient(x_data: ObjectSegmentation, structuring_element: StructuringElement) -> ObjectSegmentation:
    is_strel_2d = structuring_element.ndim == 2

    is_img_2d = x_data.ndim == 2

    if is_strel_2d and not is_img_2d:
        y_data = numpy.zeros_like(x_data)

        for index, plane in enumerate(x_data):
            y_data[index] = scipy.ndimage.morphological_gradient(
                plane, footprint=structuring_element
            )

        return y_data

    if not is_strel_2d and is_img_2d:
        raise NotImplementedError(
            "A 3D structuring element cannot be applied to a 2D image."
        )

    y_data = scipy.ndimage.morphological_gradient(x_data, footprint=structuring_element)

    return y_data


################################################################################
# ErodeObjects
################################################################################

def erode_objects_with_structuring_element(
    labels: ObjectSegmentation,
    structuring_element: StructuringElement,
    preserve_midpoints: bool = True,
    relabel_objects: bool = False
) -> ObjectSegmentation:
    """Erode objects based on the structuring element provided.
    
    This function is similar to the "Shrink" function of ExpandOrShrinkObjects,
    with two major distinctions:
    1. ErodeObjects supports 3D objects
    2. An object smaller than the structuring element will be removed entirely
       unless preserve_midpoints is enabled.
    
    Args:
        labels: Input labeled objects array
        structuring_element: Structuring element for erosion operation
        preserve_midpoints: If True, preserve central pixels to prevent object removal
        relabel_objects: If True, assign new label numbers to resulting objects
        
    Returns:
        Eroded objects array with same dimensions as input
    """
    
    
    # Calculate morphological gradient to identify object boundaries
    contours = morphological_gradient(
        labels, structuring_element
    )
    
    # Erode by removing pixels at object boundaries (where contours != 0)
    y_data = labels * (contours == 0)
    
    # Preserve midpoints if requested to prevent object removal
    if preserve_midpoints:
        missing_labels = numpy.setxor1d(labels, y_data)
        
        # Check if structuring element is disk with size 1 (special case optimization)
        # Check based on the actual array properties since we're dealing with numpy array
        is_simple_disk = (
            structuring_element.ndim == 2 and 
            structuring_element.shape == (3, 3) and
            numpy.array_equal(structuring_element, skimage.morphology.disk(1))
        )
        
        if is_simple_disk:
            # For simple disk,1 case, restore missing pixels directly
            y_data += labels * numpy.isin(labels, missing_labels)
        else:
            # For other structuring elements, find and preserve the most central pixels
            for label in missing_labels:
                binary = labels == label
                # Find pixels furthest from the object's edge using distance transform
                midpoint = scipy.ndimage.morphology.distance_transform_edt(binary)
                # Preserve pixels at maximum distance (most central)
                y_data[midpoint == numpy.max(midpoint)] = label
    
    # Relabel objects if requested
    if relabel_objects:
        y_data = skimage.morphology.label(y_data)
    
    return y_data

################################################################################
# DilateObjects
################################################################################

def dilate_objects_with_structuring_element(
    labels: ObjectSegmentation,
    structuring_element: StructuringElement
) -> ObjectSegmentation:
    """Dilate objects based on the structuring element provided.
    
    This function is similar to the "Expand" function of ExpandOrShrinkObjects,
    with two major distinctions:
    1. DilateObjects supports 3D objects, unlike ExpandOrShrinkObjects.
    2. In ExpandOrShrinkObjects, two objects closer than the expansion distance
       will expand until they meet and then stop there. In this module, the object with
       the larger object number (the object that is lower in the image) will be expanded
       on top of the object with the smaller object number.
    
    Args:
        labels: Input labeled objects array
        structuring_element: Structuring element for dilation operation
        
    Returns:
        Dilated objects array with same dimensions as input
    """
    is_strel_2d = structuring_element.ndim == 2

    is_img_2d = labels.ndim == 2

    if is_strel_2d and not is_img_2d:
        y_data = numpy.zeros_like(labels)

        for index, plane in enumerate(labels):

            y_data[index] = skimage.morphology.dilation(plane, structuring_element)

        return y_data

    if not is_strel_2d and is_img_2d:
        raise NotImplementedError(
            "A 3D structuring element cannot be applied to a 2D image."
        )

    y_data = skimage.morphology.dilation(labels, structuring_element)
    
    return y_data


###############################################################################
# ShrinkToObjectCenters
###############################################################################


def find_centroids(
        label_image: ObjectSegmentation
    ) -> ObjectSegmentation:
    input_props = skimage.measure.regionprops(
        label_image, intensity_image=None, cache=True
    )

    input_centroids = [numpy.int_(obj["centroid"]) for obj in input_props]

    output_segmented = numpy.zeros_like(label_image)

    for ind, arr in enumerate(input_centroids):
        output_segmented[tuple(arr)] = ind + 1

    return output_segmented

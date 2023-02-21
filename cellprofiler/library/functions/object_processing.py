from typing import Literal
import warnings
import centrosome.cpmorphology 
import numpy
import scipy.ndimage
import skimage.morphology
import mahotas


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
    return expand(labels,iterations)

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
    input_image,
    watershed_method: Literal["intensity", "distance", "markers"] = "distance",
    declump_method: Literal["shape", "intensity", None] = "shape",
    local_maxima_method: Literal["local", "regional"] = "local",
    intensity_image=None,
    markers_image=None,
    max_seeds: int = -1,
    downsample: int = 1,
    min_distance: int = 1,
    footprint: int = 8,
    connectivity: int = 1,
    compactness: int = 0,
    exclude_border: bool = True,
    watershed_line: bool = False,
    gaussian_sigma: int = 1,
    structuring_element: Literal[
        "ball", "cube", "diamond", "disk", "octahedron", "square", "star"
    ] = "disk",
    structuring_element_size: int = 1,
):
    # Check inputs
    if input_image.dtype != bool:
        raise ValueError(
            "Watershed expects a thresholded image as input. Did you mean to use a boolean array?"
        )

    if (
        watershed_method.casefold() == "intensity"
        or declump_method.casefold() == "intensity"
    ) and intensity_image is None:
        raise ValueError(
            f"Intensity-based methods require an intensity image to be provided"
        )

    if watershed_method.casefold() == "markers" and markers_image is None:
        raise ValueError(
            "Markers watershed method require a markers image to be provided"
        )

    # Create and check structuring element for seed dilation
    strel = getattr(skimage.morphology, structuring_element.casefold())(
        structuring_element_size
    )

    if strel.ndim != input_image.ndim:
        raise ValueError(
            "Structuring element does not match object dimensions: "
            "{} != {}".format(strel.ndim, input_image.ndim)
        )

    if input_image.ndim > 2:
        footprint = numpy.ones((footprint, footprint, footprint))
    else:
        footprint = numpy.ones((footprint, footprint))

    # Downsample input image
    if downsample > 1:
        input_shape = input_image.shape
        if input_image.ndim > 2:
            # Only scale x and y
            factors = (1, downsample, downsample)
        else:
            factors = (downsample, downsample)

        # TODO: Check better methods for ensuring non-bool types or handling the int64
        # output if not
        input_image = skimage.transform.downscale_local_mean(input_image, factors)

    smoothed_input_image = skimage.filters.gaussian(input_image, sigma=gaussian_sigma)

    # Calculate distance transform
    if declump_method.casefold() == "shape":
        # Holes in thresholded objects can negatively impact shape declumping, so fill them
        # Keep the original input_image to use as a mask later (and thus reverse hole-filling)
        input_image_filled = skimage.morphology.remove_small_holes(input_image.astype(bool))
        distance = scipy.ndimage.distance_transform_edt(input_image_filled)
    else:
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
        # No declumping
        watershed_input_image = input_image

    # Generate markers
    if watershed_method.casefold() == "distance":
        # Find maxima in the distance transform
        if local_maxima_method.casefold() == "local":
            seed_coords = skimage.feature.peak_local_max(
                distance,
                min_distance=min_distance,
                footprint=footprint,
                num_peaks=max_seeds if max_seeds != -1 else numpy.inf,
            )
            seeds = numpy.zeros(distance.shape, dtype=bool)
            seeds[tuple(seed_coords.T)] = True
        elif local_maxima_method.casefold() == "regional":
            seeds = mahotas.regmax(distance, footprint)
        else:
            raise NotImplementedError(
                f"local_maxima_method {local_maxima_method} is not supported."
            )

    elif watershed_method.casefold() == "intensity":
        # Find markers based on intensity of the intensity image
        if local_maxima_method.casefold() == "local":
            seed_coords = skimage.feature.peak_local_max(
                intensity_image,
                min_distance=min_distance,
                footprint=footprint,
                num_peaks=max_seeds if max_seeds != -1 else numpy.inf,
            )
            seeds = numpy.zeros(distance.shape, dtype=bool)
            seeds[tuple(seed_coords.T)] = True
        elif local_maxima_method.casefold() == "regional":
            seeds = mahotas.regmax(distance, footprint)
        else:
            raise NotImplementedError(
                f"local_maxima_method {local_maxima_method} is not supported."
            )
    elif watershed_method.casefold() == "markers":
        # The user has provided their own markers
        seeds = markers_image
    else:
        raise NotImplementedError

    # Seed dilation
    seeds = skimage.morphology.binary_dilation(seeds, strel)

    number_objects = skimage.measure.label(seeds, return_num=True)[1]

    seeds_dtype = numpy.uint16 if number_objects < numpy.iinfo(numpy.uint16).max else numpy.uint32

    # Label seeds
    markers = scipy.ndimage.label(seeds)[0]

    # Run watershed
    watershed_image = skimage.segmentation.watershed(
        watershed_input_image,
        markers=markers,
        mask=input_image != 0,
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
        factor = (4.0/3.0) * (radius ** 3)
    
    min_obj_size = numpy.pi * factor

    if planewise and labels.ndim != 2 and labels.shape[-1] not in (3, 4):
        for plane in array:
            for obj in numpy.unique(plane):
                if obj == 0:
                    continue
                filled_mask = skimage.morphology.remove_small_holes(plane == obj, min_obj_size)
                plane[filled_mask] = obj    
        return array
    else:
        for obj in numpy.unique(array):
            if obj == 0:
                continue
            filled_mask = skimage.morphology.remove_small_holes(array == obj, min_obj_size)
            array[filled_mask] = obj
    return array

def fill_convex_hulls(labels):
    data = skimage.measure.regionprops(labels)
    output = numpy.zeros_like(labels)
    for prop in data:
        label = prop['label']
        bbox = prop['bbox']
        cmask = prop['convex_image']
        if len(bbox) <= 4:
            output[bbox[0]:bbox[2], bbox[1]:bbox[3]][cmask] = label
        else:
            output[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]: bbox[5]][cmask] = label
    return output

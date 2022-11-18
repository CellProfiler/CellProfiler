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

def watershed_distance(
    input_image,
    footprint=8,
    downsample=1,
):
    """
    Returns a simple watershed based on distance.
    Equivalent to CellProfiler GUI with advanced
    settings off.
    """    

    if downsample > 1:
        # Check if volumetric
        if input_image.ndim > 2:
            factors = (1, downsample, downsample)
        else:
            factors = (downsample, downsample)

        input_image = skimage.transform.downscale_local_mean(input_image, factors)

    threshold = skimage.filters.threshold_otsu(input_image)

    input_image = input_image > threshold

    distance = scipy.ndimage.distance_transform_edt(input_image)

    distance = mahotas.stretch(distance)

    surface = distance.max() - distance

    if input_image.ndim > 2:
        footprint = numpy.ones((footprint, footprint, footprint))
    else:
        footprint = numpy.ones((footprint, footprint))

    peaks = mahotas.regmax(distance, footprint)

    if input_image.ndim > 2:
        markers, _ = mahotas.label(peaks, numpy.ones((16, 16, 16)))
    else:
        markers, _ = mahotas.label(peaks, numpy.ones((16, 16)))

    y_data = mahotas.cwatershed(surface, markers)

    y_data = y_data * input_image

    if downsample > 1:
        y_data = skimage.transform.resize(
            y_data, input_image.shape, mode="edge", order=0, preserve_range=True
        )

        y_data = numpy.rint(y_data).astype(numpy.uint16)

    return y_data


def watershed_markers(
    input_image, markers, mask=None, connectivity=1, compactness=0, watershed_line=False
):
    """
    Returns a simple watershed based on markers.
    Equivalent to CellProfiler GUI with advanced
    settings off.
    """
    y_data = skimage.segmentation.watershed(
        image=input_image,
        markers=markers,
        mask=mask,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
    )

    return y_data


def watershed_advanced(
    input_image,
    markers=None,
    mask=None,
    intensity_image=None,
    method="distance",
    declump_method="shape",
    footprint=8,
    downsample=1,
    connectivity=1,
    compactness=0,
    watershed_line=False,
    structuring_element="disk",
    structuring_element_size=1,
    gaussian_sigma=1,
    min_distance=1,
    min_intensity=0,
    exclude_border=0,
    max_seeds=-1,
):

    if method.casefold() == "distance":
        y_data = watershed_distance(
            input_image,
            footprint=footprint,
            downsample=downsample,
        )
    elif method.casefold() == "markers":
        y_data = watershed_markers(
            input_image,
            markers=markers,
            mask=mask,
            connectivity=connectivity,
            compactness=compactness,
            watershed_line=watershed_line,
        )
    else:
        raise ValueError(f"Watershed method {method} does not exist")

    # Advanced watershed
    strel = getattr(skimage.morphology, structuring_element.casefold())(
        structuring_element_size
    )
    if strel.ndim != input_image.ndim:
        raise ValueError(
            "Structuring element does not match object dimensions: "
            "{} != {}".format(strel.ndim, input_image.ndim)
        )

    # Get the segmentation distance transform for the watershed segmentation
    peak_image = scipy.ndimage.distance_transform_edt(y_data > 0)

    if declump_method.casefold() == "shape":
        watershed_image = -peak_image
        watershed_image -= watershed_image.min()
    if declump_method.casefold() == "intensity":
        if intensity_image is None:
            raise ValueError(
                """An intensity reference image is required 
                                to perform intensity-based declumping"""
            )
        # Set the image as a float and rescale to full bit depth
        watershed_image = skimage.img_as_float(intensity_image, force_copy=True)
        watershed_image -= watershed_image.min()
        watershed_image = 1 - watershed_image

    # Smooth the image
    watershed_image = skimage.filters.gaussian(watershed_image, sigma=gaussian_sigma)
    # Generate local peaks; returns a list of coords for each peak
    seed_coords = skimage.feature.peak_local_max(
        peak_image,
        min_distance=min_distance,
        threshold_rel=min_intensity,
        exclude_border=exclude_border,
        num_peaks=max_seeds if max_seeds != -1 else numpy.inf,
    )

    # generate an array w/ same dimensions as the original image with all elements having value False
    seeds = numpy.zeros_like(peak_image, dtype=bool)

    # set value to True at every local peak
    seeds[tuple(seed_coords.T)] = True

    # Dilate seeds based on settings
    seeds = skimage.morphology.binary_dilation(seeds, strel)

    # get the number of objects from the distance-based or marker-based watershed run above
    number_objects = skimage.measure.label(y_data, return_num=True)[1]

    seeds_dtype = (
        numpy.uint16 if number_objects < numpy.iinfo(numpy.uint16).max else numpy.uint32
    )

    # NOTE: Not my work, the comments below are courtesy of Ray
    #
    # Create a marker array where the unlabeled image has a label of
    # -(nobjects+1)
    # and every local maximum has a unique label which will become
    # the object's label. The labels are negative because that
    # makes the watershed algorithm use FIFO for the pixels which
    # yields fair boundaries when markers compete for pixels.
    #
    seeds = scipy.ndimage.label(seeds)[0]

    markers = numpy.zeros_like(seeds, dtype=seeds_dtype)
    markers[seeds > 0] = -seeds[seeds > 0]

    # Perform the watershed
    watershed_boundaries = skimage.segmentation.watershed(
        connectivity=connectivity,
        image=watershed_image,
        markers=markers,
        mask=input_image != 0,
    )

    y_data = watershed_boundaries.copy()
    # Copy the location of the "background"
    zeros = numpy.where(y_data == 0)
    # Re-shift all of the labels into the positive realm
    y_data += numpy.abs(numpy.min(y_data)) + 1
    # Re-apply the background
    y_data[zeros] = 0

    return y_data

import centrosome.cpmorphology 
import numpy
import scipy.ndimage
import skimage.morphology

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

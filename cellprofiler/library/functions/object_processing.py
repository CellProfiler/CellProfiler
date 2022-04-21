import centrosome.cpmorphology 
import numpy
import scipy.ndimage

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
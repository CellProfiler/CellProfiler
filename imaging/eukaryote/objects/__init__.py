"""
CellProfiler.Objects.py - represents a labelling of objects in an image
"""

import decorator
import numpy as np

OBJECT_TYPE_NAME = "objects"


@decorator.decorator
def memoize_method(function, *args):
    """Cache the result of a method in that class's dictionary
    
    The dictionary is indexed by function name and the values of that
    dictionary are themselves dictionaries with args[1:] as the keys
    and the result of applying function to args[1:] as the values.
    """
    sself = args[0]
    d = getattr(sself, "memoize_method_dictionary", False)
    if not d:
        d = {}
        setattr(sself, "memoize_method_dictionary", d)
    if not d.has_key(function):
        d[function] = {}
    if not d[function].has_key(args[1:]):
        d[function][args[1:]] = function(*args)
    return d[function][args[1:]]


def check_consistency(segmented, unedited_segmented, small_removed_segmented):
    """Check the three components of Objects to make sure they are consistent
    """
    assert segmented is None or np.all(segmented >= 0)
    assert unedited_segmented is None or np.all(unedited_segmented >= 0)
    assert small_removed_segmented is None or np.all(small_removed_segmented >= 0)
    assert segmented is None or segmented.ndim == 2, "Segmented label matrix must have two dimensions, has %d" % (
    segmented.ndim)
    assert unedited_segmented is None or unedited_segmented.ndim == 2, "Unedited segmented label matrix must have two dimensions, has %d" % (
    unedited_segmented.ndim)
    assert small_removed_segmented is None or small_removed_segmented.ndim == 2, "Small removed segmented label matrix must have two dimensions, has %d" % (
    small_removed_segmented.ndim)
    assert segmented is None or unedited_segmented is None or segmented.shape == unedited_segmented.shape, "Segmented %s and unedited segmented %s shapes differ" % (
    repr(segmented.shape), repr(unedited_segmented.shape))
    assert segmented is None or small_removed_segmented is None or segmented.shape == small_removed_segmented.shape, "Segmented %s and small removed segmented %s shapes differ" % (
    repr(segmented.shape), repr(small_removed_segmented.shape))


def downsample_labels(labels):
    '''Convert a labels matrix to the smallest possible integer format'''
    labels_max = np.max(labels)
    if labels_max < 128:
        return labels.astype(np.int8)
    elif labels_max < 32768:
        return labels.astype(np.int16)
    return labels.astype(np.int32)


def crop_labels_and_image(labels, image):
    '''Crop a labels matrix and an image to the lowest common size
    
    labels - a n x m labels matrix
    image - a 2-d or 3-d image
    
    Assumes that points outside of the common boundary should be masked.
    '''
    min_height = min(labels.shape[0], image.shape[0])
    min_width = min(labels.shape[1], image.shape[1])
    if image.ndim == 2:
        return (labels[:min_height, :min_width],
                image[:min_height, :min_width])
    else:
        return (labels[:min_height, :min_width],
                image[:min_height, :min_width, :])


def size_similarly(labels, secondary):
    '''Size the secondary matrix similarly to the labels matrix
    
    labels - labels matrix
    secondary - a secondary image or labels matrix which might be of
                different size.
    Return the resized secondary matrix and a mask indicating what portion
    of the secondary matrix is bogus (manufactured values).
    
    Either the mask is all ones or the result is a copy, so you can
    modify the output within the unmasked region w/o destroying the original.
    '''
    if labels.shape[:2] == secondary.shape[:2]:
        return secondary, np.ones(secondary.shape, bool)
    if (labels.shape[0] <= secondary.shape[0] and
                labels.shape[1] <= secondary.shape[1]):
        if secondary.ndim == 2:
            return (secondary[:labels.shape[0], :labels.shape[1]],
                    np.ones(labels.shape, bool))
        else:
            return (secondary[:labels.shape[0], :labels.shape[1], :],
                    np.ones(labels.shape, bool))

    #
    # Some portion of the secondary matrix does not cover the labels
    #
    result = np.zeros(list(labels.shape) + list(secondary.shape[2:]),
                      secondary.dtype)
    i_max = min(secondary.shape[0], labels.shape[0])
    j_max = min(secondary.shape[1], labels.shape[1])
    if secondary.ndim == 2:
        result[:i_max, :j_max] = secondary[:i_max, :j_max]
    else:
        result[:i_max, :j_max, :] = secondary[:i_max, :j_max, :]
    mask = np.zeros(labels.shape, bool)
    mask[:i_max, :j_max] = 1
    return result, mask

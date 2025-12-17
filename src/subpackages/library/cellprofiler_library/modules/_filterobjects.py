import numpy
import scipy
from cellprofiler_library.opts.filterobjects import FilterMethod, OverlapAssignment

def keep_one(values, filter_choice):
    """Return an array containing the single object to keep

    workspace - workspace passed into Run
    src_objects - the Objects instance to be filtered
    """
    if len(values) == 0:
        return numpy.array([], int)
    best_idx = (
        numpy.argmax(values)
        if filter_choice == FilterMethod.MAXIMAL.value
        else numpy.argmin(values)
    ) + 1
    return numpy.array([best_idx], int)


def keep_per_object(src_labels, enclosing_labels, enclosing_max, per_object_assignment, filter_choice, values):
    """Return an array containing the best object per enclosing object

    workspace - workspace passed into Run
    src_objects - the Objects instance to be filtered
    """
    if enclosing_max == 0:
        return numpy.array([], int)
    enclosing_range = numpy.arange(1, enclosing_max + 1)

    #
    # Make a vector of the value of the measurement per label index.
    # We can then label each pixel in the image with the measurement
    # value for the object at that pixel.
    # For unlabeled pixels, put the minimum value if looking for the
    # maximum value and vice-versa
    #
    wants_max = filter_choice == FilterMethod.MAXIMAL_PER_OBJECT.value
    if per_object_assignment == OverlapAssignment.PARENT_WITH_MOST_OVERLAP.value:
        #
        # Find the number of overlapping pixels in enclosing
        # and source objects
        #
        mask = enclosing_labels * src_labels != 0
        enclosing_labels = enclosing_labels[mask]
        src_labels = src_labels[mask]
        order = numpy.lexsort((enclosing_labels, src_labels))
        src_labels = src_labels[order]
        enclosing_labels = enclosing_labels[order]
        firsts = numpy.hstack(
            (
                [0],
                numpy.where(
                    (src_labels[:-1] != src_labels[1:])
                    | (enclosing_labels[:-1] != enclosing_labels[1:])
                )[0]
                + 1,
                [len(src_labels)],
            )
        )
        areas = firsts[1:] - firsts[:-1]
        enclosing_labels = enclosing_labels[firsts[:-1]]
        src_labels = src_labels[firsts[:-1]]
        #
        # Re-sort by source label value and area descending
        #
        if wants_max:
            svalues = -values
        else:
            svalues = values
        order = numpy.lexsort((-areas, svalues[src_labels - 1]))
        src_labels, enclosing_labels, areas = [
            x[order] for x in (src_labels, enclosing_labels, areas)
        ]
        firsts = numpy.hstack(
            (
                [0],
                numpy.where(src_labels[:-1] != src_labels[1:])[0] + 1,
                src_labels.shape[:1],
            )
        )
        counts = firsts[1:] - firsts[:-1]
        #
        # Process them in order. The maximal or minimal child
        # will be assigned to the most overlapping parent and that
        # parent will be excluded.
        #
        best_src_label = numpy.zeros(enclosing_max + 1, int)
        for idx, count in zip(firsts[:-1], counts):
            for i in range(count):
                enclosing_object_number = enclosing_labels[idx + i]
                if best_src_label[enclosing_object_number] == 0:
                    best_src_label[enclosing_object_number] = src_labels[idx]
                    break
        #
        # Remove best source labels = 0 and sort to get the list
        #
        best_src_label = best_src_label[best_src_label != 0]
        best_src_label.sort()
        return best_src_label
    else:
        tricky_values = numpy.zeros((len(values) + 1,))
        tricky_values[1:] = values
        if wants_max:
            tricky_values[0] = -numpy.Inf
        else:
            tricky_values[0] = numpy.Inf
        src_values = tricky_values[src_labels]
        #
        # Now find the location of the best for each of the enclosing objects
        #
        fn = (
            scipy.ndimage.maximum_position
            if wants_max
            else scipy.ndimage.minimum_position
        )
        best_pos = fn(src_values, enclosing_labels, enclosing_range)
        best_pos = numpy.array(
            (best_pos,) if isinstance(best_pos, tuple) else best_pos
        )
        best_pos = best_pos.astype(numpy.uint32)
        #
        # Get the label of the pixel at each location
        #
        # Multidimensional indexing with non-tuple values is not allowed as of numpy 1.23
        best_pos = tuple(map(tuple, best_pos.transpose()))
        indexes = src_labels[best_pos]
        indexes = set(indexes)
        indexes = list(indexes)
        indexes.sort()
        return indexes[1:] if len(indexes) > 0 and indexes[0] == 0 else indexes


def keep_within_limits(limit_groups):
    """Return an array containing the indices of objects to keep

    workspace - workspace passed into Run
    src_objects - the Objects instance to be filtered
    """
    hits = None
    MIN_LIM = "min_limit"
    MAX_LIM = "max_limit"
    VALUES = "values"
    for group in limit_groups:
        values = group[VALUES]

        if hits is None:
            hits = numpy.ones(len(values), bool)
        elif len(hits) < len(values):
            temp = numpy.ones(len(values), bool)
            temp[~hits] = False
            hits = temp
        low_limit = group[MIN_LIM]
        high_limit = group[MAX_LIM]
        if low_limit is not None:
            hits[values < low_limit] = False
        if high_limit is not None:
            hits[values > high_limit] = False
    assert hits is not None
    indexes = numpy.argwhere(hits)[:, 0]
    indexes = indexes + 1
    return indexes


def discard_border_objects(labels, parent_image_mask):
    """Return an array containing the indices of objects to keep
    """

    if parent_image_mask is not None:
        mask = parent_image_mask
        interior_pixels = scipy.ndimage.binary_erosion(mask)

    else:
        interior_pixels = scipy.ndimage.binary_erosion(numpy.ones_like(labels))

    border_pixels = numpy.logical_not(interior_pixels)
    border_labels = set(labels[border_pixels])
    if (border_labels == {0} and parent_image_mask):
        # The assumption here is that, if nothing touches the border,
        # the mask is a large, elliptical mask that tells you where the
        # well is. That's the way the old Matlab code works and it's duplicated here
        #
        # The operation below gets the mask pixels that are on the border of the mask
        # The erosion turns all pixels touching an edge to zero. The not of this
        # is the border + formerly masked-out pixels.

        mask = parent_image_mask
        interior_pixels = scipy.ndimage.binary_erosion(mask)
        border_pixels = numpy.logical_not(interior_pixels)
        border_labels = set(labels[border_pixels])

    return list(set(labels.ravel()).difference(border_labels))


def get_filtered_object(
        src_objects_segmented, 
        indexes, 
        label_indexes,
        max_label,
        parent_objects,
        keep_unassociated_objects
    ):
    
    if label_indexes is None:   
        new_object_count = len(indexes)
        label_indexes = numpy.zeros((max_label + 1,), int)
        label_indexes[indexes] = numpy.arange(1, new_object_count + 1)
    
    #
    # Reindex the labels of the old source image
    #
    target_objects_segmented = reindex_labels(src_objects_segmented, max_label, label_indexes, parent_objects, keep_unassociated_objects)

    return target_objects_segmented

def reindex_labels(
        src_objects_segmented, 
        max_label, 
        label_indexes, 
        parent_objects, 
        keep_unassociated_objects
    ):        
    target_labels = src_objects_segmented.copy()
    if parent_objects is None:
        target_labels[target_labels > max_label] = 0
        target_labels = label_indexes[target_labels]
    else:
        # Initialize target labels to keep all child objects
        target_label_numbers = numpy.arange(1, target_labels.max() + 1)
        
        orphan_children = target_label_numbers[parent_objects == 0]

        # label == 0 indicates parent object has to be removed
        objects_to_remove = numpy.arange(max_label+1)[label_indexes == 0][1:] # ignore the first zero as it is the background
        
        # object is removed by setting its new label to zero
        target_label_numbers = target_label_numbers*~numpy.isin(parent_objects, objects_to_remove)

        new_child_object_count = sum(target_label_numbers != 0)

        # orphan children get new labels. Labels are always continuous and start at 1
        target_label_numbers[target_label_numbers != 0] = numpy.arange(1, new_child_object_count + 1)
        
        # Add zero for background label
        target_label_numbers = numpy.pad(target_label_numbers, (1, 0))
        
        # Overwrite orphan children new labels with 0 to remove unassociated objects
        if not keep_unassociated_objects:
            target_label_numbers[orphan_children] = 0

        # Numpy fancy indexing to relabel
        target_labels = target_label_numbers[target_labels]
    return target_labels



def get_removed_objects(
        indexes,
        max_label,
        src_objects_segmented,
    ):
    removed_labels = src_objects_segmented.copy()
    # Isolate objects removed by the filter
    removed_indexes = [x for x in range(1, max_label+1) if x not in indexes]
    removed_object_count = len(removed_indexes)
    removed_label_indexes = numpy.zeros((max_label + 1,), int)
    removed_label_indexes[removed_indexes] = numpy.arange(1, removed_object_count + 1)

    #
    # Reindex the labels of the old source image
    #
    removed_labels[removed_labels > max_label] = 0
    removed_labels = removed_label_indexes[removed_labels]

    return removed_labels

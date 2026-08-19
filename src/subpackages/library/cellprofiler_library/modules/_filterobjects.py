import numpy
import scipy
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

from numpy.typing import NDArray
from pydantic import ConfigDict, Field, validate_call

from cellprofiler_library.opts.filterobjects import FilterMethod, FilterMode, OverlapAssignment
from cellprofiler_library.types import ObjectSegmentation

def keep_one(values: NDArray[numpy.float64], filter_choice: FilterMethod) -> NDArray[numpy.int_]:
    """
    Return an array containing the single object to keep

    values - measurement value per object
    filter_choice - FilterMethod.MINIMAL or FilterMethod.MAXIMAL
    """
    if len(values) == 0:
        return numpy.array([], int)
    best_idx = (
        numpy.argmax(values)
        if filter_choice == FilterMethod.MAXIMAL.value
        else numpy.argmin(values)
    ) + 1
    return numpy.array([best_idx], int)


def keep_per_object(
    src_labels: ObjectSegmentation,
    enclosing_labels: ObjectSegmentation,
    enclosing_max: int,
    per_object_assignment: OverlapAssignment,
    filter_choice: FilterMethod,
    values: NDArray[numpy.float64],
) -> Union[NDArray[numpy.int_], List[int]]:
    """
    Return an array containing the best object per enclosing object

    src_labels - segmentation of the objects being filtered
    enclosing_labels - segmentation of the enclosing (parent) objects
    enclosing_max - number of enclosing objects
    per_object_assignment - OverlapAssignment strategy for matching objects to enclosing objects
    filter_choice - FilterMethod.MINIMAL_PER_OBJECT or FilterMethod.MAXIMAL_PER_OBJECT
    values - measurement value per object
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


def keep_within_limits(limit_groups: List[Dict[str, Any]]) -> NDArray[numpy.int_]:
    """Return an array containing the indices of objects to keep

    limit_groups - a list of {"values": ndarray, "min_limit": float or None, "max_limit": float or None}
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


def keep_by_rules(scores: NDArray[numpy.float64], rules_class: int) -> NDArray[numpy.int_]:
    """Return the indexes (base 1) of objects whose highest-scoring class is rules_class

    scores - an MxN matrix as returned by Rules.score(): M objects x N classes.
             The Rules object itself is never passed into the library, only
             the plain scores it produces.
    rules_class - the 0-based class index to keep
    """
    if len(scores) == 0:
        return numpy.array([], int)
    is_not_nan = numpy.any(~numpy.isnan(scores), 1)
    best_class = numpy.argmax(scores[is_not_nan], 1).flatten()
    hits = numpy.zeros(scores.shape[0], bool)
    hits[is_not_nan] = best_class == rules_class
    return numpy.argwhere(hits).flatten() + 1


def keep_by_hits(hits: NDArray[numpy.bool_]) -> NDArray[numpy.int_]:
    """Return the indexes (base 1) of objects for which hits is True

    Used for classifier predictions (predicted_classes == target_class) -
    the classifier object itself is never passed into the library, only the
    resulting boolean hits it produces.
    """
    return numpy.argwhere(hits).flatten() + 1


def discard_border_objects(labels: ObjectSegmentation, parent_image_mask: Optional[NDArray[numpy.bool_]]) -> List[int]:
    """
    Return an array containing the object numbers to keep

    labels - segmentation of the objects being filtered
    parent_image_mask - mask of the parent image (or None); objects touching its border are discarded
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
        src_objects_segmented: ObjectSegmentation,
        indexes: Union[NDArray[numpy.int_], List[int]],
        label_indexes: Optional[NDArray[numpy.int_]],
        max_label: int,
        parent_objects: Optional[NDArray[numpy.int_]],
        keep_unassociated_objects: bool,
    ) -> ObjectSegmentation:
    """
    Relabel a segmentation so it keeps only the filtered objects

    src_objects_segmented - segmentation to filter and relabel
    indexes - object numbers (base 1) to keep
    label_indexes - mapping from old label to new label, or None to build it from indexes
    max_label - highest label value in src_objects_segmented
    parent_objects - parent object number per object, or None if unrelated to a parent
    keep_unassociated_objects - whether to keep objects that have no parent
    """
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
        src_objects_segmented: ObjectSegmentation,
        max_label: int,
        label_indexes: NDArray[numpy.int_],
        parent_objects: Optional[NDArray[numpy.int_]],
        keep_unassociated_objects: bool,
    ) -> ObjectSegmentation:
    """
    Reindex a segmentation, dropping objects whose new label is 0

    src_objects_segmented - segmentation to relabel
    max_label - highest label value of the filtered object
    label_indexes - mapping from old label to new label (0 removes the object)
    parent_objects - parent object number per object, or None if unrelated to a parent
    keep_unassociated_objects - whether to keep objects that have no parent
    """
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
        indexes: Union[NDArray[numpy.int_], List[int]],
        max_label: int,
        src_objects_segmented: ObjectSegmentation,
    ) -> ObjectSegmentation:
    """
    Return a segmentation containing only the objects removed by the filter

    indexes - object numbers (base 1) that were kept
    max_label - highest label value in src_objects_segmented
    src_objects_segmented - the original, unfiltered segmentation
    """
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


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def filter_objects(
    src_labels: Annotated[ObjectSegmentation, Field(description="Segmentation of the object being filtered")],
    mode: Annotated[FilterMode, Field(description="Which filtering mode is active")],
    keep_removed_objects: Annotated[bool, Field(description="Also compute and return the objects removed by the filter")] = False,
    additional_objects: Annotated[
        List[Tuple[ObjectSegmentation, Optional[NDArray[numpy.int_]], bool]],
        Field(description="Additional objects to relabel to match the filtered object, as (labels, parent_objects, keep_unassociated_objects) tuples"),
    ] = [],

    # mode == Measurements
    filter_choice: Annotated[Optional[FilterMethod], Field(description="Used only if mode is Measurements: which measurement-filtering method")] = None,
    values: Annotated[Optional[NDArray[numpy.float64]], Field(description="Used only for Minimal/Maximal or per-object filtering: measurement value per object")] = None,
    limit_groups: Annotated[Optional[List[Dict[str, Any]]], Field(description="Used only for Limits filtering: list of {'values', 'min_limit', 'max_limit'}")] = None,
    enclosing_labels: Annotated[Optional[ObjectSegmentation], Field(description="Used only for per-object filtering: enclosing/parent object segmentation")] = None,
    enclosing_count: Annotated[Optional[int], Field(description="Used only for per-object filtering: number of enclosing objects")] = None,
    per_object_assignment: Annotated[Optional[OverlapAssignment], Field(description="Used only for per-object filtering")] = None,

    # mode == Border
    parent_image_mask: Annotated[Optional[NDArray[numpy.bool_]], Field(description="Used only if mode is Border: parent image mask")] = None,

    # mode in (Rules, Classifiers)
    scores: Annotated[Optional[NDArray[numpy.float64]], Field(description="Used only for rules-based filtering: per-object x per-class scores from Rules.score()")] = None,
    rules_class: Annotated[Optional[int], Field(description="Used only for rules-based filtering: 0-based class index to keep")] = None,
    hits: Annotated[Optional[NDArray[numpy.bool_]], Field(description="Used only for classifier-prediction filtering: precomputed pass/fail per object")] = None,
) -> Tuple[ObjectSegmentation, List[ObjectSegmentation], Optional[ObjectSegmentation]]:
    max_label = int(numpy.max(src_labels))

    if mode == FilterMode.MEASUREMENTS.value:
        if filter_choice in (FilterMethod.MINIMAL.value, FilterMethod.MAXIMAL.value):
            indexes = keep_one(values, filter_choice)
        elif filter_choice in (FilterMethod.MINIMAL_PER_OBJECT.value, FilterMethod.MAXIMAL_PER_OBJECT.value):
            indexes = keep_per_object(
                src_labels, enclosing_labels, enclosing_count, per_object_assignment, filter_choice, values,
            )
        elif filter_choice == FilterMethod.LIMITS.value:
            indexes = keep_within_limits(limit_groups)
        else:
            raise ValueError(f"Unknown filter choice: {filter_choice} for mode {mode}")
    elif mode == FilterMode.BORDER.value:
        indexes = discard_border_objects(src_labels, parent_image_mask)
    # keep_by_class
    elif mode in (FilterMode.RULES.value, FilterMode.CLASSIFIERS.value):
        if scores is not None:
            indexes = keep_by_rules(scores, rules_class)
        elif hits is not None:
            indexes = keep_by_hits(hits)
        else:
            raise ValueError(f"mode {mode} requires either 'scores' or 'hits'")
    else:
        raise ValueError(f"Unknown filter mode: {mode}")

    new_object_count = len(indexes)
    label_indexes = numpy.zeros((max_label + 1,), int)
    label_indexes[indexes] = numpy.arange(1, new_object_count + 1)

    target_segmented = get_filtered_object(src_labels, indexes, label_indexes, max_label, None, False)

    additional_segmented = [
        get_filtered_object(labels, indexes, label_indexes, max_label, parent_objects, keep_unassociated_objects)
        for labels, parent_objects, keep_unassociated_objects in additional_objects
    ]

    removed_segmented = (
        get_removed_objects(indexes, max_label, src_labels) if keep_removed_objects else None
    )

    return target_segmented, additional_segmented, removed_segmented

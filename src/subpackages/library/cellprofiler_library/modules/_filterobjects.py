import numpy
from typing import Annotated, Any, Dict, List, Optional, Tuple

from numpy.typing import NDArray
from pydantic import ConfigDict, Field, validate_call

from cellprofiler_library.opts.filterobjects import FilterMethod, FilterMode, OverlapAssignment
from cellprofiler_library.types import ObjectSegmentation
from cellprofiler_library.functions.object_processing import (
    keep_one,
    keep_per_object,
    keep_within_limits,
    discard_border_objects,
    keep_by_rules,
    keep_by_hits,
    get_filtered_object,
    get_removed_objects
)


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

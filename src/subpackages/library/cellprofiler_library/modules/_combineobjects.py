from ..functions.object_processing import (
    merge_objects, preserve_objects, discard_objects, segment_objects
)

def combineobjects(method, labels_x, labels_y, dimensions):
    assert (
    dimensions in (2, 3)
    ), f"Only dimensions of 2 or 3 are supported, got {dimensions}"

    assert (
        method.casefold() in ("merge", "preserve", "discard", "segment")
    ), f"Method {method} not in 'merge', 'preserve', 'discard', or 'segment'"

    if method.casefold() == "merge":
        return merge_objects(labels_x, labels_y, dimensions)
    if method.casefold() == "preserve":
        return preserve_objects(labels_x, labels_y)
    if method.casefold() == "discard":
        return discard_objects(labels_x, labels_y)
    if method.casefold() == "segment":
        return segment_objects(labels_x, labels_y, dimensions)
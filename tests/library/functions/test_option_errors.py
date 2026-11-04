import numpy
import pytest

from cellprofiler_library.functions.image_processing import (
    get_adaptive_threshold,
    get_global_threshold,
    imagemath_apply_binary_operation,
)
from cellprofiler_library.modules._filterobjects import filter_objects
from cellprofiler_library.modules._splitormergeobjects import split_or_merge_objects
from cellprofiler_library.opts.filterobjects import FilterMode
from cellprofiler_library.opts.imagemath import Operator
from cellprofiler_library.opts.splitormergeobjects import MergingMethod, RelabelOption
from cellprofiler_library.opts.threshold import Method


@pytest.mark.parametrize("threshold", [get_global_threshold, get_adaptive_threshold])
@pytest.mark.parametrize(
    "method, message",
    [
        (Method.MANUAL, "Threshold method Manual not supported."),
        (Method.MEASUREMENT, "Threshold method Measurement not supported."),
        ("Unknown", "Threshold method Unknown not supported."),
    ],
)
def test_unsupported_threshold_message(threshold, method, message):
    image = numpy.arange(16).reshape(4, 4) / 16
    with pytest.raises(NotImplementedError) as error:
        threshold(image, threshold_method=method)
    assert str(error.value) == message


@pytest.mark.parametrize(
    "mode, message",
    [
        (FilterMode.RULES, "mode Rules requires either 'scores' or 'hits'"),
        (FilterMode.CLASSIFIERS, "mode Classifiers requires either 'scores' or 'hits'"),
        (FilterMode.MEASUREMENTS, "Unknown filter choice: None for mode Measurements"),
    ],
)
def test_missing_filter_inputs_message(mode, message):
    with pytest.raises(ValueError) as error:
        filter_objects(numpy.ones((4, 4), dtype=int), mode=mode)
    assert str(error.value) == message


@pytest.mark.parametrize(
    "operation, message",
    [
        (Operator.STDEV, "Unimplemented operation: Standard Deviation"),
        (Operator.INVERT, "Unimplemented operation: Invert"),
        (Operator.COMPLEMENT, "Unimplemented operation: Complement"),
        (Operator.LOG_TRANSFORM_LEGACY, "Unimplemented operation: Log transform (legacy)"),
        (Operator.LOG_TRANSFORM, "Unimplemented operation: Log transform (base 2)"),
        (Operator.NOT, "Unimplemented operation: Not"),
        ("Unknown", "Unimplemented operation: Unknown"),
    ],
)
def test_unsupported_binary_operation_message(operation, message):
    image = numpy.ones((4, 4))
    with pytest.raises(NotImplementedError) as error:
        imagemath_apply_binary_operation(
            operation, [image], None, image, None, None, False
        )
    assert str(error.value) == message


@pytest.mark.parametrize(
    "method, message",
    [
        (MergingMethod.DISCONNECTED, "Unimplemented merging method: Disconnected"),
        (MergingMethod.CONVEX_HULL, "Unimplemented merging method: Convex hull"),
        (None, "Unimplemented merging method: None"),
    ],
)
def test_missing_merge_option_message(method, message):
    with pytest.raises(NotImplementedError) as error:
        split_or_merge_objects(
            labels=numpy.ones((4, 4), dtype=int),
            relabel_option=RelabelOption.MERGE,
            objects_name="Objects",
            objects_small_removed_segmented=None,
            objects_unedited_segmented=None,
            parent_name=None,
            merge_option=None,
            merging_method=method,
            distance_threshold=None,
            image=None,
            parents_of=None,
            merge_condition=None,
            minimum_intensity_fraction=None,
            output_objects_name=None,
            output_object_volumetric=None,
            labels_ijv=None,
        )
    assert str(error.value) == message

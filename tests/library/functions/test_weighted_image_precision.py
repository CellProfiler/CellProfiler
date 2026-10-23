import numpy
import pytest

from cellprofiler_library.functions.image_processing import (
    combine_colortogray,
    imagemath_apply_binary_operation,
)
from cellprofiler_library.opts.imagemath import Operator


@pytest.mark.parametrize(
    "weights, expected",
    [
        ([0.1, 0.2, 0.3], 2.833333333333333),
        ([0.3, 0.6, 0.9], 2.833333333333334),
        ([0.299, 0.587, 0.114], 1.9290000000000003),
    ],
)
def test_colortogray_preserves_weighted_pixels(weights, expected):
    image = numpy.broadcast_to(numpy.array([1.0, 2.0, 4.0]), (2, 3, 3)).copy()
    result = combine_colortogray(image, [0, 1, 2], weights)
    expected_image = numpy.full((2, 3), expected, dtype=numpy.float64)
    assert result.dtype == expected_image.dtype
    assert result.shape == expected_image.shape
    assert result.tobytes() == expected_image.tobytes()


@pytest.mark.parametrize(
    "weights, expected",
    [
        ([0.1, 0.2, 0.3], 2.833333333333333),
        ([0.3, 0.6, 0.9], 2.8333333333333335),
        ([0.299, 0.587, 0.114], 1.929),
    ],
)
def test_imagemath_preserves_weighted_average_pixels(weights, expected):
    operands = [
        numpy.full((2, 3), value * weight, dtype=numpy.float64)
        for value, weight in zip([1.0, 2.0, 4.0], weights)
    ]
    result, mask = imagemath_apply_binary_operation(
        Operator.AVERAGE, operands, None, operands[0].copy(), None, weights, False
    )
    expected_image = numpy.full((2, 3), expected, dtype=numpy.float64)
    assert result.dtype == expected_image.dtype
    assert result.shape == expected_image.shape
    assert result.tobytes() == expected_image.tobytes()
    assert mask is None

import inspect

import numpy
import pytest
import skimage.morphology

from cellprofiler_library.functions.morphology import (
    binary_dilation,
    binary_erosion,
    get_footprint,
    remove_small_holes,
)


def test_binary_dilation_asymmetric_even_footprint():
    image = numpy.zeros((5, 5), numpy.uint8)
    image[2, 2] = 7
    footprint = numpy.array([[1, 0], [1, 1]], bool)
    expected = numpy.zeros((5, 5), bool)
    expected[1, 1] = expected[2, 1] = expected[2, 2] = True

    actual = binary_dilation(image, footprint)

    numpy.testing.assert_array_equal(actual, expected)
    assert actual.dtype == bool
    assert image[2, 2] == 7


def test_binary_erosion_asymmetric_even_footprint():
    image = numpy.ones((5, 5), bool)
    image[2, 2] = False
    footprint = numpy.array([[1, 0], [1, 1]], bool)
    expected = numpy.ones((5, 5), bool)
    expected[3, 3] = expected[2, 3] = expected[2, 2] = False

    numpy.testing.assert_array_equal(binary_erosion(image, footprint), expected)


@pytest.mark.parametrize("threshold", [3.9, 4, 4.1])
def test_remove_small_holes_native_area_threshold(threshold):
    image = numpy.ones((7, 7), bool)
    image[2:4, 2:4] = False

    actual = remove_small_holes(image, threshold)

    expected = image.copy()
    if "max_size" in inspect.signature(skimage.morphology.remove_small_holes).parameters:
        filled = 4 <= threshold
    else:
        filled = 4 < threshold
    expected[2:4, 2:4] = filled
    numpy.testing.assert_array_equal(actual, expected)
    assert not image[2, 2]


@pytest.mark.parametrize("shape,dimensions", [("square", 2), ("cube", 3)])
@pytest.mark.parametrize("size", [2, 3])
def test_rectangular_footprint_width(shape, dimensions, size):
    actual = get_footprint(shape, size)

    assert actual.shape == (size,) * dimensions
    assert actual.dtype == numpy.uint8
    assert numpy.all(actual == 1)

import numpy
import numpy.testing
import pytest
import scipy.ndimage
from skimage.feature import peak_local_max

import cellprofiler.modules.findmaxima
import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

from cellprofiler_library.opts.findmaxima import BackgroundExclusionMode

instance = cellprofiler.modules.findmaxima.FindMaxima()


def test_run_threshold_mode(image, module, image_set, workspace):
    """Test FindMaxima with threshold background exclusion mode"""
    module.x_name.value = "example"
    module.y_name.value = "maxima"
    module.exclude_mode.value = BackgroundExclusionMode.THRESHOLD.value
    module.min_intensity.value = 0.5
    module.min_distance.value = 5
    module.label_maxima.value = True

    module.run(workspace)

    actual = image_set.get_image("maxima")

    # Compute expected result using skimage directly
    maxima_coords = peak_local_max(
        image.pixel_data,
        min_distance=5,
        threshold_abs=0.5,
    )
    expected_bool = numpy.zeros(image.pixel_data.shape, dtype=bool)
    expected_bool[tuple(maxima_coords.T)] = True
    expected = scipy.ndimage.label(expected_bool)[0]

    numpy.testing.assert_array_equal(actual.pixel_data, expected)


def test_run_no_threshold(image, module, image_set, workspace):
    """Test FindMaxima with threshold set to 0 (no threshold)"""
    module.x_name.value = "example"
    module.y_name.value = "maxima"
    module.exclude_mode.value = BackgroundExclusionMode.THRESHOLD.value
    module.min_intensity.value = 0
    module.min_distance.value = 3
    module.label_maxima.value = False

    module.run(workspace)

    actual = image_set.get_image("maxima")

    # Compute expected result
    maxima_coords = peak_local_max(
        image.pixel_data,
        min_distance=3,
        threshold_abs=0,
    )
    expected = numpy.zeros(image.pixel_data.shape, dtype=bool)
    expected[tuple(maxima_coords.T)] = True

    numpy.testing.assert_array_equal(actual.pixel_data, expected)


def test_run_mask_mode(image, module, image_set, workspace, image_set_list):
    """Test FindMaxima with mask background exclusion mode"""
    # Create a binary mask
    mask = numpy.ones(image.pixel_data.shape, dtype=bool)
    if image.dimensions == 3 or image.multichannel:
        if image.dimensions == 3:
            mask[:, :5, :] = False
        else:
            mask[:5, :] = False
    else:
        mask[:5, :] = False

    mask_image = cellprofiler_core.image.Image(image=mask, dimensions=image.dimensions)
    image_set.add("mask", mask_image)

    module.x_name.value = "example"
    module.y_name.value = "maxima"
    module.exclude_mode.value = BackgroundExclusionMode.MASK.value
    module.mask_image.value = "mask"
    module.min_distance.value = 5
    module.label_maxima.value = True

    module.run(workspace)

    actual = image_set.get_image("maxima")

    # Compute expected result
    masked_data = image.pixel_data.copy()
    masked_data[~mask] = 0
    maxima_coords = peak_local_max(
        masked_data,
        min_distance=5,
        threshold_abs=None,
    )
    expected_bool = numpy.zeros(image.pixel_data.shape, dtype=bool)
    expected_bool[tuple(maxima_coords.T)] = True
    expected = scipy.ndimage.label(expected_bool)[0]

    numpy.testing.assert_array_equal(actual.pixel_data, expected)


def test_run_objects_mode(image, module, image_set, workspace, object_set):
    """Test FindMaxima with objects background exclusion mode"""
    # Create test objects
    objects = numpy.zeros(image.pixel_data.shape, dtype=numpy.uint8)
    if image.dimensions == 3:
        objects[1:10, 10:50, 10:50] = 1
        objects[1:10, 60:100, 10:50] = 2
    elif image.multichannel:
        objects[10:50, 10:50] = 1
        objects[60:100, 10:50] = 2
    else:
        objects[10:50, 10:50] = 1
        objects[60:100, 10:50] = 2

    test_objects = cellprofiler_core.object.Objects()
    test_objects.segmented = objects
    test_objects.parent_image = image
    object_set.add_objects(test_objects, "TestObjects")

    module.x_name.value = "example"
    module.y_name.value = "maxima"
    module.exclude_mode.value = BackgroundExclusionMode.OBJECTS.value
    module.mask_objects.value = "TestObjects"
    module.min_distance.value = 5
    module.label_maxima.value = True

    module.run(workspace)

    actual = image_set.get_image("maxima")

    # Compute expected result
    mask = objects.astype(bool)
    masked_data = image.pixel_data.copy()
    masked_data[~mask] = 0
    maxima_coords = peak_local_max(
        masked_data,
        min_distance=5,
        threshold_abs=None,
    )
    expected_bool = numpy.zeros(image.pixel_data.shape, dtype=bool)
    expected_bool[tuple(maxima_coords.T)] = True
    expected = scipy.ndimage.label(expected_bool)[0]

    numpy.testing.assert_array_equal(actual.pixel_data, expected)


def test_run_unlabeled_maxima(image, module, image_set, workspace):
    """Test FindMaxima with label_maxima set to False"""
    module.x_name.value = "example"
    module.y_name.value = "maxima"
    module.exclude_mode.value = BackgroundExclusionMode.THRESHOLD.value
    module.min_intensity.value = 0.3
    module.min_distance.value = 10
    module.label_maxima.value = False

    module.run(workspace)

    actual = image_set.get_image("maxima")

    # Compute expected result (boolean array, not labeled)
    maxima_coords = peak_local_max(
        image.pixel_data,
        min_distance=10,
        threshold_abs=0.3,
    )
    expected = numpy.zeros(image.pixel_data.shape, dtype=bool)
    expected[tuple(maxima_coords.T)] = True

    numpy.testing.assert_array_equal(actual.pixel_data, expected)


def test_zero_image(module, image_set_list, workspace_empty, image_set_empty):
    """Test FindMaxima on an image with all zeros"""
    zero_image = cellprofiler_core.image.Image(image=numpy.zeros((20, 20)), dimensions=2)
    image_set_empty.add("zero", zero_image)

    module.x_name.value = "zero"
    module.y_name.value = "maxima"
    module.exclude_mode.value = BackgroundExclusionMode.THRESHOLD.value
    module.min_intensity.value = 0
    module.min_distance.value = 5
    module.label_maxima.value = True

    module.run(workspace_empty)

    actual = image_set_empty.get_image("maxima")

    # Should find no maxima in a zero image
    assert numpy.all(actual.pixel_data == 0)


def test_single_peak(module, image_set_list, workspace_empty, image_set_empty):
    """Test FindMaxima on an image with a single peak"""
    data = numpy.zeros((20, 20))
    data[10, 10] = 1.0
    peak_image = cellprofiler_core.image.Image(image=data, dimensions=2)
    image_set_empty.add("peak", peak_image)

    module.x_name.value = "peak"
    module.y_name.value = "maxima"
    module.exclude_mode.value = BackgroundExclusionMode.THRESHOLD.value
    module.min_intensity.value = 0.5
    module.min_distance.value = 1
    module.label_maxima.value = True

    module.run(workspace_empty)

    actual = image_set_empty.get_image("maxima")

    # Should find exactly one maximum
    assert numpy.sum(actual.pixel_data > 0) == 1
    assert actual.pixel_data[10, 10] > 0


def test_multiple_peaks(module, image_set_list, workspace_empty, image_set_empty):
    """Test FindMaxima on an image with multiple well-separated peaks"""
    data = numpy.zeros((50, 50))
    # Create three distinct peaks
    data[10, 10] = 1.0
    data[10, 40] = 0.9
    data[40, 25] = 0.8
    peaks_image = cellprofiler_core.image.Image(image=data, dimensions=2)
    image_set_empty.add("peaks", peaks_image)

    module.x_name.value = "peaks"
    module.y_name.value = "maxima"
    module.exclude_mode.value = BackgroundExclusionMode.THRESHOLD.value
    module.min_intensity.value = 0.5
    module.min_distance.value = 5
    module.label_maxima.value = True

    module.run(workspace_empty)

    actual = image_set_empty.get_image("maxima")

    # Should find exactly three maxima
    assert numpy.sum(actual.pixel_data > 0) == 3
    assert actual.pixel_data[10, 10] > 0
    assert actual.pixel_data[10, 40] > 0
    assert actual.pixel_data[40, 25] > 0


def test_min_distance(module, image_set_list, workspace_empty, image_set_empty):
    """Test that min_distance parameter works correctly"""
    data = numpy.zeros((50, 50))
    # Create two close peaks
    data[25, 25] = 1.0
    data[25, 28] = 0.9  # Only 3 pixels away
    peaks_image = cellprofiler_core.image.Image(image=data, dimensions=2)
    image_set_empty.add("peaks", peaks_image)

    module.x_name.value = "peaks"
    module.y_name.value = "maxima"
    module.exclude_mode.value = BackgroundExclusionMode.THRESHOLD.value
    module.min_intensity.value = 0
    module.min_distance.value = 5
    module.label_maxima.value = True

    module.run(workspace_empty)

    actual = image_set_empty.get_image("maxima")

    # Should find only one maximum (the stronger one) due to min_distance
    assert numpy.sum(actual.pixel_data > 0) == 1
    assert actual.pixel_data[25, 25] > 0


def test_volume_3d(module, image_set_list, workspace_empty, image_set_empty):
    """Test FindMaxima on 3D volume data"""
    data = numpy.zeros((10, 20, 20))
    data[5, 10, 10] = 1.0
    data[5, 10, 15] = 0.8
    volume_image = cellprofiler_core.image.Image(image=data, dimensions=3)
    image_set_empty.add("volume", volume_image)

    module.x_name.value = "volume"
    module.y_name.value = "maxima"
    module.exclude_mode.value = BackgroundExclusionMode.THRESHOLD.value
    module.min_intensity.value = 0.5
    module.min_distance.value = 3
    module.label_maxima.value = True

    module.run(workspace_empty)

    actual = image_set_empty.get_image("maxima")

    # Verify dimensions are preserved
    assert actual.pixel_data.shape == data.shape
    assert actual.dimensions == 3
    
    # Should find two maxima
    assert numpy.sum(actual.pixel_data > 0) == 2
    assert actual.pixel_data[5, 10, 10] > 0
    assert actual.pixel_data[5, 10, 15] > 0
import numpy
import numpy.testing
import pytest
import skimage.morphology

import cellprofiler.modules.fillobjects

instance = cellprofiler.modules.fillobjects.FillObjects()


@pytest.fixture(scope="module")
def image_labels():
    labels = numpy.zeros((20, 20), dtype=numpy.uint8)

    labels[2:8, 2:8] = 1

    labels[0:8, 12:18] = 2

    labels[12:18, 0:8] = 3

    labels[12:20, 12:20] = 4

    return labels


@pytest.fixture(scope="module")
def volume_labels():
    labels = numpy.zeros((9, 20, 20), dtype=numpy.uint8)

    labels[0:9, 2:8, 2:8] = 1

    labels[0:5, 0:8, 12:18] = 2

    labels[4:9, 12:18, 0:8] = 3

    labels[1:8, 12:20, 12:20] = 4

    return labels


def test_run(object_set_with_data, module, workspace_with_data):
    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.size.value = 6.0

    module.run(workspace_with_data)

    actual = workspace_with_data.object_set.get_objects("OutputObjects").segmented

    if actual.ndim == 2:
        factor = 3 ** 2
    else:
        factor = (4.0 / 3.0) * (3 ** 3)

    size = numpy.pi * factor

    expected = object_set_with_data.get_objects("InputObjects").segmented

    for n in numpy.unique(expected):
        if n == 0:
            continue

        filled_mask = skimage.morphology.remove_small_holes(expected == n, size)
        expected[filled_mask] = n

    numpy.testing.assert_array_equal(actual, expected)


def test_2d_fill_holes(
    image_labels, module, object_set_empty, objects_empty, workspace_empty
):
    labels = image_labels.copy()
    labels[5, 5] = 0
    labels[2, 15] = 0
    labels[15, 2] = 0
    labels[15, 15] = 0

    objects_empty.segmented = labels

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.size.value = 2.0
    module.mode.value = "Holes"

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented
    expected = image_labels

    numpy.testing.assert_array_equal(actual, expected)


def test_3d_fill_holes(
    volume_labels, module, object_set_empty, objects_empty, workspace_empty
):
    labels = volume_labels.copy()
    labels[5, 5, 5] = 0
    labels[2, 2, 15] = 0
    labels[5, 15, 2] = 0
    labels[5, 15, 15] = 0

    objects_empty.segmented = labels

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.size.value = 2.0
    module.mode.value = "Holes"

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented
    expected = volume_labels

    numpy.testing.assert_array_equal(actual, expected)


def test_2d_fill_chull(
    image_labels, module, object_set_empty, objects_empty, workspace_empty
):
    labels = image_labels.copy()
    labels[5, 5] = 0
    labels[2, 15] = 0
    labels[15, 0] = 0
    labels[15, 12] = 0

    objects_empty.segmented = labels

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.mode.value = "Convex hull"

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented
    expected = image_labels

    numpy.testing.assert_array_equal(actual, expected)


def test_3d_fill_chull(
    volume_labels, module, object_set_empty, objects_empty, workspace_empty
):
    labels = volume_labels.copy()
    labels[0, 1, 13] = 0
    labels[5, 5, 5] = 0
    labels[5, 5, 7] = 0
    labels[2, 2, 15] = 0
    labels[5, 15, 2] = 0
    labels[5, 15, 15] = 0

    objects_empty.segmented = labels

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.mode.value = "Convex hull"

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented
    expected = volume_labels

    numpy.testing.assert_array_equal(actual, expected)


def test_fail_3d_fill_bowl(
    volume_labels, module, object_set_empty, objects_empty, workspace_empty
):
    labels = volume_labels.copy()
    # Create a 'bowl' topology
    labels[5:10, 4:6, 4:6] = 0

    objects_empty.segmented = labels

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.size.value = 2.0
    module.mode.value = "Holes"

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented
    expected = labels

    # Since a bowl morphology technically doesn't have a 3D whole, they should not be equal
    # i.e. the array should be unaffected
    numpy.testing.assert_array_equal(actual, expected)


def test_pass_3d_fill_bowl(
    volume_labels, module, object_set_empty, objects_empty, workspace_empty
):
    labels = volume_labels.copy()
    # Create a 'bowl' topology
    labels[5:10, 4:6, 4:6] = 0

    objects_empty.segmented = labels

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.size.value = 3.0
    # Set to planewise so the bowl is "filled" on each plane
    module.planewise.value = True
    module.mode.value = "Holes"

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented
    expected = volume_labels

    # We're filling plane-wise here, so each 2D plane should have its holes filled,
    # meaning the 'bowl' should be filled
    numpy.testing.assert_array_equal(actual, expected)

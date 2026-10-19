import numpy
import numpy.testing
import pytest

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler.modules.shrinktoobjectcenters
import cellprofiler_core.object

instance = cellprofiler.modules.shrinktoobjectcenters.ShrinkToObjectCenters()


@pytest.fixture(scope="module")
def image():
    return cellprofiler_core.image.Image()


@pytest.fixture(scope="module")
def image_labels():
    labels = numpy.zeros((20, 20), dtype=numpy.uint8)

    labels[2:8, 2:8] = 1

    labels[0:8, 12:18] = 2

    labels[12:18, 0:8] = 3

    labels[12:20, 12:20] = 4

    return labels


@pytest.fixture(scope="function")
def object_set(objects):
    object_set = cellprofiler_core.object.ObjectSet()

    object_set.add_objects(objects, "InputObjects")

    return object_set


@pytest.fixture(scope="function")
def objects():
    return cellprofiler_core.object.Objects()


@pytest.fixture(scope="module")
def volume_labels():
    labels = numpy.zeros((9, 20, 20), dtype=numpy.uint8)

    labels[0:9, 2:8, 2:8] = 1

    labels[0:5, 0:8, 12:18] = 2

    labels[4:9, 12:18, 0:8] = 3

    labels[1:8, 12:20, 12:20] = 4

    return labels


def test_shrink_image_labels(image_labels, module, object_set, objects, workspace):
    objects.segmented = image_labels

    module.x_name.value = "InputObjects"

    module.run(workspace)

    expected_labels = numpy.zeros((20, 20), dtype=numpy.uint8)

    expected_labels[4, 4] = 1

    expected_labels[3, 14] = 2

    expected_labels[14, 3] = 3

    expected_labels[15, 15] = 4

    numpy.testing.assert_array_equal(
        object_set.get_objects("ShrinkToObjectCenters").segmented, expected_labels
    )


def test_shrink_volume_labels(module, object_set, objects, volume_labels, workspace):
    objects.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.run(workspace)

    expected_labels = numpy.zeros((9, 20, 20), dtype=numpy.uint8)

    expected_labels[4, 4, 4] = 1

    expected_labels[2, 3, 14] = 2

    expected_labels[6, 14, 3] = 3

    expected_labels[4, 15, 15] = 4

    numpy.testing.assert_array_equal(
        object_set.get_objects("ShrinkToObjectCenters").segmented, expected_labels
    )

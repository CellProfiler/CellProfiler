import numpy
import numpy.testing
import pytest

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.resizeobjects
import cellprofiler.object


instance = cellprofiler.modules.resizeobjects.ResizeObjects()


@pytest.fixture(scope="module")
def image():
    return cellprofiler.image.Image()


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
    object_set = cellprofiler.object.ObjectSet()

    object_set.add_objects(objects, "InputObjects")

    return object_set


@pytest.fixture(scope="function")
def objects():
    return cellprofiler.object.Objects()


@pytest.fixture(scope="module")
def volume_labels():
    labels = numpy.zeros((9, 20, 20), dtype=numpy.uint8)

    labels[0:9, 2:8, 2:8] = 1

    labels[0:5, 0:8, 12:18] = 2

    labels[4:9, 12:18, 0:8] = 3

    labels[1:8, 12:20, 12:20] = 4

    return labels


def test_resize_by_factor_shrink_image_labels(image_labels, module, object_set, objects, workspace):
    objects.segmented = image_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Factor"

    module.factor.value = 0.5

    module.run(workspace)

    expected_labels = numpy.zeros((10, 10), dtype=numpy.uint8)

    expected_labels[1:4, 1:4] = 1

    expected_labels[0:4, 6:9] = 2

    expected_labels[6:9, 0:4] = 3

    expected_labels[6:10, 6:10] = 4

    numpy.testing.assert_array_equal(object_set.get_objects("ResizeObjects").segmented, expected_labels)

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "InputObjects",
            cellprofiler.measurement.FF_CHILDREN_COUNT % "ResizeObjects"
        ),
        [1, 1, 1, 1]
    )

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "ResizeObjects",
            cellprofiler.measurement.FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4]
    )


def test_resize_by_factor_enlarge_image_labels(image_labels, module, object_set, objects, workspace):
    objects.segmented = image_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Factor"

    module.factor.value = 2.0

    module.run(workspace)

    expected_labels = numpy.zeros((40, 40), dtype=numpy.uint8)

    expected_labels[4:16, 4:16] = 1

    expected_labels[0:16, 24:36] = 2

    expected_labels[24:36, 0:16] = 3

    expected_labels[24:40, 24:40] = 4

    numpy.testing.assert_array_equal(object_set.get_objects("ResizeObjects").segmented, expected_labels)

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "InputObjects",
            cellprofiler.measurement.FF_CHILDREN_COUNT % "ResizeObjects"
        ),
        [1, 1, 1, 1]
    )

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "ResizeObjects",
            cellprofiler.measurement.FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4]
    )


def test_resize_by_dimensions_shrink_image_labels(image_labels, module, object_set, objects, workspace):
    objects.segmented = image_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Dimensions"

    module.width.value = 5

    module.height.value = 10

    module.run(workspace)

    expected_labels = numpy.zeros((10, 5), dtype=numpy.uint8)

    expected_labels[1:4, 1:2] = 1

    expected_labels[0:4, 3:4] = 2

    expected_labels[6:9, 0:2] = 3

    expected_labels[6:10, 3:5] = 4

    numpy.testing.assert_array_equal(object_set.get_objects("ResizeObjects").segmented, expected_labels)

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "InputObjects",
            cellprofiler.measurement.FF_CHILDREN_COUNT % "ResizeObjects"
        ),
        [1, 1, 1, 1]
    )

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "ResizeObjects",
            cellprofiler.measurement.FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4]
    )


def test_resize_by_dimensions_enlarge_image_labels(image_labels, module, object_set, objects, workspace):
    objects.segmented = image_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Dimensions"

    module.width.value = 80

    module.height.value = 40

    module.run(workspace)

    expected_labels = numpy.zeros((40, 80), dtype=numpy.uint8)

    expected_labels[4:16, 7:32] = 1

    expected_labels[0:16, 48:73] = 2

    expected_labels[24:36, 0:32] = 3

    expected_labels[24:40, 48:80] = 4

    numpy.testing.assert_array_equal(object_set.get_objects("ResizeObjects").segmented, expected_labels)

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "InputObjects",
            cellprofiler.measurement.FF_CHILDREN_COUNT % "ResizeObjects"
        ),
        [1, 1, 1, 1]
    )

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "ResizeObjects",
            cellprofiler.measurement.FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4]
    )


def test_resize_by_factor_shrink_volume_labels(module, object_set, objects, volume_labels, workspace):
    objects.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Factor"

    module.factor.value = 0.5

    module.run(workspace)

    expected_labels = numpy.zeros((9, 10, 10), dtype=numpy.uint8)

    expected_labels[0:9, 1:4, 1:4] = 1

    expected_labels[0:5, 0:4, 6:9] = 2

    expected_labels[4:9, 6:9, 0:4] = 3

    expected_labels[1:8, 6:10, 6:10] = 4

    numpy.testing.assert_array_equal(object_set.get_objects("ResizeObjects").segmented, expected_labels)

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "InputObjects",
            cellprofiler.measurement.FF_CHILDREN_COUNT % "ResizeObjects"
        ),
        [1, 1, 1, 1]
    )

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "ResizeObjects",
            cellprofiler.measurement.FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4]
    )


def test_resize_by_factor_enlarge_volume_labels(module, object_set, objects, volume_labels, workspace):
    objects.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Factor"

    module.factor.value = 2.0

    module.run(workspace)

    expected_labels = numpy.zeros((9, 40, 40), dtype=numpy.uint8)

    expected_labels[0:9, 4:16, 4:16] = 1

    expected_labels[0:5, 0:16, 24:36] = 2

    expected_labels[4:9, 24:36, 0:16] = 3

    expected_labels[1:8, 24:40, 24:40] = 4

    numpy.testing.assert_array_equal(object_set.get_objects("ResizeObjects").segmented, expected_labels)

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "InputObjects",
            cellprofiler.measurement.FF_CHILDREN_COUNT % "ResizeObjects"
        ),
        [1, 1, 1, 1]
    )

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "ResizeObjects",
            cellprofiler.measurement.FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4]
    )


def test_resize_by_dimensions_shrink_volume_labels(module, object_set, objects, volume_labels, workspace):
    objects.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Dimensions"

    module.width.value = 5

    module.height.value = 10

    module.run(workspace)

    expected_labels = numpy.zeros((9, 10, 5), dtype=numpy.uint8)

    expected_labels[0:9, 1:4, 1:2] = 1

    expected_labels[0:5, 0:4, 3:4] = 2

    expected_labels[4:9, 6:9, 0:2] = 3

    expected_labels[1:8, 6:10, 3:5] = 4

    numpy.testing.assert_array_equal(object_set.get_objects("ResizeObjects").segmented, expected_labels)

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "InputObjects",
            cellprofiler.measurement.FF_CHILDREN_COUNT % "ResizeObjects"
        ),
        [1, 1, 1, 1]
    )

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "ResizeObjects",
            cellprofiler.measurement.FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4]
    )


def test_resize_by_dimensions_enlarge_volume_labels(module, object_set, objects, volume_labels, workspace):
    objects.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Dimensions"

    module.width.value = 80

    module.height.value = 40

    module.run(workspace)

    expected_labels = numpy.zeros((9, 40, 80), dtype=numpy.uint8)

    expected_labels[0:9, 4:16, 7:32] = 1

    expected_labels[0:5, 0:16, 48:73] = 2

    expected_labels[4:9, 24:36, 0:32] = 3

    expected_labels[1:8, 24:40, 48:80] = 4

    numpy.testing.assert_array_equal(object_set.get_objects("ResizeObjects").segmented, expected_labels)

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "InputObjects",
            cellprofiler.measurement.FF_CHILDREN_COUNT % "ResizeObjects"
        ),
        [1, 1, 1, 1]
    )

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement(
            "ResizeObjects",
            cellprofiler.measurement.FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4]
    )

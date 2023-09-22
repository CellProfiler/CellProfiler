import numpy
import numpy.testing
import pytest

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import FF_CHILDREN_COUNT, FF_PARENT

import cellprofiler.modules.resizeobjects
import cellprofiler_core.object

instance = cellprofiler.modules.resizeobjects.ResizeObjects()


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


def test_resize_by_factor_shrink_image_labels(
    image_labels, module, object_set_empty, objects_empty, workspace_empty
):
    objects_empty.segmented = image_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Factor"

    module.factor_x.value = module.factor_y.value = 0.5

    module.run(workspace_empty)

    expected_labels = numpy.zeros((10, 10), dtype=numpy.uint8)

    expected_labels[1:4, 1:4] = 1

    expected_labels[0:4, 6:9] = 2

    expected_labels[6:9, 0:4] = 3

    expected_labels[6:10, 6:10] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )


def test_resize_by_factor_enlarge_image_labels(
    image_labels, module, object_set_empty, objects_empty, workspace_empty
):
    objects_empty.segmented = image_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Factor"

    module.factor_x.value = module.factor_y.value = 2.0

    module.run(workspace_empty)

    expected_labels = numpy.zeros((40, 40), dtype=numpy.uint8)

    expected_labels[4:16, 4:16] = 1

    expected_labels[0:16, 24:36] = 2

    expected_labels[24:36, 0:16] = 3

    expected_labels[24:40, 24:40] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )


def test_resize_by_dimensions_shrink_image_labels(
    image_labels, module, object_set_empty, objects_empty, workspace_empty
):
    objects_empty.segmented = image_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Dimensions"

    module.width.value = 5

    module.height.value = 10

    module.run(workspace_empty)

    expected_labels = numpy.zeros((10, 5), dtype=numpy.uint8)

    expected_labels[1:4, 1:2] = 1

    expected_labels[0:4, 3:4] = 2

    expected_labels[6:9, 0:2] = 3

    expected_labels[6:10, 3:5] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )


def test_resize_by_dimensions_enlarge_image_labels(
    image_labels, module, object_set_empty, objects_empty, workspace_empty
):
    objects_empty.segmented = image_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Dimensions"

    module.width.value = 80

    module.height.value = 40

    module.run(workspace_empty)

    expected_labels = numpy.zeros((40, 80), dtype=numpy.uint8)

    expected_labels[4:16, 7:32] = 1

    expected_labels[0:16, 48:73] = 2

    expected_labels[24:36, 0:32] = 3

    expected_labels[24:40, 48:80] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )


def test_resize_by_factor_shrink_volume_labels(
    module, object_set_empty, objects_empty, volume_labels, workspace_empty
):
    objects_empty.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Factor"

    module.factor_x.value = module.factor_y.value = 0.5

    module.factor_z.value = 1

    module.run(workspace_empty)

    expected_labels = numpy.zeros((9, 10, 10), dtype=numpy.uint8)

    expected_labels[0:9, 1:4, 1:4] = 1

    expected_labels[0:5, 0:4, 6:9] = 2

    expected_labels[4:9, 6:9, 0:4] = 3

    expected_labels[1:8, 6:10, 6:10] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )


def test_resize_by_factor_shrink_volume_labels_in_z(
    module, object_set_empty, objects_empty, volume_labels, workspace_empty
):
    objects_empty.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Factor"

    module.factor_x.value = module.factor_y.value = 0.5

    module.factor_z.value = 0.5

    module.run(workspace_empty)

    expected_labels = numpy.zeros((4, 10, 10), dtype=numpy.uint8)

    expected_labels[0:4, 1:4, 1:4] = 1

    expected_labels[0:2, 0:4, 6:9] = 2

    expected_labels[2:4, 6:9, 0:4] = 3

    expected_labels[1:3, 6:10, 6:10] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )



def test_resize_by_factor_enlarge_volume_labels(
    module, object_set_empty, objects_empty, volume_labels, workspace_empty
):
    objects_empty.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Factor"

    module.factor_x.value = module.factor_y.value = 2.0

    module.factor_z.value = 1

    module.run(workspace_empty)

    expected_labels = numpy.zeros((9, 40, 40), dtype=numpy.uint8)

    expected_labels[0:9, 4:16, 4:16] = 1

    expected_labels[0:5, 0:16, 24:36] = 2

    expected_labels[4:9, 24:36, 0:16] = 3

    expected_labels[1:8, 24:40, 24:40] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )

def test_resize_by_factor_enlarge_volume_labels_in_z(
    module, object_set_empty, objects_empty, volume_labels, workspace_empty
):
    objects_empty.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Factor"

    module.factor_x.value = module.factor_y.value = 2.0

    module.factor_z.value = 2

    module.run(workspace_empty)

    expected_labels = numpy.zeros((18, 40, 40), dtype=numpy.uint8)

    expected_labels[0:18, 4:16, 4:16] = 1

    expected_labels[0:10, 0:16, 24:36] = 2

    expected_labels[8:18, 24:36, 0:16] = 3

    expected_labels[2:16, 24:40, 24:40] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )

def test_resize_by_dimensions_shrink_volume_labels(
    module, object_set_empty, objects_empty, volume_labels, workspace_empty
):
    objects_empty.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Dimensions"

    module.width.value = 5

    module.height.value = 10

    module.planes.value = 9

    module.run(workspace_empty)

    expected_labels = numpy.zeros((9, 10, 5), dtype=numpy.uint8)

    expected_labels[0:9, 1:4, 1:2] = 1

    expected_labels[0:5, 0:4, 3:4] = 2

    expected_labels[4:9, 6:9, 0:2] = 3

    expected_labels[1:8, 6:10, 3:5] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )

def test_resize_by_dimensions_shrink_volume_labels_in_z(
    module, object_set_empty, objects_empty, volume_labels, workspace_empty
):
    objects_empty.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Dimensions"

    module.width.value = 5

    module.height.value = 10

    module.planes.value = 4

    module.run(workspace_empty)

    expected_labels = numpy.zeros((4, 10, 5), dtype=numpy.uint8)

    expected_labels[0:4, 1:4, 1:2] = 1

    expected_labels[0:2, 0:4, 3:4] = 2

    expected_labels[2:4, 6:9, 0:2] = 3

    expected_labels[1:3, 6:10, 3:5] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )

def test_resize_by_dimensions_enlarge_volume_labels(
    module, object_set_empty, objects_empty, volume_labels, workspace_empty
):
    objects_empty.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Dimensions"

    module.width.value = 80

    module.height.value = 40

    module.planes.value = 9

    module.run(workspace_empty)

    expected_labels = numpy.zeros((9, 40, 80), dtype=numpy.uint8)

    expected_labels[0:9, 4:16, 7:32] = 1

    expected_labels[0:5, 0:16, 48:73] = 2

    expected_labels[4:9, 24:36, 0:32] = 3

    expected_labels[1:8, 24:40, 48:80] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )

def test_resize_by_dimensions_enlarge_volume_labels_in_z(
    module, object_set_empty, objects_empty, volume_labels, workspace_empty
):
    objects_empty.segmented = volume_labels

    module.x_name.value = "InputObjects"

    module.method.value = "Dimensions"

    module.width.value = 80

    module.height.value = 40

    module.planes.value = 18

    module.run(workspace_empty)

    expected_labels = numpy.zeros((18, 40, 80), dtype=numpy.uint8)

    expected_labels[0:18, 4:16, 7:32] = 1

    expected_labels[0:10, 0:16, 48:73] = 2

    expected_labels[8:18, 24:36, 0:32] = 3

    expected_labels[2:16, 24:40, 48:80] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )


def test_resize_by_image(
    module, object_set_empty, objects_empty, image_labels, workspace_empty
):
    objects_empty.segmented = image_labels
    pixel_data = numpy.zeros((10, 5))
    workspace_empty.image_set.add(
        "TestImage", cellprofiler_core.image.Image(pixel_data)
    )
    module.x_name.value = "InputObjects"
    module.method.value = "Match Image"
    module.specific_image.value = "TestImage"

    module.run(workspace_empty)

    expected_labels = numpy.zeros((10, 5), dtype=numpy.uint8)
    expected_labels[1:4, 1:2] = 1
    expected_labels[0:4, 3:4] = 2
    expected_labels[6:9, 0:2] = 3
    expected_labels[6:10, 3:5] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )


def test_resize_by_image_volume(
    module, object_set_empty, objects_empty, volume_labels, workspace_empty
):
    objects_empty.segmented = volume_labels
    pixel_data = numpy.zeros((9, 10, 5))
    workspace_empty.image_set.add(
        "TestImage", cellprofiler_core.image.Image(pixel_data, dimensions=3)
    )
    module.x_name.value = "InputObjects"
    module.method.value = "Match Image"
    module.specific_image.value = "TestImage"

    module.run(workspace_empty)

    expected_labels = numpy.zeros((9, 10, 5), dtype=numpy.uint8)
    expected_labels[0:9, 1:4, 1:2] = 1
    expected_labels[0:5, 0:4, 3:4] = 2
    expected_labels[4:9, 6:9, 0:2] = 3
    expected_labels[1:8, 6:10, 3:5] = 4

    numpy.testing.assert_array_equal(
        object_set_empty.get_objects("ResizeObjects").segmented, expected_labels
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "InputObjects", FF_CHILDREN_COUNT % "ResizeObjects",
        ),
        [1, 1, 1, 1],
    )

    numpy.testing.assert_array_equal(
        workspace_empty.measurements.get_current_measurement(
            "ResizeObjects", FF_PARENT % "InputObjects"
        ),
        [1, 2, 3, 4],
    )

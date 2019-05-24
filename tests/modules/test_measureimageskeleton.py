import numpy
import pytest
import skimage.morphology

import cellprofiler.measurement
import cellprofiler.modules.measureimageskeleton


instance = cellprofiler.modules.measureimageskeleton.MeasureImageSkeleton()


@pytest.fixture(scope="module")
def image_skeleton():
    data = numpy.zeros((9, 9), dtype=numpy.float32)

    data[4, :] = 1.0

    data[:, 4] = 1.0

    return data


@pytest.fixture(scope="module")
def volume_skeleton():
    data = numpy.zeros((9, 9, 9), dtype=numpy.float32)

    data[4, 4, :] = 1.0

    data[4, :, 4] = 1.0

    data[:, 4, 4] = 1.0

    return data


@pytest.fixture(
    scope="module",
    params=[
        (image_skeleton(), 2),
        (volume_skeleton(), 3)
    ],
    ids=[
        "image_skeleton",
        "volume_skeleton"
    ]
)
def image(request):
    data, dimensions = request.param

    return cellprofiler.image.Image(image=data, dimensions=dimensions, convert=False)


def test_get_categories_image(module, pipeline):
    expected_categories = [
        "Skeleton"
    ]

    categories = module.get_categories(pipeline, cellprofiler.measurement.IMAGE)

    assert categories == expected_categories


def test_get_categories_other(module, pipeline):
    expected_categories = []

    categories = module.get_categories(pipeline, "foo")

    assert categories == expected_categories


def test_get_measurement_columns(module, pipeline):
    module.skeleton_name.value = "example"

    expected_columns = [
        (
            cellprofiler.measurement.IMAGE,
            "Skeleton_Branches_example",
            cellprofiler.measurement.COLTYPE_INTEGER
        ),
        (
            cellprofiler.measurement.IMAGE,
            "Skeleton_Endpoints_example",
            cellprofiler.measurement.COLTYPE_INTEGER
        )
    ]

    columns = module.get_measurement_columns(pipeline)

    assert columns == expected_columns


def test_get_measurements_image_skeleton(module, pipeline):
    module.skeleton_name.value = "example"

    expected_measurements = [
        "Skeleton_Branches_example",
        "Skeleton_Endpoints_example"
    ]

    measurements = module.get_measurements(pipeline, cellprofiler.measurement.IMAGE, "Skeleton")

    assert measurements == expected_measurements


def test_get_measurements_image_other(module, pipeline):
    module.skeleton_name.value = "example"

    expected_measurements = []

    measurements = module.get_measurements(pipeline, cellprofiler.measurement.IMAGE, "foo")

    assert measurements == expected_measurements


def test_get_measurements_other(module, pipeline):
    module.skeleton_name.value = "example"

    expected_measurements = []

    measurements = module.get_measurements(pipeline, "foo", "Skeleton")

    assert measurements == expected_measurements


def test_get_measurement_images(module, pipeline):
    module.skeleton_name.value = "example"

    expected_images = ["example"]

    images = module.get_measurement_images(
        pipeline,
        cellprofiler.measurement.IMAGE,
        "Skeleton",
        "Skeleton_Branches_example"
    )

    assert images == expected_images


def test_run(image, module, workspace):
    module.skeleton_name.value = "example"

    module.run(workspace)

    branches = workspace.measurements.get_current_measurement(
        cellprofiler.measurement.IMAGE,
        "Skeleton_Branches_example"
    )

    endpoints = workspace.measurements.get_current_measurement(
        cellprofiler.measurement.IMAGE,
        "Skeleton_Endpoints_example"
    )

    if image.volumetric:
        expected_branches = 7

        expected_endpoints = 6
    else:
        expected_branches = 5

        expected_endpoints = 4

    assert branches == expected_branches

    assert endpoints == expected_endpoints

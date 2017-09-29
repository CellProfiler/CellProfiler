import numpy
import numpy.random
import pytest

import cellprofiler.measurement
import cellprofiler.modules.convertimagetoobjects


instance = cellprofiler.modules.convertimagetoobjects.ConvertImageToObjects()


@pytest.fixture
def binary_image():
    data = numpy.zeros((100, 100), dtype=numpy.bool)

    data[10:30, 10:30] = True

    data[40:90, 40:50] = True
    data[40:50, 40:90] = True

    return data


@pytest.fixture
def binary_volume():
    data = numpy.zeros((10, 100, 100), dtype=numpy.bool)

    data[2:6, 10:30, 10:30] = True

    data[4:6, 40:90, 40:50] = True
    data[4:6, 40:50, 40:90] = True
    data[2:8, 40:50, 40:50] = True

    return data


def binary_to_grayscale(binary):
    data = numpy.random.randint(0, 128, binary.shape).astype(numpy.uint8)

    foreground = numpy.random.randint(128, 256, binary.shape).astype(numpy.uint8)

    data[binary] = foreground[binary]

    return data


@pytest.fixture(
    scope="module",
    params=[
        binary_image(),
        binary_to_grayscale(binary_image()),
        binary_volume(),
        binary_to_grayscale(binary_volume())
    ],
    ids=[
        "binary_image",
        "grayscale_image",
        "binary_volume",
        "grayscale_volume"
    ]
)
def image(request):
    data = request.param

    dimensions = data.ndim

    return cellprofiler.image.Image(image=data, dimensions=dimensions)


def test_run(image, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "labeled"

    module.run(workspace)

    objects = workspace.object_set.get_objects("labeled")

    assert len(numpy.unique(objects.segmented)) == 3

    measurements = workspace.measurements

    assert measurements.has_current_measurements("labeled", cellprofiler.measurement.M_LOCATION_CENTER_X)
    assert measurements.has_current_measurements("labeled", cellprofiler.measurement.M_LOCATION_CENTER_Y)
    assert measurements.has_current_measurements("labeled", cellprofiler.measurement.M_LOCATION_CENTER_Z)
    assert measurements.has_current_measurements("labeled", cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER)
    assert measurements.has_current_measurements(
        cellprofiler.measurement.IMAGE,
        cellprofiler.measurement.FF_COUNT % "labeled"
    )

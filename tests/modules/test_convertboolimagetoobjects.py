import numpy
import numpy.random
import pytest

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y, M_LOCATION_CENTER_Z, \
    M_NUMBER_OBJECT_NUMBER, FF_COUNT

import cellprofiler.modules.convertimagetoobjects

instance = cellprofiler.modules.convertimagetoobjects.ConvertImageToObjects()


@pytest.fixture
def binary_image():
    data = numpy.zeros((100, 100), dtype=bool)

    data[10:30, 10:30] = True

    data[40:90, 40:50] = True
    data[40:50, 40:90] = True

    return data


@pytest.fixture
def binary_volume():
    data = numpy.zeros((10, 100, 100), dtype=bool)

    data[2:6, 10:30, 10:30] = True

    data[4:6, 40:90, 40:50] = True
    data[4:6, 40:50, 40:90] = True
    data[2:8, 40:50, 40:50] = True

    return data


@pytest.fixture
def binary_image_to_grayscale(binary_image):
    data = numpy.random.randint(0, 128, binary_image.shape).astype(numpy.uint8)

    foreground = numpy.random.randint(128, 256, binary_image.shape).astype(numpy.uint8)

    data[binary_image] = foreground[binary_image]

    return data


@pytest.fixture
def binary_volume_to_grayscale(binary_volume):
    data = numpy.random.randint(0, 128, binary_volume.shape).astype(numpy.uint8)

    foreground = numpy.random.randint(128, 256, binary_volume.shape).astype(numpy.uint8)

    data[binary_volume] = foreground[binary_volume]

    return data


@pytest.fixture(
    params=[
        "binary_image",
        "binary_image_to_grayscale",
        "binary_volume",
        "binary_volume_to_grayscale",
    ]
)
def image(request):
    data = request.getfixturevalue(request.param)

    dimensions = data.ndim

    return cellprofiler_core.image.Image(image=data, dimensions=dimensions)


def test_run_boolean(image, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "labeled"

    module.cast_to_bool.value = True

    module.run(workspace)

    objects = workspace.object_set.get_objects("labeled")

    assert len(numpy.unique(objects.segmented)) == 3

    measurements = workspace.measurements

    assert measurements.has_current_measurements(
        "labeled", M_LOCATION_CENTER_X
    )
    assert measurements.has_current_measurements(
        "labeled", M_LOCATION_CENTER_Y
    )
    assert measurements.has_current_measurements(
        "labeled", M_LOCATION_CENTER_Z
    )
    assert measurements.has_current_measurements(
        "labeled", M_NUMBER_OBJECT_NUMBER
    )
    assert measurements.has_current_measurements(
        "Image",
        FF_COUNT % "labeled",
    )

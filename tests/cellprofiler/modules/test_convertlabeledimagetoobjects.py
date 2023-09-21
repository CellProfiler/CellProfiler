import numpy
import numpy.random
import pytest

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import M_LOCATION_CENTER_X, FF_COUNT, M_LOCATION_CENTER_Y, \
    M_LOCATION_CENTER_Z, M_NUMBER_OBJECT_NUMBER

import cellprofiler.modules.convertimagetoobjects

instance = cellprofiler.modules.convertimagetoobjects.ConvertImageToObjects()


@pytest.fixture
def labeled_image():
    # Pre-labeled, connected image
    data = numpy.zeros((100, 100), dtype=numpy.uint8)

    data[10:30, 10:30] = 1

    data[40:90, 40:50] = 2
    data[40:50, 40:90] = 3

    return data


@pytest.fixture
def labeled_volume():
    # Pre-labeled, connected volume
    data = numpy.zeros((10, 100, 100), dtype=numpy.uint8)

    data[2:6, 10:30, 10:30] = 1

    data[4:6, 40:90, 40:50] = 2
    data[4:6, 40:50, 40:90] = 3

    return data


@pytest.fixture(params=["labeled_image", "labeled_volume"])
def image(request):
    data = request.getfixturevalue(request.param)

    dimensions = data.ndim

    return cellprofiler_core.image.Image(image=data, dimensions=dimensions)


def test_run_labels(image, module, workspace):
    # Ensure that pre-labeled objects retain their labels
    # even if they are connected
    module.x_name.value = "example"

    module.y_name.value = "labeled"

    module.cast_to_bool.value = False

    module.run(workspace)

    objects = workspace.object_set.get_objects("labeled")

    assert len(numpy.unique(objects.segmented)) == 4

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

import numpy
import numpy.testing
import pytest

import cellprofiler.image
import cellprofiler.modules.clearborder
import cellprofiler.measurement
import cellprofiler.object

instance = cellprofiler.modules.clearborder.ClearBorder()


@pytest.fixture(scope="function")
def object_set(objects):
    object_set = cellprofiler.object.ObjectSet()

    object_set.add_objects(objects, "InputObjects")

    return object_set


@pytest.fixture(scope="function")
def objects():
    return cellprofiler.object.Objects()


def test_run(module, object_set, objects, workspace):
    x = numpy.array(
        [[0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    objects.segmented = x

    module.x_name.value = "InputObjects"

    module.run(workspace)

    x = object_set.get_objects("ClearBorder").segmented

    y = numpy.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    numpy.testing.assert_array_equal(x, y)

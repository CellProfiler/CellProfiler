import numpy
import numpy.testing
import skimage.morphology

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.fillobjects
import cellprofiler.object


instance = cellprofiler.modules.fillobjects.FillObjects()

labels = numpy.zeros((20, 20), dtype=numpy.uint8)

labels[2:8, 2:8] = 1
labels[5, 5] = 0

labels[0:8, 12:18] = 2
labels[2, 15] = 0

labels[12:18, 0:8] = 3
labels[15, 2] = 0

labels[12:20, 12:20] = 4
labels[15, 15] = 0

labels = numpy.zeros((9, 20, 20), dtype=numpy.uint8)

labels[0:9, 2:8, 2:8] = 1

labels[0:5, 0:8, 12:18] = 2

labels[4:9, 12:18, 0:8] = 3

labels[1:8, 12:20, 12:20] = 4


def test_run(object_set_with_data, module, workspace_with_data):
    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.size.value = 6.

    module.run(workspace_with_data)

    actual = workspace_with_data.object_set.get_objects("OutputObjects").segmented

    if actual.ndim == 2:
        factor = 3 ** 2
    else:
        factor = (4.0 / 3.0) * (3 ** 3)

    size = numpy.pi * factor

    expected = actual.copy()

    for n in numpy.unique(expected):
        if n == 0:
            continue

        filled_mask = skimage.morphology.remove_small_holes(expected == n, size)
        expected[filled_mask] = n

    numpy.testing.assert_array_equal(actual, expected)



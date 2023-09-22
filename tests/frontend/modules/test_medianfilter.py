import numpy.testing
import scipy.signal

import cellprofiler.modules.medianfilter

instance = cellprofiler.modules.medianfilter.MedianFilter()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "MedianFilter"

    module.window.value = 3

    module.run(workspace)

    actual = image_set.get_image("MedianFilter")

    desired = scipy.signal.medfilt(image.pixel_data, 3)

    numpy.testing.assert_array_almost_equal(actual.pixel_data, desired)

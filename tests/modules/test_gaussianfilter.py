from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import cellprofiler.modules.gaussianfilter
import numpy.testing
import skimage.filters

instance = cellprofiler.modules.gaussianfilter.GaussianFilter()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "GaussianFilter"

    module.run(workspace)

    actual = image_set.get_image("GaussianFilter")

    desired = skimage.filters.gaussian(
        image=image.pixel_data,
        sigma=1
    )

    numpy.testing.assert_array_almost_equal(actual.pixel_data, desired)

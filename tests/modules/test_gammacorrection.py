from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import cellprofiler.modules.gammacorrection
import numpy.testing
import skimage.exposure

instance = cellprofiler.modules.gammacorrection.GammaCorrection()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "GammaCorrection"

    module.run(workspace)

    actual = image_set.get_image("GammaCorrection")

    desired = skimage.exposure.adjust_gamma(image.pixel_data)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)

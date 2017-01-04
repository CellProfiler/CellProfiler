from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import cellprofiler.modules.dilation
import numpy.testing
import skimage.morphology

instance = cellprofiler.modules.dilation.Dilation()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "dilation"

    if image.dimensions is 3 or image.multichannel:
        module.structuring_element.shape = "ball"

        selem = skimage.morphology.ball(1)
    else:
        selem = skimage.morphology.disk(1)

    module.run(workspace)

    actual = image_set.get_image("dilation")

    desired = skimage.morphology.dilation(image.pixel_data, selem)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)

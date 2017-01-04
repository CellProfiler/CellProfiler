from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import cellprofiler.modules.randomwalkeralgorithm
import numpy
import numpy.testing
import skimage.measure
import skimage.segmentation

instance = cellprofiler.modules.randomwalkeralgorithm.RandomWalkerAlgorithm()


def test_run(image, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "RandomWalkerAlgorithm"

    module.first_phase.value = 0.5

    module.second_phase.value = 0.5

    module.beta.value = 130.0

    module.run(workspace)

    x_data = image.pixel_data

    labels_data = numpy.zeros_like(x_data, numpy.uint)

    labels_data[x_data < 0.5] = 1
    labels_data[x_data > 0.5] = 2

    expected = skimage.segmentation.random_walker(
        beta=130.0,
        data=x_data,
        labels=labels_data,
        multichannel=image.multichannel,
        spacing=image.spacing
    )

    expected = skimage.measure.label(expected)

    actual = workspace.get_objects("RandomWalkerAlgorithm")

    numpy.testing.assert_array_equal(expected, actual.segmented)

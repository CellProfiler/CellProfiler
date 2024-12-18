import numpy.testing
import skimage.filters

import cellprofiler.modules.gaussianfilter

instance = cellprofiler.modules.gaussianfilter.GaussianFilter()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "GaussianFilter"

    module.run(workspace)

    actual = image_set.get_image("GaussianFilter")

    if image.multichannel:
        channel_axis = -1
    else:
        channel_axis = None

    desired = skimage.filters.gaussian(image=image.pixel_data, sigma=1, channel_axis=channel_axis)

    numpy.testing.assert_array_almost_equal(actual.pixel_data, desired)

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

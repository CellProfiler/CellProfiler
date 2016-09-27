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

import cellprofiler.modules.gaussianfilter
import numpy.testing
import skimage.filters
import cellprofiler.image

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

    # FIXME: Change this once 3D rescaling is better
    if image.dimensions == 3:
        desired = cellprofiler.image.Image(
            image=desired,
            dimensions=3
        ).pixel_data

    numpy.testing.assert_array_almost_equal(actual.pixel_data, desired)

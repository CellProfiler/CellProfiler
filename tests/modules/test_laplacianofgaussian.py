import cellprofiler.image
import cellprofiler.modules.laplacianofgaussian
import numpy.testing
import scipy.ndimage.filters
import skimage.color

instance = cellprofiler.modules.laplacianofgaussian.LaplacianOfGaussian()


def test_run(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "LaplacianOfGaussian"

    module.x.value = 1.1

    module.y.value = 1.2

    module.z.value = 1.3

    module.run(workspace)

    actual = image_set.get_image("LaplacianOfGaussian")

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    data = skimage.img_as_float(data)

    if image.dimensions == 2:
        sigma = (1.1, 1.2)
    else:
        sigma = (1.3, 1.1, 1.2)

    expected_data = scipy.ndimage.filters.gaussian_laplace(data, sigma)

    expected = cellprofiler.image.Image(
        dimensions=image.dimensions,
        image=expected_data,
        parent_image=image
    )

    numpy.testing.assert_array_equal(expected.pixel_data, actual.pixel_data)

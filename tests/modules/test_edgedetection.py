import cellprofiler.image
import cellprofiler.modules.edgedetection
import numpy
import numpy.random
import numpy.testing
import skimage.color
import skimage.filters

instance = cellprofiler.modules.edgedetection.EdgeDetection()


def test_run_without_mask(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "EdgeDetection"

    module.mask.value = "Leave blank"

    module.run(workspace)

    actual = image_set.get_image("EdgeDetection")

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    if image.dimensions == 2:
        expected_data = skimage.filters.sobel(data)
    else:
        expected_data = numpy.zeros_like(data)

        for idx, img in enumerate(data):
            expected_data[idx] = skimage.filters.sobel(img)

    expected = cellprofiler.image.Image(
        image=expected_data,
        parent_image=image,
        dimensions=image.dimensions
    )

    numpy.testing.assert_array_equal(expected.pixel_data, actual.pixel_data)


def test_run_with_mask(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "EdgeDetection"

    module.mask.value = "mask"

    mask_shape = image.pixel_data.shape

    if image.dimensions == 2:
        mask_data = numpy.random.rand(mask_shape[0], mask_shape[1])

        mask_data[:5] = 0

        mask_data[-5:] = 0

        mask_data[:, :5] = 0

        mask_data[:, -5:] = 0
    else:
        mask_data = numpy.random.rand(*mask_shape)

        mask_data[:, :5] = 0

        mask_data[:, -5:] = 0

        mask_data[:, :, :5] = 0

        mask_data[:, :, -5:] = 0

    mask_data = mask_data != 0

    mask = cellprofiler.image.Image(
        image=mask_data
    )

    image_set.add("mask", mask)

    module.run(workspace)

    actual = image_set.get_image("EdgeDetection")

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    if image.dimensions == 2:
        expected_data = skimage.filters.sobel(data, mask=mask_data)
    else:
        expected_data = numpy.zeros_like(data)

        for idx, img in enumerate(data):
            expected_data[idx] = skimage.filters.sobel(img, mask=mask_data[idx])

    expected = cellprofiler.image.Image(
        image=expected_data,
        parent_image=image,
        dimensions=image.dimensions
    )

    numpy.testing.assert_array_equal(expected.pixel_data, actual.pixel_data)

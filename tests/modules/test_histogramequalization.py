import cellprofiler.image
import cellprofiler.modules.histogramequalization
import numpy
import numpy.testing
import skimage.exposure

instance = cellprofiler.modules.histogramequalization.HistogramEqualization()


def test_run(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "HistogramEqualization"

    module.nbins.value = 256

    module.mask.value = "Leave blank"

    module.local.value = False

    module.run(workspace)

    actual = image_set.get_image("HistogramEqualization")

    data = image.pixel_data

    expected_data = skimage.exposure.equalize_hist(data)

    expected = cellprofiler.image.Image(
        image=expected_data,
        parent_image=image,
        dimensions=image.dimensions
    )

    numpy.testing.assert_array_equal(expected.pixel_data, actual.pixel_data)


def test_run_local(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "HistogramEqualization"

    module.nbins.value = 256

    module.local.value = True

    module.run(workspace)

    actual = image_set.get_image("HistogramEqualization")

    data = image.pixel_data

    if image.volumetric:
        expected_data = numpy.zeros_like(data)

        for index, plane in enumerate(data):
            expected_data[index] = skimage.exposure.equalize_adapthist(plane, kernel_size=256)
    else:
        expected_data = skimage.exposure.equalize_adapthist(data, kernel_size=256)

    expected = cellprofiler.image.Image(
        image=expected_data,
        parent_image=image,
        dimensions=image.dimensions
    )

    numpy.testing.assert_array_equal(expected.pixel_data, actual.pixel_data)


def test_run_nbins(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "HistogramEqualization"

    module.nbins.value = 128

    module.local.value = False

    module.mask.value = "Leave blank"

    module.run(workspace)

    actual = image_set.get_image("HistogramEqualization")

    data = image.pixel_data

    expected_data = skimage.exposure.equalize_hist(data, nbins=128)

    expected = cellprofiler.image.Image(
        image=expected_data,
        parent_image=image,
        dimensions=image.dimensions
    )

    numpy.testing.assert_array_equal(expected.pixel_data, actual.pixel_data)


def test_run_mask(image, image_set, module, workspace):
    data = image.pixel_data

    mask_data = numpy.zeros_like(data, dtype="bool")

    if image.multichannel:
        mask_data[5:-5, 5:-5, :] = True
    elif image.dimensions == 3:
        mask_data[:, 5:-5, 5:-5] = True
    else:
        mask_data[5:-5, 5:-5] = True

    mask = cellprofiler.image.Image(
        image=mask_data,
        dimensions=image.dimensions
    )

    image_set.add("Mask", mask)

    module.x_name.value = "example"

    module.y_name.value = "HistogramEqualization"

    module.nbins.value = 256

    module.local.value = False

    module.mask.value = "Mask"

    module.run(workspace)

    actual = image_set.get_image("HistogramEqualization")

    expected_data = skimage.exposure.equalize_hist(data, mask=mask_data)

    expected = cellprofiler.image.Image(
        image=expected_data,
        parent_image=image,
        dimensions=image.dimensions
    )

    numpy.testing.assert_array_equal(expected.pixel_data, actual.pixel_data)

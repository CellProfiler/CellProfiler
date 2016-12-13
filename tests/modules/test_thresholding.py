# -*- coding: utf-8 -*-

import numpy
import numpy.testing
import pytest
import skimage.data
import skimage.exposure
import skimage.filters
import skimage.morphology

import cellprofiler.image
import cellprofiler.modules.thresholding

instance = cellprofiler.modules.thresholding.Thresholding()


@pytest.fixture(
    scope="module",
    params=[
        (skimage.data.camera()[0:128, 0:128], 2),
        (numpy.tile(skimage.data.camera()[0:32, 0:32], (2, 1)).reshape(2, 32, 32), 3)
    ],
    ids=[
        "grayscale_image",
        "grayscale_volume"
    ]
)
def image(request):
    data, dimensions = request.param

    return cellprofiler.image.Image(image=data, dimensions=dimensions)


def test_run_global_iterative_selection(image, module, workspace):
    module.x_name.value = "example"

    module.local.value = False

    module.global_operation.value = u"Iterative selection thresholding"

    module.bins.value = 256

    module.run(workspace)

    output = workspace.image_set.get_image("Thresholding")

    actual = output.pixel_data

    data = skimage.img_as_ubyte(image.pixel_data)

    threshold = skimage.filters.threshold_isodata(
        image=data,
        nbins=256
    )

    expected = data >= threshold

    numpy.testing.assert_array_equal(expected, actual)


def test_run_manual(image, module, workspace):
    module.x_name.value = "example"

    module.local.value = False

    module.global_operation.value = u"Manual"

    module.minimum.value = 0.2

    module.maximum.value = 0.8

    module.run(workspace)

    output = workspace.image_set.get_image("Thresholding")

    actual = output.pixel_data

    data = skimage.img_as_float(image.pixel_data)

    data = skimage.exposure.rescale_intensity(data)

    expected = numpy.zeros_like(data, dtype=numpy.bool)

    expected[data > 0.2] = True

    expected[data < 0.8] = False

    numpy.testing.assert_array_equal(expected, actual)


def test_run_global_minimum_cross_entropy(image, module, workspace):
    module.x_name.value = "example"

    module.local.value = False

    module.global_operation.value = u"Minimum cross entropy thresholding"

    module.run(workspace)

    output = workspace.image_set.get_image("Thresholding")

    actual = output.pixel_data

    data = skimage.img_as_ubyte(image.pixel_data)

    threshold = skimage.filters.threshold_li(data)

    expected = data >= threshold

    numpy.testing.assert_array_equal(expected, actual)


def test_run_global_otsu(image, module, workspace):
    module.x_name.value = "example"

    module.local.value = False

    module.global_operation.value = u"Otsu’s method"

    module.bins.value = 256

    module.run(workspace)

    output = workspace.image_set.get_image("Thresholding")

    actual = output.pixel_data

    data = skimage.img_as_ubyte(image.pixel_data)

    threshold = skimage.filters.threshold_otsu(
        image=data,
        nbins=256
    )

    expected = data >= threshold

    numpy.testing.assert_array_equal(expected, actual)


def test_run_global_yen(image, module, workspace):
    module.x_name.value = "example"

    module.local.value = False

    module.global_operation.value = u"Yen’s method"

    module.bins.value = 256

    module.run(workspace)

    output = workspace.image_set.get_image("Thresholding")

    actual = output.pixel_data

    data = skimage.img_as_ubyte(image.pixel_data)

    threshold = skimage.filters.threshold_yen(
        image=data,
        nbins=256
    )

    expected = data >= threshold

    numpy.testing.assert_array_equal(expected, actual)


def test_run_local_adaptive_gaussian(image, module, workspace):
    module.x_name.value = "example"

    module.local.value = True

    module.local_operation.value = u"Adaptive"

    module.adaptive_method.value = u"Gaussian"

    module.sigma.value = 2

    module.block_size.value = 7

    module.offset.value = 0

    module.run(workspace)

    output = workspace.image_set.get_image("Thresholding")

    actual = output.pixel_data

    data = skimage.img_as_ubyte(image.pixel_data)

    if image.volumetric:
        expected = numpy.zeros_like(data, dtype=numpy.bool)

        for index, plane in enumerate(data):
            expected[index] = skimage.filters.threshold_adaptive(
                image=plane,
                block_size=7,
                method="gaussian",
                offset=0,
                param=2
            )
    else:
        expected = skimage.filters.threshold_adaptive(
            image=data,
            block_size=7,
            method="gaussian",
            offset=0,
            param=2
        )

    numpy.testing.assert_array_equal(expected, actual)


def test_run_local_adaptive_mean(image, module, workspace):
    module.x_name.value = "example"

    module.local.value = True

    module.local_operation.value = u"Adaptive"

    module.adaptive_method.value = u"Mean"

    module.block_size.value = 7

    module.offset.value = 0

    module.run(workspace)

    output = workspace.image_set.get_image("Thresholding")

    actual = output.pixel_data

    data = skimage.img_as_ubyte(image.pixel_data)

    if image.volumetric:
        expected = numpy.zeros_like(data, dtype=numpy.bool)

        for index, plane in enumerate(data):
            expected[index] = skimage.filters.threshold_adaptive(
                image=plane,
                block_size=7,
                method="mean",
                offset=0
            )
    else:
        expected = skimage.filters.threshold_adaptive(
            image=data,
            block_size=7,
            method="mean",
            offset=0
        )

    numpy.testing.assert_array_equal(expected, actual)


def test_run_local_adaptive_median(image, module, workspace):
    module.x_name.value = "example"

    module.local.value = True

    module.local_operation.value = u"Adaptive"

    module.adaptive_method.value = u"Median"

    module.block_size.value = 7

    module.offset.value = 0

    module.run(workspace)

    output = workspace.image_set.get_image("Thresholding")

    actual = output.pixel_data

    data = skimage.img_as_ubyte(image.pixel_data)

    if image.volumetric:
        expected = numpy.zeros_like(data, dtype=numpy.bool)

        for index, plane in enumerate(data):
            expected[index] = skimage.filters.threshold_adaptive(
                image=plane,
                block_size=7,
                method="median",
                offset=0
            )
    else:
        expected = skimage.filters.threshold_adaptive(
            image=data,
            block_size=7,
            method="median",
            offset=0
        )

    numpy.testing.assert_array_equal(expected, actual)


def test_run_local_otsu(image, module, workspace):
    module.x_name.value = "example"

    module.local.value = True

    module.local_operation.value = u"Otsu’s method"

    module.radius.value = 8

    module.run(workspace)

    output = workspace.image_set.get_image("Thresholding")

    actual = output.pixel_data

    data = skimage.img_as_ubyte(image.pixel_data)

    disk = skimage.morphology.disk(8)

    if image.volumetric:
        expected = numpy.zeros_like(data, dtype=numpy.bool)

        for index, plane in enumerate(data):
            expected[index] = skimage.filters.rank.otsu(plane, disk)
    else:
        expected = skimage.filters.rank.otsu(data, disk)

    expected = expected >= data

    numpy.testing.assert_array_equal(expected, actual)


def test_run_local_percentile(image, module, workspace):
    module.x_name.value = "example"

    module.local.value = True

    module.local_operation.value = u"Percentile"

    module.radius.value = 8

    module.run(workspace)

    output = workspace.image_set.get_image("Thresholding")

    actual = output.pixel_data

    data = skimage.img_as_ubyte(image.pixel_data)

    disk = skimage.morphology.disk(8)

    if image.volumetric:
        expected = numpy.zeros_like(data, dtype=numpy.bool)

        for index, plane in enumerate(data):
            expected[index] = skimage.filters.rank.percentile(plane, disk)
    else:
        expected = skimage.filters.rank.percentile(data, disk)

    expected = expected >= data

    numpy.testing.assert_array_equal(expected, actual)

# coding=utf-8

import cellprofiler.image
import cellprofiler.modules.blobdetection
import cellprofiler.workspace
import numpy
import numpy.testing
import pytest
import skimage.color
import skimage.data
import skimage.draw
import skimage.feature


data = skimage.data.hubble_deep_field()[0:500, 0:500]


@pytest.fixture(
    scope="module",
    params=[
        (skimage.color.rgb2gray(data), 2),
        (data, 2),
        (numpy.tile(skimage.color.rgb2gray(data), (3, 1)).reshape((3, 500, 500)), 3)
    ],
    ids=[
        "grayscale_image",
        "multichannel_image",
        "grayscale_volume"
    ]
)
def image(request):
    data, dimensions = request.param

    return cellprofiler.image.Image(image=data, dimensions=dimensions)


instance = cellprofiler.modules.blobdetection.BlobDetection()


def test_run_dog(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "blobs"

    module.operation.value = "Difference of Gaussians (DoG)"

    module.on_setting_changed(module.operation, workspace.pipeline)

    module.run(workspace)

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    expected = numpy.zeros_like(data)

    if image.dimensions == 2:
        blobs = skimage.feature.blob_dog(data)

        blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

        for r, c, radius in blobs:
            rr, cc = skimage.draw.circle(r, c, radius)

            in_bounds = numpy.all(
                [
                    rr >= 0,
                    cc >= 0,
                    rr < expected.shape[0],
                    cc < expected.shape[1]
                ],
                axis=0
            )

            expected[rr[in_bounds], cc[in_bounds]] = 1
    else:
        for z, plane in enumerate(data):
            blobs = skimage.feature.blob_dog(plane)

            blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

            for r, c, radius in blobs:
                rr, cc = skimage.draw.circle(r, c, radius)

                in_bounds = numpy.all(
                    [
                        rr >= 0,
                        cc >= 0,
                        rr < plane.shape[0],
                        cc < plane.shape[1]
                    ],
                    axis=0
                )

                expected[z, rr[in_bounds], cc[in_bounds]] = 1

    actual = image_set.get_image("blobs")

    numpy.testing.assert_array_equal(expected, actual.pixel_data)


def test_run_doh_with_linear_scale(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "blobs"

    module.operation.value = "Determinant of the Hessian (DoH)"

    module.on_setting_changed(module.operation, workspace.pipeline)

    module.scale.value = "Linear interpolation"

    module.run(workspace)

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    expected = numpy.zeros_like(data)

    if image.dimensions == 2:
        blobs = skimage.feature.blob_doh(data)

        blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

        for r, c, radius in blobs:
            rr, cc = skimage.draw.circle(r, c, radius)

            in_bounds = numpy.all(
                [
                    rr >= 0,
                    cc >= 0,
                    rr < expected.shape[0],
                    cc < expected.shape[1]
                ],
                axis=0
            )

            expected[rr[in_bounds], cc[in_bounds]] = 1
    else:
        for z, plane in enumerate(data):
            blobs = skimage.feature.blob_doh(plane)

            blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

            for r, c, radius in blobs:
                rr, cc = skimage.draw.circle(r, c, radius)

                in_bounds = numpy.all(
                    [
                        rr >= 0,
                        cc >= 0,
                        rr < plane.shape[0],
                        cc < plane.shape[1]
                    ],
                    axis=0
                )

                expected[z, rr[in_bounds], cc[in_bounds]] = 1

    actual = image_set.get_image("blobs")

    numpy.testing.assert_array_equal(expected, actual.pixel_data)


def test_run_doh_with_logarithmic_scale(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "blobs"

    module.operation.value = "Determinant of the Hessian (DoH)"

    module.on_setting_changed(module.operation, workspace.pipeline)

    module.scale.value = "Logarithm"

    module.run(workspace)

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    expected = numpy.zeros_like(data)

    if image.dimensions == 2:
        blobs = skimage.feature.blob_doh(data, log_scale=True)

        blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

        for r, c, radius in blobs:
            rr, cc = skimage.draw.circle(r, c, radius)

            in_bounds = numpy.all(
                [
                    rr >= 0,
                    cc >= 0,
                    rr < expected.shape[0],
                    cc < expected.shape[1]
                ],
                axis=0
            )

            expected[rr[in_bounds], cc[in_bounds]] = 1
    else:
        for z, plane in enumerate(data):
            blobs = skimage.feature.blob_doh(plane, log_scale=True)

            blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

            for r, c, radius in blobs:
                rr, cc = skimage.draw.circle(r, c, radius)

                in_bounds = numpy.all(
                    [
                        rr >= 0,
                        cc >= 0,
                        rr < plane.shape[0],
                        cc < plane.shape[1]
                    ],
                    axis=0
                )

                expected[z, rr[in_bounds], cc[in_bounds]] = 1

    actual = image_set.get_image("blobs")

    numpy.testing.assert_array_equal(expected, actual.pixel_data)


def test_run_log_with_linear_scale(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "blobs"

    module.operation.value = "Laplacian of Gaussian (LoG)"

    module.on_setting_changed(module.operation, workspace.pipeline)

    module.scale.value = "Linear interpolation"

    module.run(workspace)

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    expected = numpy.zeros_like(data)

    if image.dimensions == 2:
        blobs = skimage.feature.blob_log(data)

        blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

        for r, c, radius in blobs:
            rr, cc = skimage.draw.circle(r, c, radius)

            in_bounds = numpy.all(
                [
                    rr >= 0,
                    cc >= 0,
                    rr < expected.shape[0],
                    cc < expected.shape[1]
                ],
                axis=0
            )

            expected[rr[in_bounds], cc[in_bounds]] = 1
    else:
        for z, plane in enumerate(data):
            blobs = skimage.feature.blob_log(plane)

            blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

            for r, c, radius in blobs:
                rr, cc = skimage.draw.circle(r, c, radius)

                in_bounds = numpy.all(
                    [
                        rr >= 0,
                        cc >= 0,
                        rr < plane.shape[0],
                        cc < plane.shape[1]
                    ],
                    axis=0
                )

                expected[z, rr[in_bounds], cc[in_bounds]] = 1

    actual = image_set.get_image("blobs")

    numpy.testing.assert_array_equal(expected, actual.pixel_data)


def test_run_log_with_logarithmic_scale(image, image_set, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "blobs"

    module.operation.value = "Laplacian of Gaussian (LoG)"

    module.on_setting_changed(module.operation, workspace.pipeline)

    module.scale.value = "Logarithm"

    module.run(workspace)

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    expected = numpy.zeros_like(data)

    if image.dimensions == 2:
        blobs = skimage.feature.blob_log(data, log_scale=True)

        blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

        for r, c, radius in blobs:
            rr, cc = skimage.draw.circle(r, c, radius)

            in_bounds = numpy.all(
                [
                    rr >= 0,
                    cc >= 0,
                    rr < expected.shape[0],
                    cc < expected.shape[1]
                ],
                axis=0
            )

            expected[rr[in_bounds], cc[in_bounds]] = 1
    else:
        for z, plane in enumerate(data):
            blobs = skimage.feature.blob_log(plane, log_scale=True)

            blobs[:, 2] = blobs[:, 2] * numpy.sqrt(2)

            for r, c, radius in blobs:
                rr, cc = skimage.draw.circle(r, c, radius)

                in_bounds = numpy.all(
                    [
                        rr >= 0,
                        cc >= 0,
                        rr < plane.shape[0],
                        cc < plane.shape[1]
                    ],
                    axis=0
                )

                expected[z, rr[in_bounds], cc[in_bounds]] = 1

    actual = image_set.get_image("blobs")

    numpy.testing.assert_array_equal(expected, actual.pixel_data)

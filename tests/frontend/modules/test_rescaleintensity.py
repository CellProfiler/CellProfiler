import base64
import io
import zlib

import numpy
import pytest
import skimage.data
import skimage.exposure

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.module
import cellprofiler_core.modules.injectimage
import cellprofiler.modules.rescaleintensity
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

INPUT_NAME = "input"
OUTPUT_NAME = "output"
REFERENCE_NAME = "reference"
MEASUREMENT_NAME = "measurement"


@pytest.fixture(scope="function")
def image():
    data = numpy.tile(skimage.data.camera(), (3, 1)).reshape(3, 512, 512)

    return cellprofiler_core.image.Image(image=data, dimensions=3, convert=False)


@pytest.fixture(scope="function")
def mask(image):
    data = image.pixel_data

    mask = numpy.ones_like(data, dtype=bool)

    mask[data < 50] = False

    mask[data > 205] = False

    return mask


@pytest.fixture(scope="function")
def module():
    module = cellprofiler.modules.rescaleintensity.RescaleIntensity()

    module.x_name.value = "input"

    module.y_name.value = "RescaleIntensity"

    return module


@pytest.fixture(scope="function")
def workspace(image, module):
    image_set_list = cellprofiler_core.image.ImageSetList()

    image_set = image_set_list.get_image_set(0)

    image_set.add("input", image)

    return cellprofiler_core.workspace.Workspace(
        pipeline=cellprofiler_core.pipeline.Pipeline(),
        module=module,
        image_set=image_set,
        object_set=cellprofiler_core.object.ObjectSet(),
        measurements=cellprofiler_core.measurement.Measurements(),
        image_set_list=image_set_list,
    )


def test_stretch(image, module, workspace):
    data = image.pixel_data

    module.rescale_method.value = "Stretch each image to use the full intensity range"

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = skimage.exposure.rescale_intensity(1.0 * data)

    numpy.testing.assert_array_equal(expected, actual)

    assert numpy.any(actual == 0.0)

    assert numpy.any(actual == 1.0)


def test_stretch_masked(image, mask, module, workspace):
    data = image.pixel_data

    image.mask = mask

    module.rescale_method.value = "Stretch each image to use the full intensity range"

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = skimage.exposure.rescale_intensity(1.0 * data, in_range=(50, 205))

    numpy.testing.assert_array_equal(expected, actual)

    assert numpy.any(actual == 0.0)

    assert numpy.any(actual == 1.0)


def test_manual_input_range(image, module, workspace):
    data = image.pixel_data

    module.rescale_method.value = (
        "Choose specific values to be reset to the full intensity range"
    )

    module.wants_automatic_low.value = "Custom"

    module.wants_automatic_high.value = "Custom"

    module.source_scale.value = (50, 205)

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = skimage.exposure.rescale_intensity(1.0 * data, (50, 205))

    numpy.testing.assert_array_equal(expected, actual)

    assert numpy.any(actual == 0.0)

    assert numpy.any(actual == 1.0)


def test_manual_io_range(image, module, workspace):
    data = image.pixel_data

    module.rescale_method.value = "Choose specific values to be reset to a custom range"

    module.wants_automatic_low.value = "Custom"

    module.wants_automatic_high.value = "Custom"

    module.source_scale.value = (50, 205)

    module.dest_scale.value = (0.2, 0.8)

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = skimage.exposure.rescale_intensity(1.0 * data, (50, 205), (0.2, 0.8))

    numpy.testing.assert_array_equal(expected, actual)

    assert numpy.any(actual == 0.2)

    assert numpy.any(actual == 0.8)


def test_divide_by_image_minimum_zero(module, workspace):
    module.rescale_method.value = "Divide by the image's minimum"

    with pytest.raises(ZeroDivisionError):
        module.run(workspace)


def test_divide_by_image_minimum(image, module, workspace):
    data = image.pixel_data

    data[data < 50] = 50

    image.pixel_data = data

    module.rescale_method.value = "Divide by the image's minimum"

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = data / 50.0

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_divide_by_image_minimum_masked(image, mask, module, workspace):
    data = image.pixel_data

    image.mask = mask

    module.rescale_method.value = "Divide by the image's minimum"

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = data / 50.0

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_divide_by_image_maximum_zero(image, module, workspace):
    data = numpy.zeros_like(image.pixel_data)

    image.pixel_data = data

    module.rescale_method.value = "Divide by the image's maximum"

    with pytest.raises(ZeroDivisionError):
        module.run(workspace)


def test_divide_by_image_maximum(image, module, workspace):
    data = image.pixel_data

    module.rescale_method.value = "Divide by the image's maximum"

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = data / 255.0

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_divide_by_image_maximum_masked(image, mask, module, workspace):
    data = image.pixel_data

    image.mask = mask

    module.rescale_method.value = "Divide by the image's maximum"

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = data / 205.0

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_divide_by_value(image, module, workspace):
    data = image.pixel_data

    module.rescale_method.value = "Divide each image by the same value"

    module.divisor_value.value = 50.0

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = data / 50.0

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_divide_by_measurement_zero(module, workspace):
    module.rescale_method.value = "Divide each image by a previously calculated value"

    module.divisor_measurement.value = "foo"

    measurements = workspace.measurements

    measurements.add_image_measurement("foo", [0.0])

    with pytest.raises(ZeroDivisionError):
        module.run(workspace)


def test_divide_by_measurement(image, module, workspace):
    data = image.pixel_data

    module.rescale_method.value = "Divide each image by a previously calculated value"

    module.divisor_measurement.value = "foo"

    measurements = workspace.measurements

    measurements.add_image_measurement("foo", [50.0])

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = data / 50.0

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_scale_by_image_maximum_zero(image, module, workspace):
    data = numpy.zeros_like(image.pixel_data)

    match_data = image.pixel_data

    image.pixel_data = data

    match_image = cellprofiler_core.image.Image(
        image=match_data, dimensions=3, convert=False
    )

    workspace.image_set.add("match", match_image)

    module.rescale_method.value = "Match the image's maximum to another image's maximum"

    module.matching_image_name.value = "match"

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    numpy.testing.assert_array_equal(data, actual)


def test_scale_by_image_maximum(image, module, workspace):
    data = image.pixel_data

    match_data = data / 2.0

    match_image = cellprofiler_core.image.Image(
        image=match_data, dimensions=3, convert=False
    )

    workspace.image_set.add("match", match_image)

    module.rescale_method.value = "Match the image's maximum to another image's maximum"

    module.matching_image_name.value = "match"

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = data * (127.5 / 255.0)

    numpy.testing.assert_array_equal(expected, actual)


def test_scale_by_image_maximum_masked(image, mask, module, workspace):
    data = image.pixel_data

    match_data = data / 2.0

    match_image = cellprofiler_core.image.Image(
        image=match_data, mask=mask, dimensions=3, convert=False
    )

    workspace.image_set.add("match", match_image)

    image.mask = mask

    module.rescale_method.value = "Match the image's maximum to another image's maximum"

    module.matching_image_name.value = "match"

    module.run(workspace)

    output = workspace.image_set.get_image("RescaleIntensity")

    actual = output.pixel_data

    expected = data * (102.5 / 205.0)

    numpy.testing.assert_array_equal(expected, actual)

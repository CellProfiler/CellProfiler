import StringIO
import base64
import os.path
import zlib

import centrosome.filter
import numpy
import pytest
import scipy.ndimage
import skimage.exposure
import skimage.filters
import skimage.transform

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.enhanceorsuppressfeatures
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()


@pytest.fixture(scope="function")
def image():
    return cellprofiler.image.Image()


@pytest.fixture(scope="function")
def module():
    module = cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures()

    module.x_name.value = "input"

    module.y_name.value = "output"

    return module


@pytest.fixture(scope="function")
def workspace(image, module):
    image_set_list = cellprofiler.image.ImageSetList()

    image_set = image_set_list.get_image_set(0)

    image_set.add("input", image)

    return cellprofiler.workspace.Workspace(
        pipeline=cellprofiler.pipeline.Pipeline(),
        module=module,
        image_set=image_set,
        object_set=cellprofiler.object.ObjectSet(),
        measurements=cellprofiler.measurement.Measurements(),
        image_set_list=image_set_list
    )


def test_enhance_zero(image, module, workspace):
    image.pixel_data = numpy.zeros((10, 10))

    module.method.value = "Enhance"

    module.enhance_method.value = "Speckles"

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    assert numpy.all(actual == 0)


def test_suppress_zero(image, module, workspace):
    image.pixel_data = numpy.zeros((10, 10))

    module.method.value = "Suppress"

    module.object_size.value = 10

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    assert numpy.all(actual == 0)


@pytest.fixture(
    params=["Slow", "Fast"],
    scope="function"
)
def accuracy(request):
    return request.param


def test_enhance(accuracy, image, module, workspace):
    data = numpy.zeros((20, 30))

    expected = numpy.zeros((20, 30))

    i, j = numpy.mgrid[-10:10, -10:20]

    data[i ** 2 + j ** 2 <= 7 ** 2] = 1

    i, j = numpy.mgrid[-10:10, -25:5]

    data[i ** 2 + j ** 2 <= 9] = 1

    expected[i ** 2 + j ** 2 <= 9] = 1

    image.pixel_data = data

    module.method.value = "Enhance"

    module.enhance_method.value = "Speckles"

    module.speckle_accuracy.value = accuracy

    module.object_size.value = 14

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    numpy.testing.assert_array_equal(expected, actual)


def test_enhance_volume(accuracy, image, module, workspace):
    data = numpy.zeros((20, 20, 30))

    expected = numpy.zeros((20, 20, 30))

    k, i, j = numpy.mgrid[-10:10, -10:10, -10:20]

    data[k ** 2 + i ** 2 + j ** 2 <= 7 ** 2] = 1

    k, i, j = numpy.mgrid[-10:10, -10:10, -25:5]

    data[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    expected[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    image.pixel_data = data

    image.dimensions = 3

    module.method.value = "Enhance"

    module.enhance_method.value = "Speckles"

    module.speckle_accuracy.value = accuracy

    module.object_size.value = 14

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    numpy.testing.assert_array_equal(expected, actual)


def test_enhance_masked(accuracy, image, module, workspace):
    data = numpy.zeros((20, 30))

    mask = numpy.ones_like(data, dtype=numpy.bool)

    i, j = numpy.mgrid[-10:10, -10:20]

    data[i ** 2 + j ** 2 <= 7 ** 2] = 1

    mask[i ** 2 + j ** 2 <= 7 ** 2] = False

    i, j = numpy.mgrid[-10:10, -25:5]

    data[i ** 2 + j ** 2 <= 9] = 1

    expected = data

    image.pixel_data = data

    image.mask = mask

    module.method.value = "Enhance"

    module.enhance_method.value = "Speckles"

    module.speckle_accuracy.value = accuracy

    module.object_size.value = 14

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    numpy.testing.assert_array_equal(expected, actual)


def test_enhance_masked_volume(accuracy, image, module, workspace):
    data = numpy.zeros((20, 20, 30))

    mask = numpy.ones_like(data, dtype=numpy.bool)

    k, i, j = numpy.mgrid[-10:10, -10:10, -10:20]

    data[k ** 2 + i ** 2 + j ** 2 <= 7 ** 2] = 1

    mask[k ** 2 + i ** 2 + j ** 2 <= 7 ** 2] = False

    k, i, j = numpy.mgrid[-10:10, -10:10, -25:5]

    data[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    expected = data

    image.pixel_data = data

    image.mask = mask

    image.dimensions = 3

    module.method.value = "Enhance"

    module.enhance_method.value = "Speckles"

    module.speckle_accuracy.value = accuracy

    module.object_size.value = 14

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    numpy.testing.assert_array_equal(expected, actual)


def test_suppress(image, module, workspace):
    data = numpy.zeros((20, 30))

    expected = numpy.zeros((20, 30))

    i, j = numpy.mgrid[-10:10, -10:20]

    data[i ** 2 + j ** 2 <= 7 ** 2] = 1

    expected[i ** 2 + j ** 2 <= 7 ** 2] = 1

    i, j = numpy.mgrid[-10:10, -25:5]

    data[i ** 2 + j ** 2 <= 9] = 1

    image.pixel_data = data

    module.method.value = "Suppress"

    module.object_size.value = 14

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    numpy.testing.assert_array_equal(expected, actual)


def test_suppress_volume(image, module, workspace):
    data = numpy.zeros((20, 20, 30))

    expected = numpy.zeros((20, 20, 30))

    k, i, j = numpy.mgrid[-10:10, -10:10, -10:20]

    data[k ** 2 + i ** 2 + j ** 2 <= 7 ** 2] = 1

    expected[k ** 2 + i ** 2 + j ** 2 <= 7 ** 2] = 1

    k, i, j = numpy.mgrid[-10:10, -10:10, -25:5]

    data[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    image.pixel_data = data

    image.dimensions = 3

    module.method.value = "Suppress"

    module.object_size.value = 14

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    numpy.testing.assert_array_equal(expected, actual)


def test_suppress_masked(image, module, workspace):
    data = numpy.zeros((20, 30))

    mask = numpy.ones_like(data, dtype=numpy.bool)

    i, j = numpy.mgrid[-10:10, -10:20]

    data[i ** 2 + j ** 2 <= 7 ** 2] = 1

    i, j = numpy.mgrid[-10:10, -25:5]

    data[i ** 2 + j ** 2 <= 9] = 1

    mask[i ** 2 + j ** 2 <= 9] = False

    expected = data

    image.pixel_data = data

    image.mask = mask

    module.method.value = "Suppress"

    module.object_size.value = 14

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    numpy.testing.assert_array_equal(expected, actual)


def test_suppress_masked_volume(image, module, workspace):
    data = numpy.zeros((20, 20, 30))

    mask = numpy.ones_like(data, dtype=numpy.bool)

    k, i, j = numpy.mgrid[-10:10, -10:10, -10:20]

    data[k ** 2 + i ** 2 + j ** 2 <= 7 ** 2] = 1

    k, i, j = numpy.mgrid[-10:10, -10:10, -25:5]

    data[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    mask[k ** 2 + i ** 2 + j ** 2 <= 9] = False

    expected = data

    image.pixel_data = data

    image.mask = mask

    image.dimensions = 3

    module.method.value = "Suppress"

    module.object_size.value = 14

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    numpy.testing.assert_array_equal(expected, actual)


def test_enhance_neurites_gradient(image, module, workspace):
    resources = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "resources"))

    data = numpy.load(os.path.join(resources, "neurite.npy"))

    data = skimage.exposure.rescale_intensity(1.0 * data)

    image.pixel_data = data

    module.method.value = "Enhance"

    module.enhance_method.value = "Neurites"

    module.neurite_choice.value = "Line structures"

    module.object_size.value = 8

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    expected = numpy.load(os.path.join(resources, "enhanced_neurite.npy"))

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_enhance_neurites_gradient_volume(image, module, workspace):
    resources = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "resources"))

    data = numpy.load(os.path.join(resources, "neurite.npy"))

    data = skimage.exposure.rescale_intensity(1.0 * data)

    data = numpy.tile(data, (3, 1)).reshape(3, *data.shape)

    image.pixel_data = data

    image.dimensions = 3

    module.method.value = "Enhance"

    module.enhance_method.value = "Neurites"

    module.neurite_choice.value = "Line structures"

    module.object_size.value = 8

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    expected = numpy.load(os.path.join(resources, "enhanced_neurite.npy"))

    expected = numpy.tile(expected, (3, 1)).reshape(3, *expected.shape)

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_enhance_neurites_tubeness_positive(image, module, workspace):
    data = numpy.zeros((20, 30))

    data[5:15, 10:20] = numpy.identity(10)

    image.pixel_data = data

    module.method.value = "Enhance"

    module.enhance_method.value = "Neurites"

    module.neurite_choice.value = "Tubeness"

    module.smoothing.value = 1.0

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    expected = centrosome.filter.hessian(
        scipy.ndimage.gaussian_filter(data, 1.0),
        return_hessian=False,
        return_eigenvectors=False
    )

    expected = -expected[:, :, 0] * (expected[:, :, 0] < 0)

    expected = skimage.exposure.rescale_intensity(expected)

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_enhance_neurites_tubeness_negative(image, module, workspace):
    data = numpy.ones((20, 30))

    data[5:15, 10:20] -= numpy.identity(10)

    image.pixel_data = data

    module.method.value = "Enhance"

    module.enhance_method.value = "Neurites"

    module.neurite_choice.value = "Tubeness"

    module.smoothing.value = 1.0

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    expected = centrosome.filter.hessian(
        scipy.ndimage.gaussian_filter(data, 1.0),
        return_hessian=False,
        return_eigenvectors=False
    )

    expected = -expected[:, :, 0] * (expected[:, :, 0] < 0)

    expected = skimage.exposure.rescale_intensity(expected)

    numpy.testing.assert_array_almost_equal(expected, actual, decimal=5)


def test_enhance_neurites_tubeness_positive_volume(image, module, workspace):
    data = numpy.zeros((5, 20, 30))

    data[1:4, 5:15, 10:20] = numpy.identity(10)

    image.pixel_data = data

    image.dimensions = 3

    module.method.value = "Enhance"

    module.enhance_method.value = "Neurites"

    module.neurite_choice.value = "Tubeness"

    module.smoothing.value = 1.0

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    smoothed = scipy.ndimage.gaussian_filter(data, 1.0)

    expected = numpy.zeros_like(smoothed)

    for index, plane in enumerate(smoothed):
        hessian = centrosome.filter.hessian(
            plane,
            return_hessian=False,
            return_eigenvectors=False
        )

        expected[index] = -hessian[:, :, 0] * (hessian[:, :, 0] < 0)

    expected = skimage.exposure.rescale_intensity(expected)

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_enhance_neurites_tubeness_negative_volume(image, module, workspace):
    data = numpy.ones((5, 20, 30))

    data[1:4, 5:15, 10:20] -= numpy.identity(10)

    image.pixel_data = data

    image.dimensions = 3

    module.method.value = "Enhance"

    module.enhance_method.value = "Neurites"

    module.neurite_choice.value = "Tubeness"

    module.smoothing.value = 1.0

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    smoothed = scipy.ndimage.gaussian_filter(data, 1.0)

    expected = numpy.zeros_like(smoothed)

    for index, plane in enumerate(smoothed):
        hessian = centrosome.filter.hessian(
            plane,
            return_hessian=False,
            return_eigenvectors=False
        )

        expected[index] = -hessian[:, :, 0] * (hessian[:, :, 0] < 0)

    expected = skimage.exposure.rescale_intensity(1.0 * expected)

    numpy.testing.assert_array_almost_equal(expected, actual, decimal=5)


def test_enhance_circles(image, module, workspace):
    i, j = numpy.mgrid[-15:16, -15:16]

    circle = numpy.abs(numpy.sqrt(i * i + j * j) - 6) <= 1.5

    image.pixel_data = circle

    module.method.value = "Enhance"

    module.enhance_method.value = "Circles"

    module.object_size.value = 12

    module.run(workspace)

    actual = workspace.image_set.get_image("output").pixel_data

    assert actual[15, 15] == 1

    assert numpy.all(actual[numpy.abs(numpy.sqrt(i * i + j * j) - 6) < 1.5] < .25)


def test_enhance_circles_masked(image, module, workspace):
    data = numpy.zeros((31, 62))

    mask = numpy.ones_like(data, dtype=numpy.bool)

    i, j = numpy.mgrid[-15:16, -15:16]

    circle = numpy.abs(numpy.sqrt(i * i + j * j) - 6) <= 1.5

    data[:, :31] = circle

    data[:, 31:] = circle

    mask[:, 31:][circle] = False

    image.pixel_data = data

    image.mask = mask

    module.method.value = "Enhance"

    module.enhance_method.value = "Circles"

    module.object_size.value = 12

    module.run(workspace)

    actual = workspace.image_set.get_image("output").pixel_data

    assert actual[15, 15] == 1

    assert actual[15, 15 + 31] == 0


def test_enhance_circles_volume(image, module, workspace):
    k, i, j = numpy.mgrid[-15:16, -15:16, -15:16]

    data = numpy.abs(numpy.sqrt(k * k + i * i + j * j) - 6) <= 1.5

    data = data.astype(float)

    image.pixel_data = data

    image.dimensions = 3

    module.method.value = "Enhance"

    module.enhance_method.value = "Circles"

    module.object_size.value = 12

    module.run(workspace)

    actual = workspace.image_set.get_image("output").pixel_data

    expected = numpy.zeros_like(data)

    for index, plane in enumerate(data):
        expected[index] = skimage.transform.hough_circle(plane, 6)[0]

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_enhance_circles_masked_volume(image, module, workspace):
    data = numpy.zeros((31, 31, 62))

    mask = numpy.ones_like(data, dtype=numpy.bool)

    k, i, j = numpy.mgrid[-15:16, -15:16, -15:16]

    circle = numpy.abs(numpy.sqrt(k * k + i * i + j * j) - 6) <= 1.5

    data[:, :, :31] = circle

    data[:, :, 31:] = circle

    data = data.astype(float)

    mask[:, :, 31:][circle] = False

    image.pixel_data = data

    image.mask = mask

    image.dimensions = 3

    module.method.value = cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE

    module.enhance_method.value = cellprofiler.modules.enhanceorsuppressfeatures.E_CIRCLES

    module.object_size.value = 12

    module.run(workspace)

    actual = workspace.image_set.get_image("output").pixel_data

    expected = numpy.zeros_like(data)

    data[:, :, 31:] = 0

    for index, plane in enumerate(data):
        expected[index] = skimage.transform.hough_circle(plane, 6)[0]

    expected[:, :, 31:] = circle

    expected = expected.astype(float)

    numpy.testing.assert_array_almost_equal(expected, actual)


def test_enhance_texture(image, module, workspace):
    r = numpy.random.RandomState()

    r.seed(81)

    data = r.uniform(size=(19, 24))

    image.pixel_data = data

    sigma = 2.1

    module.method.value = "Enhance"

    module.enhance_method.value = "Texture"

    module.smoothing.value = sigma

    module.run(workspace)

    gaussian_mask = skimage.filters.gaussian(numpy.ones_like(data), sigma, mode='constant')

    gaussian = skimage.filters.gaussian(data, sigma, mode='constant') / gaussian_mask

    squared_gaussian = skimage.filters.gaussian(data ** 2, sigma, mode='constant') / gaussian_mask

    expected = squared_gaussian - gaussian ** 2

    actual = workspace.image_set.get_image("output").pixel_data

    numpy.testing.assert_almost_equal(expected, actual)


def test_enhance_texture_masked(image, module, workspace):
    r = numpy.random.RandomState()

    r.seed(81)

    data = r.uniform(size=(19, 24))

    mask = r.uniform(size=data.shape) > .25

    image.pixel_data = data

    image.mask = mask

    sigma = 2.1

    module.method.value = "Enhance"

    module.enhance_method.value = "Texture"

    module.smoothing.value = sigma

    module.run(workspace)

    masked_data = numpy.zeros_like(data)

    masked_data[mask] = data[mask]

    gaussian_mask = skimage.filters.gaussian(mask, sigma, mode='constant')

    gaussian = skimage.filters.gaussian(masked_data, sigma, mode='constant') / gaussian_mask

    squared_gaussian = skimage.filters.gaussian(masked_data ** 2, sigma, mode='constant') / gaussian_mask

    expected = squared_gaussian - gaussian ** 2

    expected[~mask] = data[~mask]

    actual = workspace.image_set.get_image("output").pixel_data

    numpy.testing.assert_almost_equal(actual[mask], expected[mask])


def test_enhance_texture_volume(image, module, workspace):
    r = numpy.random.RandomState()

    r.seed(81)

    data = r.uniform(size=(8, 19, 24))

    image.pixel_data = data

    image.dimensions = 3

    sigma = 2.1

    module.method.value = "Enhance"

    module.enhance_method.value = "Texture"

    module.smoothing.value = sigma

    module.run(workspace)

    gaussian_mask = skimage.filters.gaussian(numpy.ones_like(data), sigma, mode='constant')

    gaussian = skimage.filters.gaussian(data, sigma, mode='constant') / gaussian_mask

    squared_gaussian = skimage.filters.gaussian(data ** 2, sigma, mode='constant') / gaussian_mask

    expected = squared_gaussian - gaussian ** 2

    actual = workspace.image_set.get_image("output").pixel_data

    numpy.testing.assert_almost_equal(expected, actual)


def test_enhance_texture_masked_volume(image, module, workspace):
    r = numpy.random.RandomState()

    r.seed(81)

    data = r.uniform(size=(8, 19, 24))

    mask = r.uniform(size=data.shape) > .25

    image.pixel_data = data

    image.mask = mask

    image.dimensions = 3

    sigma = 2.1

    module.method.value = "Enhance"

    module.enhance_method.value = "Texture"

    module.smoothing.value = sigma

    module.run(workspace)

    masked_data = numpy.zeros_like(data)

    masked_data[mask] = data[mask]

    gaussian_mask = skimage.filters.gaussian(mask, sigma, mode='constant')

    gaussian = skimage.filters.gaussian(masked_data, sigma, mode='constant') / gaussian_mask

    squared_gaussian = skimage.filters.gaussian(masked_data ** 2, sigma, mode='constant') / gaussian_mask

    expected = squared_gaussian - gaussian ** 2

    expected[~mask] = data[~mask]

    actual = workspace.image_set.get_image("output").pixel_data

    numpy.testing.assert_almost_equal(actual[mask], expected[mask])


def test_enhance_dic(image, module, workspace):
    data = numpy.ones((21, 43)) * .5

    data[5:15, 10] = 1

    data[5:15, 15] = 0

    image.pixel_data = data

    module.method.value = "Enhance"

    module.enhance_method.value = "DIC"

    module.angle.value = 90

    module.decay.value = 1

    module.smoothing.value = 0

    module.run(workspace)

    actual = workspace.image_set.get_image("output").pixel_data

    expected = numpy.zeros(data.shape)

    expected[5:15, 10] = .5

    expected[5:15, 11:15] = 1

    expected[5:15, 15] = .5

    numpy.testing.assert_almost_equal(actual, expected)

    module.decay.value = .9

    module.run(workspace)

    actual = workspace.image_set.get_image("output").pixel_data

    assert numpy.all(actual[5:15, 12:14] < 1)

    module.decay.value = 1

    module.smoothing.value = 1

    actual = workspace.image_set.get_image("output").pixel_data

    assert numpy.all(actual[4, 11:15] > .1)


def test_load_v1():
    data = ('eJztWNFO2zAUdUqBsUkr28v26Ee60aotQ4NqKu0oEtUIVLRiQohtpnXba'
            'EkcOQlrNyHtcZ+0x33OHvcJs4NDUhMIbccmpqay2nt9zz3Xx0lsV600dy'
            'qv4Wo2B9VKM9PRdAzrOnI6hBpFaDrLcJNi5OA2JGYRqsSEKhrAfB7mV4o'
            'rheLqOizkcutgvEupqQ/Z19pjAObY9z3WEqJrVthKqHG7gR1HM7v2LEiC'
            'p8L/g7UDRDV0ouMDpLvYDih8f83skObAuuhSSdvV8S4ywsHs2nWNE0ztv'
            'Y4PFN11rY/1hvYZS0Pww/bxqWZrxBR4kV/2XvASR+LlOnyfD3RQJB24Lq'
            'mQn8dvgyA+GaHbo1D8orA1s62dam0X6VAzUPeiCm8eYvLNS/m4rQ5qPI2'
            'HL8fgFyU8b03cdzJbfdRyoIGcVu8meVJSnpRXx5bZQ2YLt4N6cjF5lKE8'
            'CliZQAfBfiMd7kt4blcJNIkDXRsH8xFXf2IoTwLkX46H2yWXcXMSzr983'
            'AII6oy7D59I4+V2FXeQqzvQmy1Y1ShuOYQOJqrjX+Kixj0zNO4ZcMietr'
            'vK9zdwk75/bktX+T2R/4P6RPElh/iS7Pk08SR8X2P43oBhXbn9bmmj/op'
            'vBHAp+zz9nltvsa7vk0+lo0qmfpz2PZtEdw2zdJTLrB9/yS8Xzs6DGxpD'
            'es505LhHqb8XU/+aVD+3eQ2HGFFR2IuzdIa72AbG6QlfQfiqaBB4Jqnz5'
            '9xo6/e4POUYPaLWF2+x71LiWrfPH7XOB/yQbUGwdZfeS1PcFPc/4soh3P'
            'Q5nuJGxS0qV6938jmDx38A199vz8Dw/cbtFttiWJTw/yVo1vAOz3ZWJ6h'
            '9fnrN7rCftdBBlvP0Y3i2JZ7tq3jw+aGOUNu1LIpt27Zw6yPvEce9PdoQ'
            'PQ3RI+u5EMEf1iXBPqnk9fMg6x/My6+NcfgSymW+BzG4pFCS476B0eZ96'
            'Zp4f2zjxv8G/FcCeg==')
    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[1]
    assert module.module_name == 'EnhanceOrSuppressFeatures'
    assert module.x_name.value == 'MyImage'
    assert module.y_name.value == 'MyEnhancedImage'
    assert module.method.value == cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
    assert module.object_size == 17


def test_load_v2():
    data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10583

EnhanceOrSuppressFeatures:[module_num:1|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
Select the input image:Initial
Name the output image:EnhancedSpeckles
Select the operation:Enhance
Feature size:11
Feature type:Speckles
Range of hole sizes:1,10

EnhanceOrSuppressFeatures:[module_num:2|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
Select the input image:EnhancedSpeckles
Name the output image:EnhancedNeurites
Select the operation:Enhance
Feature size:9
Feature type:Neurites
Range of hole sizes:1,10

EnhanceOrSuppressFeatures:[module_num:3|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
Select the input image:EnhancedNeurites
Name the output image:EnhancedDarkHoles
Select the operation:Enhance
Feature size:9
Feature type:Dark holes
Range of hole sizes:4,11

EnhanceOrSuppressFeatures:[module_num:4|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
Select the input image:EnhancedDarkHoles
Name the output image:EnhancedCircles
Select the operation:Enhance
Feature size:9
Feature type:Circles
Range of hole sizes:4,11

EnhanceOrSuppressFeatures:[module_num:5|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
Select the input image:EnhancedCircles
Name the output image:Suppressed
Select the operation:Suppress
Feature size:13
Feature type:Circles
Range of hole sizes:4,11
"""
    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(StringIO.StringIO(data))
    assert len(pipeline.modules()) == 5
    for module, (input_name, output_name, operation, feature_size,
                 feature_type, min_range, max_range) in zip(
            pipeline.modules(), (
                    ("Initial", "EnhancedSpeckles", cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE, 11, cellprofiler.modules.enhanceorsuppressfeatures.E_SPECKLES, 1, 10),
                    ("EnhancedSpeckles", "EnhancedNeurites", cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE, 9, cellprofiler.modules.enhanceorsuppressfeatures.E_NEURITES, 1, 10),
                    ("EnhancedNeurites", "EnhancedDarkHoles", cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE, 9, cellprofiler.modules.enhanceorsuppressfeatures.E_DARK_HOLES, 4, 11),
                    ("EnhancedDarkHoles", "EnhancedCircles", cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE, 9, cellprofiler.modules.enhanceorsuppressfeatures.E_CIRCLES, 4, 11),
                    ("EnhancedCircles", "Suppressed", cellprofiler.modules.enhanceorsuppressfeatures.SUPPRESS, 13, cellprofiler.modules.enhanceorsuppressfeatures.E_CIRCLES, 4, 11))):
        assert module.module_name == 'EnhanceOrSuppressFeatures'
        assert isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures)
        assert module.x_name == input_name
        assert module.y_name == output_name
        assert module.method == operation
        assert module.enhance_method == feature_type
        assert module.object_size == feature_size
        assert module.hole_size.min == min_range
        assert module.hole_size.max == max_range


def test_test_load_v3():
    data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10999

EnhanceOrSuppressFeatures:[module_num:1|svn_version:\'10591\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
Select the input image:DNA
Name the output image:EnhancedTexture
Select the operation:Enhance
Feature size:10
Feature type:Texture
Range of hole sizes:1,10
Smoothing scale:3.5
Shear angle:45
Decay:0.90

EnhanceOrSuppressFeatures:[module_num:2|svn_version:\'10591\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
Select the input image:EnhancedTexture
Name the output image:EnhancedDIC
Select the operation:Enhance
Feature size:10
Feature type:DIC
Range of hole sizes:1,10
Smoothing scale:1.5
Shear angle:135
Decay:0.99
'''
    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(StringIO.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures)
    assert module.x_name == "DNA"
    assert module.y_name == "EnhancedTexture"
    assert module.method == cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
    assert module.enhance_method == cellprofiler.modules.enhanceorsuppressfeatures.E_TEXTURE
    assert module.smoothing == 3.5
    assert module.object_size == 10
    assert module.hole_size.min == 1
    assert module.hole_size.max == 10
    assert module.angle == 45
    assert module.decay == .9
    assert module.speckle_accuracy == cellprofiler.modules.enhanceorsuppressfeatures.S_SLOW

    module = pipeline.modules()[1]
    assert isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures)
    assert module.enhance_method == cellprofiler.modules.enhanceorsuppressfeatures.E_DIC


def test_01_05_load_v4():
    data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20120516145742

EnhanceOrSuppressFeatures:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
Select the input image:Dendrite
Name the output image:EnhancedDendrite
Select the operation:Enhance
Feature size:10
Feature type:Neurites
Range of hole sizes:1,10
Smoothing scale:2.0
Shear angle:0
Decay:0.95
Enhancement method:Tubeness

EnhanceOrSuppressFeatures:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
Select the input image:Axon
Name the output image:EnhancedAxon
Select the operation:Enhance
Feature size:10
Feature type:Neurites
Range of hole sizes:1,10
Smoothing scale:2.0
Shear angle:0
Decay:0.95
Enhancement method:Line structures
'''
    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(StringIO.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures)
    assert module.x_name == "Dendrite"
    assert module.y_name == "EnhancedDendrite"
    assert module.method == cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
    assert module.enhance_method == cellprofiler.modules.enhanceorsuppressfeatures.E_NEURITES
    assert module.smoothing == 2.0
    assert module.object_size == 10
    assert module.hole_size.min == 1
    assert module.hole_size.max == 10
    assert module.angle == 0
    assert module.decay == .95
    assert module.neurite_choice == cellprofiler.modules.enhanceorsuppressfeatures.N_TUBENESS

    module = pipeline.modules()[1]
    assert isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures)
    assert module.neurite_choice == cellprofiler.modules.enhanceorsuppressfeatures.N_GRADIENT


def test_load_v5():
    data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150414135713
GitHash:3bad577
ModuleCount:2
HasImagePlaneDetails:False

EnhanceOrSuppressFeatures:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
Select the input image:Dendrite
Name the output image:EnhancedDendrite
Select the operation:Enhance
Feature size:10
Feature type:Neurites
Range of hole sizes:1,10
Smoothing scale:2.0
Shear angle:0
Decay:0.95
Enhancement method:Tubeness
Speed and accuracy:Slow / circular

EnhanceOrSuppressFeatures:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
Select the input image:Axon
Name the output image:EnhancedAxon
Select the operation:Enhance
Feature size:10
Feature type:Neurites
Range of hole sizes:1,10
Smoothing scale:2.0
Shear angle:0
Decay:0.95
Enhancement method:Line structures
Speed and accuracy:Fast / hexagonal
'''
    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(StringIO.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures)
    assert module.x_name == "Dendrite"
    assert module.y_name == "EnhancedDendrite"
    assert module.method == cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
    assert module.enhance_method == cellprofiler.modules.enhanceorsuppressfeatures.E_NEURITES
    assert module.smoothing == 2.0
    assert module.object_size == 10
    assert module.hole_size.min == 1
    assert module.hole_size.max == 10
    assert module.angle == 0
    assert module.decay == .95
    assert module.neurite_choice == cellprofiler.modules.enhanceorsuppressfeatures.N_TUBENESS
    assert module.speckle_accuracy == cellprofiler.modules.enhanceorsuppressfeatures.S_SLOW

    module = pipeline.modules()[1]
    assert isinstance(module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures)
    assert module.speckle_accuracy == cellprofiler.modules.enhanceorsuppressfeatures.S_FAST

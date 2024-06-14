import io
import os.path

import centrosome.filter
import numpy
import pytest
import scipy.ndimage
import skimage.exposure
import skimage.filters
import skimage.transform


import tests.frontend.modules
import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.module
import cellprofiler.modules.enhanceorsuppressfeatures
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

import tests.frontend


@pytest.fixture(scope="function")
def image():
    return cellprofiler_core.image.Image()


@pytest.fixture(scope="function")
def module():
    module = cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures()

    module.x_name.value = "input"

    module.y_name.value = "output"

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


@pytest.fixture(params=["Slow", "Fast"], scope="function")
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

    mask = numpy.ones_like(data, dtype=bool)

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

    mask = numpy.ones_like(data, dtype=bool)

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

    mask = numpy.ones_like(data, dtype=bool)

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

    mask = numpy.ones_like(data, dtype=bool)

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
    resources = os.path.realpath(
        os.path.join(os.path.dirname(tests.frontend.__file__), "resources")
    )

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
    resources = os.path.realpath(
        os.path.join(os.path.dirname(tests.frontend.__file__), "resources")
    )

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

    module.wants_rescale.value = True

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    expected = centrosome.filter.hessian(
        scipy.ndimage.gaussian_filter(data, 1.0),
        return_hessian=False,
        return_eigenvectors=False,
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

    module.wants_rescale.value = True

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    expected = centrosome.filter.hessian(
        scipy.ndimage.gaussian_filter(data, 1.0),
        return_hessian=False,
        return_eigenvectors=False,
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

    module.wants_rescale.value = True

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    smoothed = scipy.ndimage.gaussian_filter(data, 1.0)

    expected = numpy.zeros_like(smoothed)

    for index, plane in enumerate(smoothed):
        hessian = centrosome.filter.hessian(
            plane, return_hessian=False, return_eigenvectors=False
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

    module.wants_rescale.value = True

    module.run(workspace)

    output = workspace.image_set.get_image("output")

    actual = output.pixel_data

    smoothed = scipy.ndimage.gaussian_filter(data, 1.0)

    expected = numpy.zeros_like(smoothed)

    for index, plane in enumerate(smoothed):
        hessian = centrosome.filter.hessian(
            plane, return_hessian=False, return_eigenvectors=False
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

    assert numpy.all(actual[numpy.abs(numpy.sqrt(i * i + j * j) - 6) < 1.5] < 0.25)


def test_enhance_circles_masked(image, module, workspace):
    data = numpy.zeros((31, 62))

    mask = numpy.ones_like(data, dtype=bool)

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

    mask = numpy.ones_like(data, dtype=bool)

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

    module.enhance_method.value = (
        cellprofiler.modules.enhanceorsuppressfeatures.E_CIRCLES
    )

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

    gaussian_mask = skimage.filters.gaussian(
        numpy.ones_like(data), sigma, mode="constant"
    )

    gaussian = skimage.filters.gaussian(data, sigma, mode="constant") / gaussian_mask

    squared_gaussian = (
        skimage.filters.gaussian(data ** 2, sigma, mode="constant") / gaussian_mask
    )

    expected = squared_gaussian - gaussian ** 2

    actual = workspace.image_set.get_image("output").pixel_data

    numpy.testing.assert_almost_equal(expected, actual)


def test_enhance_texture_masked(image, module, workspace):
    r = numpy.random.RandomState()

    r.seed(81)

    data = r.uniform(size=(19, 24))

    mask = r.uniform(size=data.shape) > 0.25

    image.pixel_data = data

    image.mask = mask

    sigma = 2.1

    module.method.value = "Enhance"

    module.enhance_method.value = "Texture"

    module.smoothing.value = sigma

    module.run(workspace)

    masked_data = numpy.zeros_like(data)

    masked_data[mask] = data[mask]

    gaussian_mask = skimage.filters.gaussian(mask, sigma, mode="constant")

    gaussian = (
        skimage.filters.gaussian(masked_data, sigma, mode="constant") / gaussian_mask
    )

    squared_gaussian = (
        skimage.filters.gaussian(masked_data ** 2, sigma, mode="constant")
        / gaussian_mask
    )

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

    gaussian_mask = skimage.filters.gaussian(
        numpy.ones_like(data), sigma, mode="constant"
    )

    gaussian = skimage.filters.gaussian(data, sigma, mode="constant") / gaussian_mask

    squared_gaussian = (
        skimage.filters.gaussian(data ** 2, sigma, mode="constant") / gaussian_mask
    )

    expected = squared_gaussian - gaussian ** 2

    actual = workspace.image_set.get_image("output").pixel_data

    numpy.testing.assert_almost_equal(expected, actual)


def test_enhance_texture_masked_volume(image, module, workspace):
    r = numpy.random.RandomState()

    r.seed(81)

    data = r.uniform(size=(8, 19, 24))

    mask = r.uniform(size=data.shape) > 0.25

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

    gaussian_mask = skimage.filters.gaussian(mask, sigma, mode="constant")

    gaussian = (
        skimage.filters.gaussian(masked_data, sigma, mode="constant") / gaussian_mask
    )

    squared_gaussian = (
        skimage.filters.gaussian(masked_data ** 2, sigma, mode="constant")
        / gaussian_mask
    )

    expected = squared_gaussian - gaussian ** 2

    expected[~mask] = data[~mask]

    actual = workspace.image_set.get_image("output").pixel_data

    numpy.testing.assert_almost_equal(actual[mask], expected[mask])


def test_enhance_dic(image, module, workspace):
    data = numpy.ones((21, 43)) * 0.5

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

    expected[5:15, 10] = 0.5

    expected[5:15, 11:15] = 1

    expected[5:15, 15] = 0.5

    numpy.testing.assert_almost_equal(actual, expected)

    module.decay.value = 0.9

    module.run(workspace)

    actual = workspace.image_set.get_image("output").pixel_data

    assert numpy.all(actual[5:15, 12:14] < 1)

    module.decay.value = 1

    module.smoothing.value = 1

    actual = workspace.image_set.get_image("output").pixel_data

    assert numpy.all(actual[4, 11:15] > 0.1)


def test_load_v2():
    file = tests.frontend.modules.get_test_resources_directory(
        "enhanceorsuppressfeatures/v2.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 5
    for (
        module,
        (
            input_name,
            output_name,
            operation,
            feature_size,
            feature_type,
            min_range,
            max_range,
        ),
    ) in zip(
        pipeline.modules(),
        (
            (
                "Initial",
                "EnhancedSpeckles",
                cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE,
                11,
                cellprofiler.modules.enhanceorsuppressfeatures.E_SPECKLES,
                1,
                10,
            ),
            (
                "EnhancedSpeckles",
                "EnhancedNeurites",
                cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE,
                9,
                cellprofiler.modules.enhanceorsuppressfeatures.E_NEURITES,
                1,
                10,
            ),
            (
                "EnhancedNeurites",
                "EnhancedDarkHoles",
                cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE,
                9,
                cellprofiler.modules.enhanceorsuppressfeatures.E_DARK_HOLES,
                4,
                11,
            ),
            (
                "EnhancedDarkHoles",
                "EnhancedCircles",
                cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE,
                9,
                cellprofiler.modules.enhanceorsuppressfeatures.E_CIRCLES,
                4,
                11,
            ),
            (
                "EnhancedCircles",
                "Suppressed",
                cellprofiler.modules.enhanceorsuppressfeatures.SUPPRESS,
                13,
                cellprofiler.modules.enhanceorsuppressfeatures.E_CIRCLES,
                4,
                11,
            ),
        ),
    ):
        assert module.module_name == "EnhanceOrSuppressFeatures"
        assert isinstance(
            module,
            cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures,
        )
        assert module.x_name == input_name
        assert module.y_name == output_name
        assert module.method == operation
        assert module.enhance_method == feature_type
        assert module.object_size == feature_size
        assert module.hole_size.min == min_range
        assert module.hole_size.max == max_range


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory(
        "enhanceorsuppressfeatures/v3.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures
    )
    assert module.x_name == "DNA"
    assert module.y_name == "EnhancedTexture"
    assert module.method == cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
    assert (
        module.enhance_method
        == cellprofiler.modules.enhanceorsuppressfeatures.E_TEXTURE
    )
    assert module.smoothing == 3.5
    assert module.object_size == 10
    assert module.hole_size.min == 1
    assert module.hole_size.max == 10
    assert module.angle == 45
    assert module.decay == 0.9
    assert (
        module.speckle_accuracy == cellprofiler.modules.enhanceorsuppressfeatures.S_SLOW
    )

    module = pipeline.modules()[1]
    assert isinstance(
        module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures
    )
    assert module.enhance_method == cellprofiler.modules.enhanceorsuppressfeatures.E_DIC


def test_load_v4():
    file = tests.frontend.modules.get_test_resources_directory(
        "enhanceorsuppressfeatures/v4.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures
    )
    assert module.x_name == "Dendrite"
    assert module.y_name == "EnhancedDendrite"
    assert module.method == cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
    assert (
        module.enhance_method
        == cellprofiler.modules.enhanceorsuppressfeatures.E_NEURITES
    )
    assert module.smoothing == 2.0
    assert module.object_size == 10
    assert module.hole_size.min == 1
    assert module.hole_size.max == 10
    assert module.angle == 0
    assert module.decay == 0.95
    assert (
        module.neurite_choice
        == cellprofiler.modules.enhanceorsuppressfeatures.N_TUBENESS
    )

    module = pipeline.modules()[1]
    assert isinstance(
        module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures
    )
    assert (
        module.neurite_choice
        == cellprofiler.modules.enhanceorsuppressfeatures.N_GRADIENT
    )


def test_load_v5():
    file = tests.frontend.modules.get_test_resources_directory(
        "enhanceorsuppressfeatures/v5.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures
    )
    assert module.x_name == "Dendrite"
    assert module.y_name == "EnhancedDendrite"
    assert module.method == cellprofiler.modules.enhanceorsuppressfeatures.ENHANCE
    assert (
        module.enhance_method
        == cellprofiler.modules.enhanceorsuppressfeatures.E_NEURITES
    )
    assert module.smoothing == 2.0
    assert module.object_size == 10
    assert module.hole_size.min == 1
    assert module.hole_size.max == 10
    assert module.angle == 0
    assert module.decay == 0.95
    assert (
        module.neurite_choice
        == cellprofiler.modules.enhanceorsuppressfeatures.N_TUBENESS
    )
    assert (
        module.speckle_accuracy.value
        == cellprofiler.modules.enhanceorsuppressfeatures.S_SLOW
    )

    module = pipeline.modules()[1]
    assert isinstance(
        module, cellprofiler.modules.enhanceorsuppressfeatures.EnhanceOrSuppressFeatures
    )
    assert (
        module.speckle_accuracy == cellprofiler.modules.enhanceorsuppressfeatures.S_FAST
    )

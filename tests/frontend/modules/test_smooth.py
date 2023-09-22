import io

import centrosome.filter
import centrosome.smooth
import numpy
import scipy.ndimage
import skimage.restoration

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler.modules.smooth
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

INPUT_IMAGE_NAME = "myimage"
OUTPUT_IMAGE_NAME = "myfilteredimage"


def make_workspace(image, mask):
    """Make a workspace for testing FilterByObjectMeasurement"""
    module = cellprofiler.modules.smooth.Smooth()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    object_set = cellprofiler_core.object.ObjectSet()
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    image_set.add(INPUT_IMAGE_NAME, cellprofiler_core.image.Image(image, mask))
    module.image_name.value = INPUT_IMAGE_NAME
    module.filtered_image_name.value = OUTPUT_IMAGE_NAME
    return workspace, module


def test_load_v02():
    file = tests.frontend.modules.get_test_resources_directory("smooth/v2.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    smooth = pipeline.modules()[0]
    assert isinstance(smooth, cellprofiler.modules.smooth.Smooth)
    assert smooth.image_name == "InputImage"
    assert smooth.filtered_image_name == "OutputImage"
    assert smooth.wants_automatic_object_size
    assert smooth.object_size == 19
    assert smooth.smoothing_method == cellprofiler.modules.smooth.MEDIAN_FILTER
    assert not smooth.clip


def test_fit_polynomial():
    """Test the smooth module with polynomial fitting"""
    numpy.random.seed(0)
    #
    # Make an image that has a single sinusoidal cycle with different
    # phase in i and j. Make it a little out-of-bounds to start to test
    # clipping
    #
    i, j = numpy.mgrid[0:100, 0:100].astype(float) * numpy.pi / 50
    image = (numpy.sin(i) + numpy.cos(j)) / 1.8 + 0.9
    image += numpy.random.uniform(size=(100, 100)) * 0.1
    mask = numpy.ones(image.shape, bool)
    mask[40:60, 45:65] = False
    for clip in (False, True):
        expected = centrosome.smooth.fit_polynomial(image, mask, clip)
        assert numpy.all((expected >= 0) & (expected <= 1)) == clip
        workspace, module = make_workspace(image, mask)
        module.smoothing_method.value = cellprofiler.modules.smooth.FIT_POLYNOMIAL
        module.clip.value = clip
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        assert not (result is None)
        numpy.testing.assert_almost_equal(result.pixel_data, expected)


def test_gaussian_auto_small():
    """Test the smooth module with Gaussian smoothing in automatic mode"""
    sigma = 100.0 / 40.0 / 2.35
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(100, 100)).astype(numpy.float32)
    mask = numpy.ones(image.shape, bool)
    mask[40:60, 45:65] = False
    fn = lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0.0)
    expected = centrosome.smooth.smooth_with_function_and_mask(image, fn, mask)
    workspace, module = make_workspace(image, mask)
    module.smoothing_method.value = cellprofiler.modules.smooth.GAUSSIAN_FILTER
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert not (result is None)
    numpy.testing.assert_almost_equal(result.pixel_data, expected)


def test_gaussian_auto_large():
    """Test the smooth module with Gaussian smoothing in large automatic mode"""
    sigma = 30.0 / 2.35
    image = numpy.random.uniform(size=(3200, 100)).astype(numpy.float32)
    mask = numpy.ones(image.shape, bool)
    mask[40:60, 45:65] = False
    fn = lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0.0)
    expected = centrosome.smooth.smooth_with_function_and_mask(image, fn, mask)
    workspace, module = make_workspace(image, mask)
    module.smoothing_method.value = cellprofiler.modules.smooth.GAUSSIAN_FILTER
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert not (result is None)
    numpy.testing.assert_almost_equal(result.pixel_data, expected)


def test_gaussian_manual():
    """Test the smooth module with Gaussian smoothing, manual sigma"""
    sigma = 15.0 / 2.35
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(100, 100)).astype(numpy.float32)
    mask = numpy.ones(image.shape, bool)
    mask[40:60, 45:65] = False
    fn = lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0.0)
    expected = centrosome.smooth.smooth_with_function_and_mask(image, fn, mask)
    workspace, module = make_workspace(image, mask)
    module.smoothing_method.value = cellprofiler.modules.smooth.GAUSSIAN_FILTER
    module.wants_automatic_object_size.value = False
    module.object_size.value = 15.0
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert not (result is None)
    numpy.testing.assert_almost_equal(result.pixel_data, expected)


def test_median():
    """test the smooth module with median filtering"""
    object_size = 100.0 / 40.0
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(100, 100)).astype(numpy.float32)
    mask = numpy.ones(image.shape, bool)
    mask[40:60, 45:65] = False
    expected = centrosome.filter.median_filter(image, mask, object_size / 2 + 1)
    workspace, module = make_workspace(image, mask)
    module.smoothing_method.value = cellprofiler.modules.smooth.MEDIAN_FILTER
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert not (result is None)
    numpy.testing.assert_almost_equal(result.pixel_data, expected)


def test_bilateral():
    """test the smooth module with bilateral filtering"""
    sigma = 16.0
    sigma_range = 0.2
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(100, 100)).astype(numpy.float32)
    mask = numpy.ones(image.shape, bool)
    mask[40:60, 45:65] = False
    expected = skimage.restoration.denoise_bilateral(
        image=image.astype(float),
        sigma_color=sigma_range,
        sigma_spatial=sigma,
    )
    workspace, module = make_workspace(image, mask)
    module.smoothing_method.value = cellprofiler.modules.smooth.SMOOTH_KEEPING_EDGES
    module.sigma_range.value = sigma_range
    module.wants_automatic_object_size.value = False
    module.object_size.value = 16.0 * 2.35
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert not (result is None)
    numpy.testing.assert_almost_equal(result.pixel_data, expected)

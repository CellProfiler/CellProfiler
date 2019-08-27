import base64
import io
import zlib

import centrosome.filter
import centrosome.smooth
import numpy
import scipy.ndimage
import skimage.restoration

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.smooth
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.workspace

INPUT_IMAGE_NAME = "myimage"
OUTPUT_IMAGE_NAME = "myfilteredimage"


def make_workspace(image, mask):
    """Make a workspace for testing FilterByObjectMeasurement"""
    module = cellprofiler.modules.smooth.Smooth()
    pipeline = cellprofiler.pipeline.Pipeline()
    object_set = cellprofiler.object.ObjectSet()
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    workspace = cellprofiler.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler.measurement.Measurements(),
        image_set_list,
    )
    image_set.add(INPUT_IMAGE_NAME, cellprofiler.image.Image(image, mask))
    module.image_name.value = INPUT_IMAGE_NAME
    module.filtered_image_name.value = OUTPUT_IMAGE_NAME
    return workspace, module


def test_load_v01():
    data = base64.b64decode(
        "eJztWN1u0zAUdrqsbCDt5wourV0hxKJ0iGr0hnUrE5GWrqLVBHd4rdtZcuLKcaaWJ"
        "+CSR+JxuNwjEJekSUxo0mxMqlRXlnvs853P57MT17WbvYvmKXxrmNBu9g6HhGLYoU"
        "gMGXca0BWv4RnHSOABZG4DnnMCbTSF5jGs1RtmvVE7gkem+Q6UK5pl7wTNj30AqkG"
        "7FdRKOLQZ2lqiSruLhSDuyNsEOngR9v8M6hXiBF1TfIWoj72YIuq33CHrTcfzIZsN"
        "fIrbyEk6B6XtO9eYe5fDCBgOd8gE0y75hpUUIrdP+JZ4hLkhPoyv9s55mVB4pQ7mk"
        "1gHTdFB6rKb6Jf+H0Hsr2fotp/w3wtt4g7ILRn4iELioNF8FjLecU68LSWetC85GZ"
        "0Gkkv8SQ5+T8HL2sMTcfhhgvoCOkj0b8rO44xxHs3DzMFrKbwG3oT5581/R+GV9jk"
        "RsMPo1GUOQbRYnKdKHGm3GHSZgL6H4/XIy2MjFWcDfAlWswiuksJVQJsV49NTOB3U"
        "6oZZZB8+V/KVdgsPkU8FtOQmhC3CcV8wPi2Vt2nUCuHUdTcydK4quKhEuO2wfSidV"
        "5lP3Q+tZse6D9993z+rkueyfG3mPmp+q8b3q7rcuVmW5yQnr6z3+uyQHXHmj/8/f9"
        "b5GvPD4OjH44da1zVujVvj1s/xGvf4uLsETj3v1N+B0v8rWLzfXoH0fpN2H1M65kz"
        "+H8ANZ3Zp9QzK0ODPrdG4CL5aiQuk5Pmcw3Og8Bz8i8dzGBM3RnfWZOu1nRE/mXcl"
        "+OxWF+us6hvrfve+DF9F+5vvWQ5OD5WSuO9guXV9ucA/yq2s/29KodH7"
    )
    data = zlib.decompress(data)
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 2
    smooth = pipeline.modules()[1]
    assert smooth.module_name == "Smooth"
    assert smooth.image_name.value == "OrigBlue"
    assert smooth.filtered_image_name.value == "CorrBlue"
    assert smooth.smoothing_method.value == cellprofiler.modules.smooth.FIT_POLYNOMIAL
    assert smooth.wants_automatic_object_size
    assert smooth.clip


def test_load_v02():
    with open("./tests/resources/modules/smooth/v2.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()
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
        image=image.astype(numpy.float),
        multichannel=False,
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

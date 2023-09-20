import centrosome.filter
import centrosome.kirsch
import centrosome.otsu
import numpy

import cellprofiler_core.image
import cellprofiler_core.measurement


import cellprofiler.modules.enhanceedges
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"


def make_workspace(image, mask=None):
    """Make a workspace for testing FindEdges"""
    module = cellprofiler.modules.enhanceedges.FindEdges()
    module.image_name.value = INPUT_IMAGE_NAME
    module.output_image_name.value = OUTPUT_IMAGE_NAME
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
    image_set.add(
        INPUT_IMAGE_NAME,
        cellprofiler_core.image.Image(image)
        if mask is None
        else cellprofiler_core.image.Image(image, mask),
    )
    return workspace, module


def test_sobel_horizontal():
    """Test the Sobel horizontal transform"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_SOBEL
    module.direction.value = cellprofiler.modules.enhanceedges.E_HORIZONTAL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == centrosome.filter.hsobel(image))


def test_sobel_vertical():
    """Test the Sobel vertical transform"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_SOBEL
    module.direction.value = cellprofiler.modules.enhanceedges.E_VERTICAL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == centrosome.filter.vsobel(image))


def test_sobel_all():
    """Test the Sobel transform"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_SOBEL
    module.direction.value = cellprofiler.modules.enhanceedges.E_ALL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == centrosome.filter.sobel(image))


def test_prewitt_horizontal():
    """Test the prewitt horizontal transform"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_PREWITT
    module.direction.value = cellprofiler.modules.enhanceedges.E_HORIZONTAL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == centrosome.filter.hprewitt(image))


def test_prewitt_vertical():
    """Test the prewitt vertical transform"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_PREWITT
    module.direction.value = cellprofiler.modules.enhanceedges.E_VERTICAL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == centrosome.filter.vprewitt(image))


def test_prewitt_all():
    """Test the prewitt transform"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_PREWITT
    module.direction.value = cellprofiler.modules.enhanceedges.E_ALL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == centrosome.filter.prewitt(image))


def test_roberts():
    """Test the roberts transform"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_ROBERTS
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == centrosome.filter.roberts(image))


def test_log_automatic():
    """Test the laplacian of gaussian with automatic sigma"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_LOG
    module.sigma.value = 20
    module.wants_automatic_sigma.value = True
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    sigma = 2.0
    expected = centrosome.filter.laplacian_of_gaussian(
        image, numpy.ones(image.shape, bool), int(sigma * 4) + 1, sigma
    ).astype(numpy.float32)

    assert numpy.all(output.pixel_data == expected)


def test_log_manual():
    """Test the laplacian of gaussian with manual sigma"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_LOG
    module.sigma.value = 4
    module.wants_automatic_sigma.value = False
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    sigma = 4.0
    expected = centrosome.filter.laplacian_of_gaussian(
        image, numpy.ones(image.shape, bool), int(sigma * 4) + 1, sigma
    ).astype(numpy.float32)

    assert numpy.all(output.pixel_data == expected)


def test_canny():
    """Test the canny method"""
    i, j = numpy.mgrid[-20:20, -20:20]
    image = numpy.logical_and(i > j, i ** 2 + j ** 2 < 300).astype(numpy.float32)
    numpy.random.seed(0)
    image = image * 0.5 + numpy.random.uniform(size=image.shape) * 0.3
    image = numpy.ascontiguousarray(image, numpy.float32)
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_CANNY
    module.wants_automatic_threshold.value = True
    module.wants_automatic_low_threshold.value = True
    module.wants_automatic_sigma.value = True
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    t1, t2 = centrosome.otsu.otsu3(centrosome.filter.sobel(image))
    result = centrosome.filter.canny(image, numpy.ones(image.shape, bool), 1.0, t1, t2)
    assert numpy.all(output.pixel_data == result)


def test_kirsch():
    r = numpy.random.RandomState([ord(_) for _ in "test_07_01_kirsch"])
    i, j = numpy.mgrid[-20:20, -20:20]
    image = (numpy.sqrt(i * i + j * j) <= 10).astype(float) * 0.5
    image = image + r.uniform(size=image.shape) * 0.1
    workspace, module = make_workspace(image)
    module.method.value = cellprofiler.modules.enhanceedges.M_KIRSCH
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    result = centrosome.kirsch.kirsch(image)
    numpy.testing.assert_almost_equal(output.pixel_data, result, decimal=4)

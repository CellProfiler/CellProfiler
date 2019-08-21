"""test_enhanceedges - test the EnhanceEdges module
"""

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.modules.enhanceedges as F
import centrosome.filter as FIL
from centrosome.kirsch import kirsch
from centrosome.otsu import otsu3

INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"


def make_workspace(self, image, mask=None):
    """Make a workspace for testing FindEdges"""
    module = F.FindEdges()
    module.image_name.value = INPUT_IMAGE_NAME
    module.output_image_name.value = OUTPUT_IMAGE_NAME
    pipeline = cpp.Pipeline()
    object_set = cpo.ObjectSet()
    image_set_list = cpi.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    workspace = cpw.Workspace(
        pipeline, module, image_set, object_set, cpmeas.Measurements(), image_set_list
    )
    image_set.add(
        INPUT_IMAGE_NAME, cpi.Image(image) if mask is None else cpi.Image(image, mask)
    )
    return workspace, module


def test_sobel_horizontal(self):
    """Test the Sobel horizontal transform"""
    np.random.seed(0)
    image = np.random.uniform(size=(20, 20)).astype(np.float32)
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_SOBEL
    module.direction.value = F.E_HORIZONTAL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert np.all(output.pixel_data == FIL.hsobel(image))


def test_sobel_vertical(self):
    """Test the Sobel vertical transform"""
    np.random.seed(0)
    image = np.random.uniform(size=(20, 20)).astype(np.float32)
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_SOBEL
    module.direction.value = F.E_VERTICAL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert np.all(output.pixel_data == FIL.vsobel(image))


def test_sobel_all(self):
    """Test the Sobel transform"""
    np.random.seed(0)
    image = np.random.uniform(size=(20, 20)).astype(np.float32)
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_SOBEL
    module.direction.value = F.E_ALL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert np.all(output.pixel_data == FIL.sobel(image))


def test_prewitt_horizontal(self):
    """Test the prewitt horizontal transform"""
    np.random.seed(0)
    image = np.random.uniform(size=(20, 20)).astype(np.float32)
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_PREWITT
    module.direction.value = F.E_HORIZONTAL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert np.all(output.pixel_data == FIL.hprewitt(image))


def test_prewitt_vertical(self):
    """Test the prewitt vertical transform"""
    np.random.seed(0)
    image = np.random.uniform(size=(20, 20)).astype(np.float32)
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_PREWITT
    module.direction.value = F.E_VERTICAL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert np.all(output.pixel_data == FIL.vprewitt(image))


def test_prewitt_all(self):
    """Test the prewitt transform"""
    np.random.seed(0)
    image = np.random.uniform(size=(20, 20)).astype(np.float32)
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_PREWITT
    module.direction.value = F.E_ALL
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert np.all(output.pixel_data == FIL.prewitt(image))


def test_roberts(self):
    """Test the roberts transform"""
    np.random.seed(0)
    image = np.random.uniform(size=(20, 20)).astype(np.float32)
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_ROBERTS
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert np.all(output.pixel_data == FIL.roberts(image))


def test_log_automatic(self):
    """Test the laplacian of gaussian with automatic sigma"""
    np.random.seed(0)
    image = np.random.uniform(size=(20, 20)).astype(np.float32)
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_LOG
    module.sigma.value = 20
    module.wants_automatic_sigma.value = True
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    sigma = 2.0
    expected = FIL.laplacian_of_gaussian(
        image, np.ones(image.shape, bool), int(sigma * 4) + 1, sigma
    ).astype(np.float32)

    assert np.all(output.pixel_data == expected)


def test_log_manual(self):
    """Test the laplacian of gaussian with manual sigma"""
    np.random.seed(0)
    image = np.random.uniform(size=(20, 20)).astype(np.float32)
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_LOG
    module.sigma.value = 4
    module.wants_automatic_sigma.value = False
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    sigma = 4.0
    expected = FIL.laplacian_of_gaussian(
        image, np.ones(image.shape, bool), int(sigma * 4) + 1, sigma
    ).astype(np.float32)

    assert np.all(output.pixel_data == expected)


def test_canny(self):
    """Test the canny method"""
    i, j = np.mgrid[-20:20, -20:20]
    image = np.logical_and(i > j, i ** 2 + j ** 2 < 300).astype(np.float32)
    np.random.seed(0)
    image = image * 0.5 + np.random.uniform(size=image.shape) * 0.3
    image = np.ascontiguousarray(image, np.float32)
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_CANNY
    module.wants_automatic_threshold.value = True
    module.wants_automatic_low_threshold.value = True
    module.wants_automatic_sigma.value = True
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    t1, t2 = otsu3(FIL.sobel(image))
    result = FIL.canny(image, np.ones(image.shape, bool), 1.0, t1, t2)
    assert np.all(output.pixel_data == result)


def test_kirsch(self):
    r = np.random.RandomState([ord(_) for _ in "test_07_01_kirsch"])
    i, j = np.mgrid[-20:20, -20:20]
    image = (np.sqrt(i * i + j * j) <= 10).astype(float) * 0.5
    image = image + r.uniform(size=image.shape) * 0.1
    workspace, module = self.make_workspace(image)
    module.method.value = F.M_KIRSCH
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    result = kirsch(image)
    np.testing.assert_almost_equal(output.pixel_data, result, decimal=4)

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.workspace
import skimage.data
import pytest


@pytest.fixture(scope="module")
def image():
    example = skimage.data.camera()

    return cellprofiler.image.Image(example)


@pytest.fixture(scope="module")
def image_set(image, image_set_list):
    image_set = image_set_list.get_image_set(0)

    image_set.add("example", image)

    return image_set


@pytest.fixture(scope="module")
def image_set_list():
    return cellprofiler.image.ImageSetList()


@pytest.fixture(scope="module")
def measurements():
    return cellprofiler.measurement.Measurements()


@pytest.fixture(scope="module")
def module(request):
    instance = getattr(request.module, "instance")

    return instance


@pytest.fixture(scope="module")
def object_set():
    return cellprofiler.object.ObjectSet()


@pytest.fixture(scope="module")
def pipeline():
    return cellprofiler.pipeline.Pipeline()


@pytest.fixture()
def workspace(pipeline, module, image_set, object_set, measurements, image_set_list):
    return cellprofiler.workspace.Workspace(pipeline, module, image_set, object_set, measurements, image_set_list)

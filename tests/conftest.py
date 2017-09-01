import numpy
import pytest
import skimage.data

import cellprofiler.__main__
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.utilities.cpjvm
import cellprofiler.workspace


def pytest_sessionstart(session):
    cellprofiler.preferences.set_headless()

    cellprofiler.utilities.cpjvm.cp_start_vm()


def pytest_sessionfinish(session, exitstatus):
    cellprofiler.__main__.stop_cellprofiler()

    return exitstatus


@pytest.fixture(
    scope="module",
    params=[
        (skimage.data.camera()[0:128, 0:128], 2),
        (skimage.data.astronaut()[0:128, 0:128, :], 2),
        (numpy.tile(skimage.data.camera()[0:32, 0:32], (2, 1)).reshape(2, 32, 32), 3)
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


@pytest.fixture(scope="function")
def image_set(image, image_set_list):
    image_set = image_set_list.get_image_set(0)

    image_set.add("example", image)

    return image_set


@pytest.fixture(scope="function")
def image_set_list():
    return cellprofiler.image.ImageSetList()


@pytest.fixture(scope="function")
def measurements():
    return cellprofiler.measurement.Measurements()


@pytest.fixture(scope="module")
def module(request):
    instance = getattr(request.module, "instance")

    return instance


@pytest.fixture(scope="function")
def object_set():
    return cellprofiler.object.ObjectSet()


@pytest.fixture(scope="function")
def pipeline():
    return cellprofiler.pipeline.Pipeline()


@pytest.fixture(scope="function")
def workspace(pipeline, module, image_set, object_set, measurements, image_set_list):
    return cellprofiler.workspace.Workspace(pipeline, module, image_set, object_set, measurements, image_set_list)

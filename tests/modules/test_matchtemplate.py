import os

import pytest
import skimage
import skimage.data
import skimage.feature
import skimage.io

import cellprofiler_core.image
import cellprofiler.measurement
import cellprofiler.modules.matchtemplate
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.workspace


@pytest.fixture()
def image_set(image_set_list):
    return image_set_list.get_image_set(0)


@pytest.fixture()
def image_set_list():
    return cellprofiler_core.image.ImageSetList()


@pytest.fixture()
def measurements():
    return cellprofiler.measurement.Measurements()


@pytest.fixture()
def module():
    return cellprofiler.modules.matchtemplate.MatchTemplate()


@pytest.fixture()
def object_set():
    return cellprofiler.object.ObjectSet()


@pytest.fixture()
def pipeline():
    return cellprofiler.pipeline.Pipeline()


@pytest.fixture()
def workspace(pipeline, module, image_set, object_set, measurements, image_set_list):
    return cellprofiler.workspace.Workspace(
        pipeline, module, image_set, object_set, measurements, image_set_list
    )


def test_run(module, image_set, workspace):
    coins = skimage.data.coins()

    image = cellprofiler_core.image.Image(coins)

    image_set.add("image", image)

    module.input_image_name.value = "image"

    module.output_image_name.value = "output"

    module.template_name.value = os.path.join(
        os.path.dirname(__file__), "../resources/template.png"
    )

    module.run(workspace)

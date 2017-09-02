import numpy
import numpy.random
import numpy.testing
import pytest
import skimage.measure
import skimage.morphology
import skimage.segmentation

import cellprofiler.image
import cellprofiler.modules.removeobjectsbysize
import cellprofiler.object

numpy.random.seed(23)

instance = cellprofiler.modules.removeobjectsbysize.RemoveObjectsBySize()


@pytest.fixture(scope="module")
def image():
    return cellprofiler.image.Image()


@pytest.fixture(
    scope="module",
    params=[
        skimage.measure.label(numpy.random.rand(10, 10) > 0.7),
        skimage.measure.label(numpy.random.rand(10, 10, 10) > 0.85)
    ],
    ids=[
        "image_labels",
        "volume_labels"
    ]
)
def objects(request):
    objects = cellprofiler.object.Objects()

    objects.segmented = request.param

    return objects


@pytest.fixture(scope="function")
def object_set(objects):
    object_set = cellprofiler.object.ObjectSet()

    object_set.add_objects(objects, "input")

    return object_set


def test_run_small(module, objects, workspace):
    module.x_name.value = "input"

    module.y_name.value = "output"

    module.size.value = (3.0, numpy.inf)

    module.run(workspace)

    actual = workspace.object_set.get_objects("output").segmented

    if objects.dimensions == 2:
        factor = ((3.0 / 2.0) ** 2)
    else:
        factor = (4.0 / 3.0) * ((3.0 / 2.0) ** 3)

    expected = skimage.morphology.remove_small_objects(objects.segmented, numpy.pi * factor)

    expected = skimage.segmentation.relabel_sequential(expected)[0]

    numpy.testing.assert_array_equal(actual, expected)


def test_run_large(module, objects, workspace):
    module.x_name.value = "input"

    module.y_name.value = "output"

    module.size.value = (0.0, 3.0)

    module.run(workspace)

    actual = workspace.object_set.get_objects("output").segmented

    if objects.dimensions == 2:
        factor = ((3.0 / 2.0) ** 2)
    else:
        factor = (4.0 / 3.0) * ((3.0 / 2.0) ** 3)

    expected = objects.segmented ^ skimage.morphology.remove_small_objects(objects.segmented, numpy.pi * factor)

    expected = skimage.segmentation.relabel_sequential(expected)[0]

    numpy.testing.assert_array_equal(actual, expected)


def test_run(module, objects, workspace):
    module.x_name.value = "input"

    module.y_name.value = "output"

    module.size.value = (1.0, 3.0)

    module.run(workspace)

    actual = workspace.object_set.get_objects("output").segmented

    if objects.dimensions == 2:
        min_factor = ((1.0 / 2.0) ** 2)

        max_factor = ((3.0 / 2.0) ** 2)
    else:
        min_factor = (4.0 / 3.0) * ((1.0 / 2.0) ** 3)

        max_factor = (4.0 / 3.0) * ((3.0 / 2.0) ** 3)

    expected = skimage.morphology.remove_small_objects(objects.segmented.copy(), numpy.pi * min_factor)

    expected ^= skimage.morphology.remove_small_objects(expected, numpy.pi * max_factor)

    expected = skimage.segmentation.relabel_sequential(expected)[0]

    numpy.testing.assert_array_equal(actual, expected)

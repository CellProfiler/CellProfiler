import numpy
import pytest
import scipy.sparse.coo

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.module


import cellprofiler.modules.convertobjectstoimage
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

OBJECTS_NAME = "inputobjects"
IMAGE_NAME = "outputimage"

instance = cellprofiler.modules.convertobjectstoimage.ConvertObjectsToImage()


@pytest.fixture(scope="function")
def image_set(image_set_list):
    image_set = image_set_list.get_image_set(0)

    return image_set


@pytest.fixture(scope="function")
def object_set(objects):
    object_set = cellprofiler_core.object.ObjectSet()

    object_set.add_objects(objects, "inputobjects")

    return object_set


@pytest.fixture(scope="module")
def image_segmentation():
    return numpy.reshape(numpy.arange(256), (16, 16))


@pytest.fixture(scope="module")
def volume_segmentation():
    segmentation = numpy.reshape(numpy.arange(256), (16, 16))

    segmentation = numpy.tile(segmentation, (3, 1))

    segmentation = numpy.reshape(segmentation, (3, 16, 16))

    return segmentation


@pytest.fixture(
    scope="module",
    params=["image_segmentation", "volume_segmentation"],
    ids=["image_segmentation", "volume_segmentation"],
)
def objects(request):
    objects = cellprofiler_core.object.Objects()

    objects.segmented = request.getfixturevalue(request.param)

    return objects


def test_run_binary(workspace, module):
    module.object_name.value = "inputobjects"

    module.image_name.value = "outputimage"

    module.image_mode.value = "Binary (black & white)"

    module.run(workspace)

    image = workspace.image_set.get_image("outputimage")

    objects = workspace.get_objects("inputobjects")

    assert image.dimensions == objects.segmented.ndim

    pixel_data = image.pixel_data

    assert pixel_data.shape == objects.shape

    if objects.segmented.ndim == 2:
        assert not pixel_data[0, 0]

        assert numpy.all(pixel_data[:, 1:])

        assert numpy.all(pixel_data[1:, :])
    else:
        assert not numpy.all(pixel_data[:, 0, 0])

        assert numpy.all(pixel_data[:, :, 1:])

        assert numpy.all(pixel_data[:, 1:, :])


def test_run_grayscale(workspace, module):
    module.object_name.value = "inputobjects"

    module.image_name.value = "outputimage"

    module.image_mode.value = "Grayscale"

    module.run(workspace)

    image = workspace.image_set.get_image("outputimage")

    objects = workspace.get_objects("inputobjects")

    assert image.dimensions == objects.segmented.ndim

    pixel_data = image.pixel_data

    assert pixel_data.shape == objects.shape

    expected = numpy.reshape(numpy.arange(256).astype(numpy.float32) / 255, (16, 16))

    if objects.segmented.ndim == 3:
        expected = numpy.tile(expected, (3, 1))

        expected = numpy.reshape(expected, (3, 16, 16))

    assert numpy.all(pixel_data == expected)


def test_run_color(workspace, module):
    for color in [
        "Default",
        "autumn",
        "bone",
        "colorcube",
        "cool",
        "copper",
        "flag",
        "gray",
        "hot",
        "hsv",
        "jet",
        "lines",
        "pink",
        "prism",
        "spring",
        "summer",
        "white",
        "winter",
    ]:
        module.object_name.value = "inputobjects"

        module.image_name.value = "outputimage"

        module.image_mode.value = "Color"

        module.colormap.value = color

        module.run(workspace)

        image = workspace.image_set.get_image("outputimage")

        objects = workspace.get_objects("inputobjects")

        assert image.dimensions == objects.segmented.ndim

        pixel_data = image.pixel_data

        assert pixel_data.shape == objects.shape + (3,)


def test_run_uint16(workspace, module):
    module.object_name.value = "inputobjects"

    module.image_name.value = "outputimage"

    module.image_mode.value = "uint16"

    module.run(workspace)

    image = workspace.image_set.get_image("outputimage")

    objects = workspace.get_objects("inputobjects")

    assert image.dimensions == objects.segmented.ndim

    pixel_data = image.pixel_data

    assert pixel_data.shape == objects.shape

    expected = numpy.reshape(numpy.arange(256), (16, 16))

    if objects.segmented.ndim == 3:
        expected = numpy.tile(expected, (3, 1))

        expected = numpy.reshape(expected, (3, 16, 16))

    assert numpy.all(pixel_data == expected)


def make_workspace_ijv():
    module = cellprofiler.modules.convertobjectstoimage.ConvertToImage()
    shape = (14, 16)
    r = numpy.random.RandomState()
    r.seed(0)
    i = r.randint(0, shape[0], size=numpy.prod(shape))
    j = r.randint(0, shape[1], size=numpy.prod(shape))
    v = r.randint(1, 8, size=numpy.prod(shape))
    order = numpy.lexsort((i, j, v))
    ijv = numpy.column_stack((i, j, v))
    ijv = ijv[order, :]
    same = numpy.all(ijv[:-1, :] == ijv[1:, :], 1)

    ijv = ijv[: numpy.prod(shape) - 1][~same, :]

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
    objects = cellprofiler_core.object.Objects()
    objects.set_ijv(ijv, shape)
    object_set.add_objects(objects, OBJECTS_NAME)
    assert len(objects.get_labels()) > 1
    module.image_name.value = IMAGE_NAME
    module.object_name.value = OBJECTS_NAME
    return workspace, module, ijv


def test_binary_ijv():
    workspace, module, ijv = make_workspace_ijv()
    assert isinstance(
        module, cellprofiler.modules.convertobjectstoimage.ConvertObjectsToImage
    )
    module.image_mode.value = "Binary (black & white)"
    module.run(workspace)
    pixel_data = workspace.image_set.get_image(IMAGE_NAME).pixel_data
    assert len(numpy.unique(ijv[:, 0] + ijv[:, 1] * pixel_data.shape[0])) == numpy.sum(
        pixel_data
    )
    assert numpy.all(pixel_data[ijv[:, 0], ijv[:, 1]])


def test_gray_ijv():
    workspace, module, ijv = make_workspace_ijv()
    assert isinstance(
        module, cellprofiler.modules.convertobjectstoimage.ConvertObjectsToImage
    )
    module.image_mode.value = "Grayscale"
    module.run(workspace)
    pixel_data = workspace.image_set.get_image(IMAGE_NAME).pixel_data

    counts = scipy.sparse.coo.coo_matrix(
        (numpy.ones(ijv.shape[0]), (ijv[:, 0], ijv[:, 1]))
    ).toarray()
    assert numpy.all(pixel_data[counts == 0] == 0)
    pd_values = numpy.unique(pixel_data)
    pd_labels = numpy.zeros(pixel_data.shape, int)
    for i in range(1, len(pd_values)):
        pd_labels[pixel_data == pd_values[i]] = i

    dest_v = numpy.zeros(numpy.max(ijv[:, 2] + 1), int)
    dest_v[ijv[:, 2]] = pd_labels[ijv[:, 0], ijv[:, 1]]
    pd_ok = numpy.zeros(pixel_data.shape, bool)
    ok = pd_labels[ijv[:, 0], ijv[:, 1]] == dest_v[ijv[:, 2]]
    pd_ok[ijv[ok, 0], ijv[ok, 1]] = True
    assert numpy.all(pd_ok[counts > 0])


def test_color_ijv():
    workspace, module, ijv = make_workspace_ijv()
    assert isinstance(
        module, cellprofiler.modules.convertobjectstoimage.ConvertObjectsToImage
    )
    module.image_mode.value = "Color"
    module.run(workspace)
    pixel_data = workspace.image_set.get_image(IMAGE_NAME).pixel_data
    #
    # convert the labels into individual bits (1, 2, 4, 8)
    # the labels matrix is a matrix of bits that are on
    #
    vbit = 2 ** (ijv[:, 2] - 1)
    vbit_color = numpy.zeros((numpy.max(vbit) * 2, 3))
    bits = scipy.sparse.coo.coo_matrix((vbit, (ijv[:, 0], ijv[:, 1]))).toarray()
    #
    # Get some color for every represented bit combo
    #
    vbit_color[bits[ijv[:, 0], ijv[:, 1]], :] = pixel_data[ijv[:, 0], ijv[:, 1], :]

    assert numpy.all(pixel_data == vbit_color[bits, :])

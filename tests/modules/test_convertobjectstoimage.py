import StringIO
import base64
import unittest
import zlib

import numpy
import pytest
import scipy.sparse.coo

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.convertobjectstoimage
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()

OBJECTS_NAME = "inputobjects"
IMAGE_NAME = "outputimage"

instance = cellprofiler.modules.convertobjectstoimage.ConvertObjectsToImage()


@pytest.fixture(scope="function")
def image_set(image_set_list):
    image_set = image_set_list.get_image_set(0)

    return image_set


@pytest.fixture(scope="function")
def object_set(objects):
    object_set = cellprofiler.object.ObjectSet()

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
    params=[
        image_segmentation(),
        volume_segmentation()
    ],
    ids=[
        "image_segmentation",
        "volume_segmentation"
    ]
)
def objects(request):
    objects = cellprofiler.object.Objects()

    objects.segmented = request.param

    return objects


def test_load_matlab(pipeline):
    data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0'
            'sSU1RyM+zUnArylTwKs1TMDRSMDS2Mja3MjBUMDIwsFQgGTAwevryMzAwrG'
            'BiYKiY8zbcMf+WgUiZgsI1v45A31uNzn3WJbI5l7c81rLK9ZAO0V5lkikuz'
            'PP1yQ2NQoE307/umL/p9b+jSR5Nj5kMPa+8M/Ce9nXO933f9z6fLr6cj2FH'
            'LlMA77UoA+8EW62FOR7BJ8I7in2Wx7HOeBC547/2jOsXjha4XDBVsfvkYLP'
            'or9zcDQ+LxOx/HTpm51j74sSpgy9+n+PYeS9BSLKtX/8j00TtP8KNTzqu7F'
            'tk+c1g8cLajvXvjtVJhv+51vzzd8ZkdueTzpYz1C/tu1DPb/br2bQ9pip/3'
            'SzaMw7G10Q4zI7me/Kt90Dm8gdThC1SCy7ax/+sn54UHvonPPznVs5DDqf3'
            'LbvBvbOaWdP1Sajgv8ufytfckD/+zZbdL7CGL/Lz9LR0tTc+H7598Y1Wf9d'
            '1KPpErNsM7qLneyer3st3XJveWe70IOT5HDnNmtY7sUVmyheKD9XK/l/K4s'
            'f55eYtvU3+jwuXfBD4xLpZ4/G094q1crs/hZv0m8zJ/Wr/YF5+rUoOi2dP8'
            'f2fhkIWbC9sBPi/zrm1fC5Lv/PjirWZy/epv91c9lzxVfLt6uuLZQL5Py89'
            'oh5lv/SBbeCHlp6XEuc/3mg/cmjRyyqhuIt/Nn1r17E5V3uuid9gm/GHgNf'
            'PdNztvn080WRe+vDSZkVHIb97ddOv2vXuf1e94LRydq3IWeez7LWc3Jbtey'
            'o73kg/+1e97g77p29bQy/P2HVm/svHOb/fbbzIn/KZN+DePrWHkY79zpuZH'
            '90pW3//yM42bSH33vLrv+7Hb/z376rp2nuzFWayKxrPF/a5OLdpemH2sXf7'
            'FnxbvubvA+Pmx/NKJl7ofL6ySfnJPYnAPvXT3uLWn8xD2sNjDwoXTb9oWfR'
            'zQuv+aVnz1y57sljE46yw8fW/PwPqpx4/Nvf/SRNr4ys/70lt+bflWl3+wc'
            'Tn/2r660zeef/XfzaV/REATdyFtA==')

    fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))

    pipeline.load(fd)

    assert len(pipeline.modules()) == 3

    module = pipeline.modules()[2]

    assert isinstance(module, cellprofiler.modules.convertobjectstoimage.ConvertToImage)

    assert module.object_name.value == "Nuclei"

    assert module.image_name.value == "NucleiImage"

    assert module.image_mode.value == "Color"

    assert module.colormap.value == "flag"


def test_load_v1(pipeline):
    data = ('eJztWltv2jAUNpeidpW6rnvYpL74cZvaKKFF2ngZFNaNqVy0ok57WxoM9WR'
            'slDgU9kv2U/a4n7SfsBhCSbzQcGm5SIlkwXH8ne+c4xM7OXI5X7/In8GMos'
            'Jyvn7cxATBGtF5k5ntLKT8CBZMpHPUgIxm4bmJ4WebQi0NtZOspmZPMzCtq'
            'u/AfFesVN5zfn69BiDl/G47Le7e2nLlmKcJ+RJxjmnL2gJJ8NLt/+O0K93E'
            '+jVBVzqxkTWmGPWXaJPV+527W2XWsAmq6G3vYOeq2O1rZFrV5gjo3q7hHiK'
            'X+CeSXBgN+4K62MKMunhXv9x7x8u4xCviAJ6P4xCT4pBw2qGnX4z/BMbjkw'
            'Fxe+YZv+/KmDZwFzdsnUDc1lt3Vgh9b0P0bUv6hFw1cevMCfk0+JSETw3ib'
            'RCEh/y5EPy+hBetjnr8+ENPNzhs69y4WYYdOxJeyAVESEkE1BNPNURPzKcn'
            'Bk6mnIctiV/Imnp0qi6ALzDCTDCd/08kvJCLDFLGoW2h6f1P+PQkwDcnG+e'
            'dv1tMOTKn44378HFQYQ+HS0m40TXC7YBxfMKe3xeSn0IuoqZuEw4HuQaL2E'
            'QGZ2Z/oXjPa/+s+a2A6fJrV/JbyFVu2fAjYdc6CeR/SLvnzQ/ZX21N7ZTzQ'
            'D3SHtXOpI8vCYr5WmkenKqo2iJ2Lrq/5ULwQftCyVmWqIV5f0K8lhHnTbO7'
            'wiiaJ/81dT3tlNeFTADfOtj5mPvDLLhciJ17wJ+vQh6+v1VtTjAVL7WrsHt'
            'T4hvZuVo7nX1sLZ+rZb53RXmwvnbK+6qSWc2850LsDMrX+i2DBtEty61srM'
            'LusO+6oLrMV4RbN6LM1hUFJWpMqiOsQ9yDvv/PmYlaJrNpY3H+7wez1cGW6'
            'eegaCYc7UyvJyhP2fUP58t9rGjZ+ebhh5g2UGeGuATV4cZxGapb9boW4SJc'
            'hItwm4zLeXDROhzhItxqcWHvWQfA/zwKmQ0rUv+9aG2S3xEuwq3Dfjft99i'
            'm+BvhIlyEWx2uFxvj5DqTXK8d1KU8PEHr0xvgX5+EbCBCOiYT5+pMpT04/G'
            'UphOmN4ekr5cL5W/IcxBI8nRCenMSTm8SDG4hy3Ox3TIfN5qytc2woJbe35'
            'vTmR72C9yaENy3xpifxGox2kck5GzipFIZinXkOSMnzthPA541/3JGeHibv'
            'nW8A/PM8nv+/7+fhiydiAz7vuYHdEFzSY9PIz99gtjx7dc/4kY/LGv8PmsR'
            'r2g==')

    fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))

    pipeline.load(fd)

    assert len(pipeline.modules()) == 3

    module = pipeline.modules()[2]

    assert isinstance(module, cellprofiler.modules.convertobjectstoimage.ConvertToImage)

    assert module.object_name.value == "Nuclei"

    assert module.image_name.value == "CellImage"

    assert module.image_mode.value == "Color"

    assert module.colormap.value == "winter"


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

    if objects.segmented.ndim is 2:
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

    if objects.segmented.ndim is 3:
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
        "winter"
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

    if objects.segmented.ndim is 3:
        expected = numpy.tile(expected, (3, 1))

        expected = numpy.reshape(expected, (3, 16, 16))

    assert numpy.all(pixel_data == expected)


class TestConvertObjectsToImage(unittest.TestCase):
    def make_workspace_ijv(self):
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
        ijv = ijv[~same, :]

        pipeline = cellprofiler.pipeline.Pipeline()
        object_set = cellprofiler.object.ObjectSet()
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     module,
                                                     image_set,
                                                     object_set,
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        objects = cellprofiler.object.Objects()
        objects.set_ijv(ijv, shape)
        object_set.add_objects(objects, OBJECTS_NAME)
        self.assertGreater(len(objects.get_labels()), 1)
        module.image_name.value = IMAGE_NAME
        module.object_name.value = OBJECTS_NAME
        return workspace, module, ijv

    def test_03_01_binary_ijv(self):
        workspace, module, ijv = self.make_workspace_ijv()
        self.assertTrue(isinstance(module, cellprofiler.modules.convertobjectstoimage.ConvertObjectsToImage))
        module.image_mode.value = "Binary (black & white)"
        module.run(workspace)
        pixel_data = workspace.image_set.get_image(IMAGE_NAME).pixel_data
        self.assertEqual(len(numpy.unique(ijv[:, 0] + ijv[:, 1] * pixel_data.shape[0])),
                         numpy.sum(pixel_data))
        self.assertTrue(numpy.all(pixel_data[ijv[:, 0], ijv[:, 1]]))

    def test_03_02_gray_ijv(self):
        workspace, module, ijv = self.make_workspace_ijv()
        self.assertTrue(isinstance(module, cellprofiler.modules.convertobjectstoimage.ConvertObjectsToImage))
        module.image_mode.value = "Grayscale"
        module.run(workspace)
        pixel_data = workspace.image_set.get_image(IMAGE_NAME).pixel_data

        counts = scipy.sparse.coo.coo_matrix((numpy.ones(ijv.shape[0]), (ijv[:, 0], ijv[:, 1]))).toarray()
        self.assertTrue(numpy.all(pixel_data[counts == 0] == 0))
        pd_values = numpy.unique(pixel_data)
        pd_labels = numpy.zeros(pixel_data.shape, int)
        for i in range(1, len(pd_values)):
            pd_labels[pixel_data == pd_values[i]] = i

        dest_v = numpy.zeros(numpy.max(ijv[:, 2] + 1), int)
        dest_v[ijv[:, 2]] = pd_labels[ijv[:, 0], ijv[:, 1]]
        pd_ok = numpy.zeros(pixel_data.shape, bool)
        ok = pd_labels[ijv[:, 0], ijv[:, 1]] == dest_v[ijv[:, 2]]
        pd_ok[ijv[ok, 0], ijv[ok, 1]] = True
        self.assertTrue(numpy.all(pd_ok[counts > 0]))

    def test_03_03_color_ijv(self):
        workspace, module, ijv = self.make_workspace_ijv()
        self.assertTrue(isinstance(module, cellprofiler.modules.convertobjectstoimage.ConvertObjectsToImage))
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

        self.assertTrue(numpy.all(pixel_data == vbit_color[bits, :]))

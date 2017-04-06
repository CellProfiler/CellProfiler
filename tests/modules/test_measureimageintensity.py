import StringIO
import base64

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.measureimageintensity
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace
import numpy
import numpy.testing
import pytest
import skimage.morphology

cellprofiler.preferences.set_headless()


@pytest.fixture(scope="function")
def image():
    return cellprofiler.image.Image()


@pytest.fixture(scope="function")
def objects(image):
    objects = cellprofiler.object.Objects()

    objects.parent_image = image

    return objects


@pytest.fixture(scope="function")
def measurements():
    return cellprofiler.measurement.Measurements()


@pytest.fixture(scope="function")
def module():
    module = cellprofiler.modules.measureimageintensity.MeasureImageIntensity()

    module.images[0].image_name.value = "image"

    module.images[0].object_name.value = "objects"

    return module


@pytest.fixture(scope="function")
def workspace(image, measurements, module, objects):
    image_set_list = cellprofiler.image.ImageSetList()

    image_set = image_set_list.get_image_set(0)

    image_set.add("image", image)

    object_set = cellprofiler.object.ObjectSet()

    object_set.add_objects(objects, "objects")

    return cellprofiler.workspace.Workspace(
        cellprofiler.pipeline.Pipeline(),
        module,
        image_set,
        object_set,
        measurements,
        image_set_list
    )


def test_volume_zeros(image, measurements, module, workspace):
    image.pixel_data = numpy.zeros((10, 10, 10))

    image.dimensions = 3

    module.run(workspace)

    expected = {
        "Intensity_TotalIntensity_image": 0.0,
        "Intensity_MeanIntensity_image": 0.0,
        "Intensity_MedianIntensity_image": 0.0,
        "Intensity_StdIntensity_image": 0.0,
        "Intensity_MADIntensity_image": 0.0,
        "Intensity_MaxIntensity_image": 0.0,
        "Intensity_MinIntensity_image": 0.0,
        "Intensity_TotalArea_image": 1000.0,
        "Intensity_PercentMaximal_image": 100.0,
        "Intensity_UpperQuartileIntensity_image": 0.0,
        "Intensity_LowerQuartileIntensity_image": 0.0
    }

    for feature, value in expected.iteritems():
        actual = measurements.get_current_measurement(
            cellprofiler.measurement.IMAGE,
            feature
        )

        assert actual == value, "{} expected {}, got {}".format(feature, value, actual)


def test_volume(image, measurements, module, workspace):
    image.set_image(skimage.morphology.ball(3), convert=False)

    image.dimensions = 3

    module.run(workspace)

    expected = {
        "Intensity_TotalIntensity_image": 123.0,
        "Intensity_MeanIntensity_image": 0.358600583090379,
        "Intensity_MedianIntensity_image": 0.0,
        "Intensity_StdIntensity_image": 0.47958962134059907,
        "Intensity_MADIntensity_image": 0.0,
        "Intensity_MaxIntensity_image": 1.0,
        "Intensity_MinIntensity_image": 0.0,
        "Intensity_TotalArea_image": 343.0,
        "Intensity_PercentMaximal_image": 35.8600583090379,
        "Intensity_UpperQuartileIntensity_image": 1.0,
        "Intensity_LowerQuartileIntensity_image": 0.0
    }

    for feature, value in expected.iteritems():
        actual = measurements.get_current_measurement(
            cellprofiler.measurement.IMAGE,
            feature
        )

        assert actual == value, "{} expected {}, got {}".format(feature, value, actual)


def test_volume_and_mask(image, measurements, module, workspace):
    mask = skimage.morphology.ball(3, dtype=numpy.bool)

    image.set_image(numpy.ones_like(mask, dtype=numpy.uint8), convert=False)

    image.mask = mask

    image.dimensions = 3

    module.run(workspace)

    expected = {
        "Intensity_TotalIntensity_image": 123.0,
        "Intensity_MeanIntensity_image": 1.0,
        "Intensity_MedianIntensity_image": 1.0,
        "Intensity_StdIntensity_image": 0.0,
        "Intensity_MADIntensity_image": 0.0,
        "Intensity_MaxIntensity_image": 1.0,
        "Intensity_MinIntensity_image": 1.0,
        "Intensity_TotalArea_image": 123.0,
        "Intensity_PercentMaximal_image": 100.0,
        "Intensity_UpperQuartileIntensity_image": 1.0,
        "Intensity_LowerQuartileIntensity_image": 1.0
    }

    for feature, value in expected.iteritems():
        actual = measurements.get_current_measurement(
            cellprofiler.measurement.IMAGE,
            feature
        )

        assert actual == value, "{} expected {}, got {}".format(feature, value, actual)


def test_volume_and_objects(image, measurements, module, objects, workspace):
    object_data = skimage.morphology.ball(3, dtype=numpy.uint8)

    image.set_image(numpy.ones_like(object_data, dtype=numpy.uint8), convert=False)

    image.dimensions = 3

    objects.segmented = object_data

    module.images[0].wants_objects.value = True

    module.run(workspace)

    expected = {
        "Intensity_TotalIntensity_image_objects": 123.0,
        "Intensity_MeanIntensity_image_objects": 1.0,
        "Intensity_MedianIntensity_image_objects": 1.0,
        "Intensity_StdIntensity_image_objects": 0.0,
        "Intensity_MADIntensity_image_objects": 0.0,
        "Intensity_MaxIntensity_image_objects": 1.0,
        "Intensity_MinIntensity_image_objects": 1.0,
        "Intensity_TotalArea_image_objects": 123.0,
        "Intensity_PercentMaximal_image_objects": 100.0,
        "Intensity_UpperQuartileIntensity_image_objects": 1.0,
        "Intensity_LowerQuartileIntensity_image_objects": 1.0
    }

    for feature, value in expected.iteritems():
        actual = measurements.get_current_measurement(
            cellprofiler.measurement.IMAGE,
            feature
        )

        assert actual == value, "{} expected {}, got {}".format(feature, value, actual)


def test_volume_and_objects_and_mask(image, measurements, module, objects, workspace):
    mask = skimage.morphology.ball(3, dtype=numpy.bool)

    image.set_image(numpy.ones_like(mask, dtype=numpy.uint8), convert=False)

    image.mask = mask

    image.dimensions = 3

    object_data = numpy.ones_like(mask, dtype=numpy.uint8)

    objects.segmented = object_data

    module.images[0].wants_objects.value = True

    module.run(workspace)

    expected = {
        "Intensity_TotalIntensity_image_objects": 123.0,
        "Intensity_MeanIntensity_image_objects": 1.0,
        "Intensity_MedianIntensity_image_objects": 1.0,
        "Intensity_StdIntensity_image_objects": 0.0,
        "Intensity_MADIntensity_image_objects": 0.0,
        "Intensity_MaxIntensity_image_objects": 1.0,
        "Intensity_MinIntensity_image_objects": 1.0,
        "Intensity_TotalArea_image_objects": 123.0,
        "Intensity_PercentMaximal_image_objects": 100.0,
        "Intensity_UpperQuartileIntensity_image_objects": 1.0,
        "Intensity_LowerQuartileIntensity_image_objects": 1.0
    }

    for feature, value in expected.iteritems():
        actual = measurements.get_current_measurement(
            cellprofiler.measurement.IMAGE,
            feature
        )

        assert actual == value, "{} expected {}, got {}".format(feature, value, actual)


def test_00_00_zeros(image, measurements, module, workspace):
    '''Test operation on a completely-masked image'''
    image.pixel_data = numpy.zeros((10, 10))

    image.mask = numpy.zeros((10, 10), bool)

    module.run(workspace)

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalArea_image") == 0

    assert len(measurements.get_object_names()) == 1

    assert measurements.get_object_names()[0] == cellprofiler.measurement.IMAGE

    columns = module.get_measurement_columns(workspace.pipeline)

    features = measurements.get_feature_names(cellprofiler.measurement.IMAGE)

    assert len(columns) == len(features)

    for column in columns:
        assert column[1] in features


def test_01_01_image(image, measurements, module, workspace):
    '''Test operation on a single unmasked image'''
    numpy.random.seed(0)

    pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * .99

    pixels[0:2, 0:2] = 1

    image.pixel_data = pixels

    module.run(workspace)

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalArea_image") == 100

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalIntensity_image") == numpy.sum(pixels)

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_MeanIntensity_image") == numpy.sum(pixels) / 100.0

    assert measurements.get_current_image_measurement('Intensity_MinIntensity_image') == numpy.min(pixels)

    assert measurements.get_current_image_measurement('Intensity_MaxIntensity_image') == numpy.max(pixels)

    assert measurements.get_current_image_measurement('Intensity_PercentMaximal_image') == 4.0


def test_01_02_image_and_mask(image, measurements, module, workspace):
    '''Test operation on a masked image'''
    numpy.random.seed(0)

    pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * .99

    pixels[1:3, 1:3] = 1

    mask = numpy.zeros((10, 10), bool)

    mask[1:9, 1:9] = True

    image.pixel_data = pixels

    image.mask = mask

    module.run(workspace)

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalArea_image") == 64

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalIntensity_image") == numpy.sum(pixels[1:9, 1:9])

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_MeanIntensity_image") == numpy.sum(pixels[1:9, 1:9]) / 64.0

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_PercentMaximal_image") == 400. / 64.


def test_01_03_image_and_objects(image, measurements, module, objects, workspace):
    '''Test operation on an image masked by objects'''
    numpy.random.seed(0)

    pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * .99

    pixels[1:3, 1:3] = 1

    image.pixel_data = pixels

    labels = numpy.zeros((10, 10), int)

    labels[1:9, 1:5] = 1

    labels[1:9, 5:9] = 2

    objects.segmented = labels

    module.images[0].wants_objects.value = True

    module.run(workspace)

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalArea_image_objects") == 64

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalIntensity_image_objects") == numpy.sum(pixels[1:9, 1:9])

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_MeanIntensity_image_objects") == numpy.sum(pixels[1:9, 1:9]) / 64.0

    numpy.testing.assert_almost_equal(measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_PercentMaximal_image_objects"), 400. / 64.)

    assert len(measurements.get_object_names()) == 1

    assert measurements.get_object_names()[0] == cellprofiler.measurement.IMAGE

    columns = module.get_measurement_columns(workspace.pipeline)

    features = measurements.get_feature_names(cellprofiler.measurement.IMAGE)

    assert len(columns) == len(features)

    for column in columns:
        assert column[1] in features


def test_01_04_image_and_objects_and_mask(image, measurements, module, objects, workspace):
    '''Test operation on an image masked by objects and a mask'''
    numpy.random.seed(0)

    pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)

    mask = numpy.zeros((10, 10), bool)

    mask[1:9, :9] = True

    image.pixel_data = pixels

    image.mask = mask

    labels = numpy.zeros((10, 10), int)

    labels[1:9, 1:5] = 1

    labels[1:9, 5:] = 2

    objects.segmented = labels

    module.images[0].wants_objects.value = True

    module.run(workspace)

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalArea_image_objects") == 64

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalIntensity_image_objects") == numpy.sum(pixels[1:9, 1:9])

    assert measurements.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_MeanIntensity_image_objects") == numpy.sum(pixels[1:9, 1:9]) / 64.0


def test_02_01_load_v1():
    '''Test loading an measure image intensity module saved in V1'''
    data = ('TUFUTEFCIDUuMCBNQVQtZmlsZSBQbGF0Zm9ybTogbnQsIENyZWF0ZWQg'
            'b246IEZyaSBBcHIgMjQgMTY6MTM6MjQgMjAwOQAAAAAAAAAAAAAAAAAA'
            'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB'
            'SU0OAAAACBIAAAYAAAAIAAAAAgAAAAAAAAAFAAAACAAAAAEAAAABAAAA'
            'AQAAAAgAAABTZXR0aW5ncwUABAAYAAAAAQAAAMAAAABWYXJpYWJsZVZh'
            'bHVlcwAAAAAAAAAAAABWYXJpYWJsZUluZm9UeXBlcwAAAAAAAABNb2R1'
            'bGVOYW1lcwAAAAAAAAAAAAAAAABOdW1iZXJzT2ZWYXJpYWJsZXMAAAAA'
            'AABQaXhlbFNpemUAAAAAAAAAAAAAAAAAAABWYXJpYWJsZVJldmlzaW9u'
            'TnVtYmVycwBNb2R1bGVSZXZpc2lvbk51bWJlcnMAAABNb2R1bGVOb3Rl'
            'cwAAAAAAAAAAAAAAAAAOAAAAkAcAAAYAAAAIAAAAAQAAAAAAAAAFAAAA'
            'CAAAAAIAAAAPAAAAAQAAAAAAAAAOAAAASAAAAAYAAAAIAAAABAAAAAAA'
            'AAAFAAAACAAAAAEAAAARAAAAAQAAAAAAAAAQAAAAEQAAAGluZGl2aWR1'
            'YWwgaW1hZ2VzAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUA'
            'AAAIAAAAAQAAAAMAAAABAAAAAAAAABAAAwBETkEADgAAAEAAAAAGAAAA'
            'CAAAAAQAAAAAAAAABQAAAAgAAAABAAAAEAAAAAEAAAAAAAAAEAAAABAA'
            'AABUZXh0LUV4YWN0IG1hdGNoDgAAADAAAAAGAAAACAAAAAQAAAAAAAAA'
            'BQAAAAgAAAABAAAAAgAAAAEAAAAAAAAAEAACAE5vAAAOAAAAMAAAAAYA'
            'AAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAABAAAAAQAAAAAAAAAQAAEA'
            'MwAAAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAQA'
            'AAABAAAAAAAAABAABABOb25lDgAAAEAAAAAGAAAACAAAAAQAAAAAAAAA'
            'BQAAAAgAAAABAAAACgAAAAEAAAAAAAAAEAAAAAoAAABEbyBub3QgdXNl'
            'AAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAA'
            'AAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABAAAAAAA'
            'AAAFAAAACAAAAAEAAAACAAAAAQAAAAAAAAAQAAIATm8AAA4AAAAwAAAA'
            'BgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkA'
            'AAAAAAAADgAAAEgAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAA'
            'FwAAAAEAAAAAAAAAEAAAABcAAABEZWZhdWx0IEltYWdlIERpcmVjdG9y'
            'eQAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAA'
            'AQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUA'
            'AAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAA'
            'CAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAA'
            'AAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAADAAAA'
            'AQAAAAAAAAAQAAMAWWVzAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUA'
            'AAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAA'
            'CAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAwAAAAEAAAAAAAAAEAADAFll'
            'cwAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAA'
            'AQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUA'
            'AAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAA'
            'CAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAA'
            'AAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAADAAAA'
            'AQAAAAAAAAAQAAMARE5BAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUA'
            'AAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAA'
            'CAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAQAAAAEAAAAAAAAAEAABADEA'
            'AAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAA'
            'AQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUA'
            'AAAIAAAAAQAAAAQAAAABAAAAAAAAABAABABOb25lDgAAADAAAAAGAAAA'
            'CAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAA'
            'AAAOAAAAgAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAABLAAAA'
            'AQAAAAAAAAAQAAAASwAAAF4oP1A8UGxhdGU+LispXyg/UDxXZWxsUm93'
            'PltBLVBdKSg/UDxXZWxsQ29sdW1uPlswLTldezEsMn0pXyg/UDxTaXRl'
            'PlswLTldKQAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgA'
            'AAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAaAAAAAYAAAAIAAAA'
            'BAAAAAAAAAAFAAAACAAAAAEAAAA4AAAAAQAAAAAAAAAQAAAAOAAAACg/'
            'UDxZZWFyPlswLTldezR9KS0oP1A8TW9udGg+WzAtOV17Mn0pLSg/UDxE'
            'YXk+WzAtOV17Mn0pDgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgA'
            'AAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAA6AYAAAYAAAAIAAAA'
            'AQAAAAAAAAAFAAAACAAAAAIAAAAPAAAAAQAAAAAAAAAOAAAAMAAAAAYA'
            'AAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAA'
            'AAAAAA4AAABAAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAoA'
            'AAABAAAAAAAAABAAAAAKAAAAaW1hZ2Vncm91cAAAAAAAAA4AAAAwAAAA'
            'BgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkA'
            'AAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAA'
            'AAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAA'
            'AAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAABAAAAA'
            'BgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAsAAAABAAAAAAAAABAA'
            'AAALAAAAb2JqZWN0Z3JvdXAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAA'
            'AAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAA'
            'AAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAA'
            'CQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAA'
            'AAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAA'
            'AAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAA'
            'AAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAA'
            'CQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAA'
            'AAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAA'
            'AAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAA'
            'AAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAA'
            'CQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAA'
            'AAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAA'
            'AAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAA'
            'AAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAA'
            'CQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAA'
            'AAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAA'
            'AAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAA'
            'AAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAA'
            'CQAAAAAAAAAOAAAAQAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEA'
            'AAAQAAAAAQAAAAAAAAAQAAAAEAAAAGltYWdlZ3JvdXAgaW5kZXAOAAAA'
            'MAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAA'
            'AAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAA'
            'AAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYA'
            'AAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAA'
            'MAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAA'
            'AAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAA'
            'AAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYA'
            'AAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAA'
            'MAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAA'
            'AAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAA'
            'AAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYA'
            'AAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAA'
            'CAEAAAYAAAAIAAAAAQAAAAAAAAAFAAAACAAAAAEAAAACAAAAAQAAAAAA'
            'AAAOAAAAYAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAqAAAA'
            'AQAAAAAAAAAQAAAAKgAAAGNlbGxwcm9maWxlci5tb2R1bGVzLmxvYWRp'
            'bWFnZXMuTG9hZEltYWdlcwAAAAAAAA4AAABwAAAABgAAAAgAAAAEAAAA'
            'AAAAAAUAAAAIAAAAAQAAAEAAAAABAAAAAAAAABAAAABAAAAAY2VsbHBy'
            'b2ZpbGVyLm1vZHVsZXMubWVhc3VyZWltYWdlaW50ZW5zaXR5Lk1lYXN1'
            'cmVJbWFnZUludGVuc2l0eQ4AAAAwAAAABgAAAAgAAAAJAAAAAAAAAAUA'
            'AAAIAAAAAQAAAAIAAAABAAAAAAAAAAIAAgAPAwAADgAAADAAAAAGAAAA'
            'CAAAAAYAAAAAAAAABQAAAAAAAAABAAAAAAAAAAkAAAAIAAAAAAAAAAAA'
            '8D8OAAAAMAAAAAYAAAAIAAAACQAAAAAAAAAFAAAACAAAAAEAAAACAAAA'
            'AQAAAAAAAAACAAIAAgEAAA4AAAAwAAAABgAAAAgAAAALAAAAAAAAAAUA'
            'AAAIAAAAAQAAAAIAAAABAAAAAAAAAAQABAAAAAAADgAAAIgAAAAGAAAA'
            'CAAAAAEAAAAAAAAABQAAAAgAAAABAAAAAgAAAAEAAAAAAAAADgAAACgA'
            'AAAGAAAACAAAAAEAAAAAAAAABQAAAAgAAAAAAAAAAQAAAAEAAAAAAAAA'
            'DgAAACgAAAAGAAAACAAAAAEAAAAAAAAABQAAAAgAAAAAAAAAAQAAAAEA'
            'AAAAAAAA')

    fd = StringIO.StringIO(base64.b64decode(data))

    p = cellprofiler.pipeline.Pipeline()

    def error_handler(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    p.add_listener(error_handler)

    p.load(fd)

    assert len(p.modules()) == 2

    module = p.modules()[1]

    assert len(module.images) == 1

    assert module.images[0].image_name.value == 'DNA'

    assert not module.images[0].wants_objects.value


def test_03_01_get_measurement_columns(module):
    image_names = ['image%d' % i for i in range(3)]

    object_names = ['object%d' % i for i in range(3)]

    first = True

    expected_suffixes = []

    for image_name in image_names:
        if first:
            first = False
        else:
            module.add_image_measurement()

        im = module.images[-1]

        im.image_name.value = image_name

        im.wants_objects.value = False

        expected_suffixes.append(image_name)

        for object_name in object_names:
            module.add_image_measurement()

            im = module.images[-1]

            im.image_name.value = image_name

            im.wants_objects.value = True

            im.object_name.value = object_name

            expected_suffixes.append("%s_%s" % (image_name, object_name))

    columns = module.get_measurement_columns(None)

    assert all([column[0] == cellprofiler.measurement.IMAGE for column in columns])

    for expected_suffix in expected_suffixes:
        for feature, coltype in (
                (cellprofiler.modules.measureimageintensity.F_TOTAL_INTENSITY, cellprofiler.measurement.COLTYPE_FLOAT),
                (cellprofiler.modules.measureimageintensity.F_MEAN_INTENSITY, cellprofiler.measurement.COLTYPE_FLOAT),
                (cellprofiler.modules.measureimageintensity.F_MIN_INTENSITY, cellprofiler.measurement.COLTYPE_FLOAT),
                (cellprofiler.modules.measureimageintensity.F_MAX_INTENSITY, cellprofiler.measurement.COLTYPE_FLOAT),
                (cellprofiler.modules.measureimageintensity.F_TOTAL_AREA, cellprofiler.measurement.COLTYPE_INTEGER),
                (cellprofiler.modules.measureimageintensity.F_PERCENT_MAXIMAL, cellprofiler.measurement.COLTYPE_FLOAT),
                (cellprofiler.modules.measureimageintensity.F_MAD_INTENSITY, cellprofiler.measurement.COLTYPE_FLOAT),
                (cellprofiler.modules.measureimageintensity.F_LOWER_QUARTILE, cellprofiler.measurement.COLTYPE_FLOAT),
                (cellprofiler.modules.measureimageintensity.F_UPPER_QUARTILE, cellprofiler.measurement.COLTYPE_FLOAT)):
            # feature names are now formatting strings
            feature_name = feature % expected_suffix

            assert any([(column[1] == feature_name and column[2] == coltype) for column in columns])

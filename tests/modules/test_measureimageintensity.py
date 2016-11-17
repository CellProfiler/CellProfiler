import StringIO
import base64
import unittest

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.measureimageintensity
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace
import numpy
import pytest
import skimage.morphology

cellprofiler.preferences.set_headless()


@pytest.fixture(scope="function")
def measurements():
    return cellprofiler.measurement.Measurements()


@pytest.fixture(scope="function")
def module():
    return cellprofiler.modules.measureimageintensity.MeasureImageIntensity()


@pytest.fixture(scope="function")
def workspace(measurements, module):
    image_set_list = cellprofiler.image.ImageSetList()

    image_set = image_set_list.get_image_set(0)

    return cellprofiler.workspace.Workspace(
        cellprofiler.pipeline.Pipeline(),
        module,
        image_set,
        cellprofiler.object.ObjectSet(),
        measurements,
        image_set_list
    )


def test_zeros(measurements, module, workspace):
    image = cellprofiler.image.Image(numpy.zeros((10, 10, 10)), dimensions=3)

    workspace.image_set.add("image", image)

    module.images[0].image_name.value = "image"

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


def test_volume(measurements, module, workspace):
    data = skimage.morphology.ball(3)

    image = cellprofiler.image.Image(data, dimensions=3, convert=False)

    workspace.image_set.add("image", image)

    module.images[0].image_name.value = "image"

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


def test_volume_and_mask(measurements, module, workspace):
    mask = skimage.morphology.ball(3, dtype=numpy.bool)

    data = numpy.ones_like(mask, dtype=numpy.uint8)

    image = cellprofiler.image.Image(data, mask=mask, dimensions=3, convert=False)

    workspace.image_set.add("image", image)

    module.images[0].image_name.value = "image"

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


def test_volume_and_objects(measurements, module, workspace):
    object_data = skimage.morphology.ball(3, dtype=numpy.uint8)

    image_data = numpy.ones_like(object_data, dtype=numpy.uint8)

    image = cellprofiler.image.Image(image_data, dimensions=3, convert=False)

    objects = cellprofiler.object.Objects()

    objects.segmented = object_data

    objects.parent_image = image

    workspace.image_set.add("image", image)

    workspace.object_set.add_objects(objects, "objects")

    module.images[0].image_name.value = "image"

    module.images[0].wants_objects.value = True

    module.images[0].object_name.value = "objects"

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

def test_volume_and_objects_and_mask(measurements, module, workspace):
    mask = skimage.morphology.ball(3, dtype=numpy.bool)

    image_data = numpy.ones_like(mask, dtype=numpy.uint8)

    object_data = numpy.ones_like(mask, dtype=numpy.uint8)

    image = cellprofiler.image.Image(image_data, mask=mask, dimensions=3, convert=False)

    objects = cellprofiler.object.Objects()

    objects.segmented = object_data

    objects.parent_image = image

    workspace.image_set.add("image", image)

    workspace.object_set.add_objects(objects, "objects")

    module.images[0].image_name.value = "image"

    module.images[0].wants_objects.value = True

    module.images[0].object_name.value = "objects"

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


class TestMeasureImageIntensity(unittest.TestCase):
    def make_workspace(self, object_dict={}, image_dict={}):
        '''Make a workspace for testing MeasureImageIntensity'''
        module = cellprofiler.modules.measureimageintensity.MeasureImageIntensity()
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
        for key in image_dict.keys():
            image_set.add(key, cellprofiler.image.Image(image_dict[key]))
        for key in object_dict.keys():
            o = cellprofiler.object.Objects()
            o.segmented = object_dict[key]
            object_set.add_objects(o, key)
        return workspace, module

    def test_00_00_zeros(self):
        '''Test operation on a completely-masked image'''
        workspace, module = self.make_workspace({},
                                                {"my_image": numpy.zeros((10, 10))})
        image = workspace.image_set.get_image("my_image")
        image.mask = numpy.zeros((10, 10), bool)
        module.images[0].image_name.value = "my_image"
        module.run(workspace)
        m = workspace.measurements
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalArea_my_image"), 0)

        self.assertEqual(len(m.get_object_names()), 1)
        self.assertEqual(m.get_object_names()[0], cellprofiler.measurement.IMAGE)
        columns = module.get_measurement_columns(workspace.pipeline)
        features = m.get_feature_names(cellprofiler.measurement.IMAGE)
        self.assertEqual(len(columns), len(features))
        for column in columns:
            self.assertTrue(column[1] in features)

    def test_01_01_image(self):
        '''Test operation on a single unmasked image'''
        numpy.random.seed(0)
        pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * .99
        pixels[0:2, 0:2] = 1
        workspace, module = self.make_workspace({},
                                                {"my_image": pixels})
        image = workspace.image_set.get_image("my_image")
        module.images[0].image_name.value = "my_image"
        module.run(workspace)
        m = workspace.measurements
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalArea_my_image"), 100)
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalIntensity_my_image"),
                         numpy.sum(pixels))
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_MeanIntensity_my_image"),
                         numpy.sum(pixels) / 100.0)
        self.assertEqual(m.get_current_image_measurement('Intensity_MinIntensity_my_image'),
                         numpy.min(pixels))
        self.assertEqual(m.get_current_image_measurement('Intensity_MaxIntensity_my_image'),
                         numpy.max(pixels))
        self.assertEqual(m.get_current_image_measurement(
                'Intensity_PercentMaximal_my_image'), 4.0)

    def test_01_02_image_and_mask(self):
        '''Test operation on a masked image'''
        numpy.random.seed(0)
        pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * .99
        pixels[1:3, 1:3] = 1
        mask = numpy.zeros((10, 10), bool)
        mask[1:9, 1:9] = True
        workspace, module = self.make_workspace({},
                                                {"my_image": pixels})
        image = workspace.image_set.get_image("my_image")
        image.mask = mask
        module.images[0].image_name.value = "my_image"
        module.run(workspace)
        m = workspace.measurements
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalArea_my_image"), 64)
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalIntensity_my_image"),
                         numpy.sum(pixels[1:9, 1:9]))
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_MeanIntensity_my_image"),
                         numpy.sum(pixels[1:9, 1:9]) / 64.0)
        self.assertAlmostEqual(m.get_current_measurement(
                cellprofiler.measurement.IMAGE, "Intensity_PercentMaximal_my_image"), 400. / 64.)

    def test_01_03_image_and_objects(self):
        '''Test operation on an image masked by objects'''
        numpy.random.seed(0)
        pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * .99
        pixels[1:3, 1:3] = 1
        objects = numpy.zeros((10, 10), int)
        objects[1:9, 1:5] = 1
        objects[1:9, 5:9] = 2
        workspace, module = self.make_workspace({"my_objects": objects},
                                                {"my_image": pixels})
        module.images[0].image_name.value = "my_image"
        module.images[0].wants_objects.value = True
        module.images[0].object_name.value = "my_objects"
        module.run(workspace)
        m = workspace.measurements
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalArea_my_image_my_objects"), 64)
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalIntensity_my_image_my_objects"),
                         numpy.sum(pixels[1:9, 1:9]))
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_MeanIntensity_my_image_my_objects"),
                         numpy.sum(pixels[1:9, 1:9]) / 64.0)
        self.assertAlmostEqual(m.get_current_measurement(
                cellprofiler.measurement.IMAGE, "Intensity_PercentMaximal_my_image_my_objects"), 400. / 64.)

        self.assertEqual(len(m.get_object_names()), 1)
        self.assertEqual(m.get_object_names()[0], cellprofiler.measurement.IMAGE)
        columns = module.get_measurement_columns(workspace.pipeline)
        features = m.get_feature_names(cellprofiler.measurement.IMAGE)
        self.assertEqual(len(columns), len(features))
        for column in columns:
            self.assertTrue(column[1] in features)

    def test_01_04_image_and_objects_and_mask(self):
        '''Test operation on an image masked by objects and a mask'''
        numpy.random.seed(0)
        pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        objects = numpy.zeros((10, 10), int)
        objects[1:9, 1:5] = 1
        objects[1:9, 5:] = 2
        mask = numpy.zeros((10, 10), bool)
        mask[1:9, :9] = True
        workspace, module = self.make_workspace({"my_objects": objects},
                                                {"my_image": pixels})
        image = workspace.image_set.get_image("my_image")
        image.mask = mask

        module.images[0].image_name.value = "my_image"
        module.images[0].wants_objects.value = True
        module.images[0].object_name.value = "my_objects"
        module.run(workspace)
        m = workspace.measurements
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalArea_my_image_my_objects"), 64)
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_TotalIntensity_my_image_my_objects"),
                         numpy.sum(pixels[1:9, 1:9]))
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "Intensity_MeanIntensity_my_image_my_objects"),
                         numpy.sum(pixels[1:9, 1:9]) / 64.0)

    def test_02_01_load_matlab(self):
        '''Test loading a measure image intensity module saved in Matlab'''
        data = ('TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmV'
                'hdGVkIG9uOiBGcmkgQXByIDI0IDE1OjMxOjE0IDIwMDkgICAgICAgIC'
                'AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI'
                'AABSU0PAAAA0AEAAHic7VXNTsJAEJ6WQlATgiQmHjj06MW/ePGoRhNJ'
                '5CdCSEw8uMBSNyld0m4J+AQ+j0/AY3l0F1tolwotICemmTSzne+b3W+'
                'nuzkAeDsAyPB3lrsKv5b2YiXgIq5jxohlOGnQ4NgbH3NvIpuglombyH'
                'SxA1Pzx0tWlzZG/emnMu24Jq6gXjCZW8XttbDtVLs+0PtcI0Ns1skHh'
                'rD5ac94QBxCLQ/v8cuj07qUSXVz3PXMTAdF0kHoUgiMi/wbmOVrEbrl'
                'A/l5zxt4yE4fhqjN9B5i7XfBc72EJyvxiLhqE+OOSy3wF0vwWgivwf1'
                'trRQHp4RwyiR/G/OV617G1Htfqivie6pblOmu4zXOOuveVH3Y8W2EL+'
                'k+Xv3zfB6X8B1KfCImVocMSMdFpk56yJieSknWqYZ4VajQ9XQ6S6jTn'
                'rQuERs2GjltZOIAT9z5pEJ8KXjhqmxTD7m+dY5C+G9l8T0R1GPdfps0'
                'hWFTtz/PF5c36h6a8eq8BXE/infHH80/hr/3X+6/Vff9iaJOKXAgxDl'
                'fjiQeEZcxclwbT6hKFsOWQ9gI5v+DzJJ1qPwpaKudJ/79nbReWk2O0/'
                'jzWvwqCtwnJNunkwX5vq2a/wMdg65H')
        fd = StringIO.StringIO(base64.b64decode(data))
        p = cellprofiler.pipeline.Pipeline()

        def error_handler(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        p.add_listener(error_handler)
        p.load(fd)
        self.assertEqual(len(p.modules()), 2)
        module = p.modules()[1]
        self.assertEqual(len(module.images), 1)
        self.assertEqual(module.images[0].image_name.value, 'OrigBlue')
        self.assertFalse(module.images[0].wants_objects.value)

    def test_02_01_load_v1(self):
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
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        p.add_listener(error_handler)
        p.load(fd)
        self.assertEqual(len(p.modules()), 2)
        module = p.modules()[1]
        self.assertEqual(len(module.images), 1)
        self.assertEqual(module.images[0].image_name.value, 'DNA')
        self.assertFalse(module.images[0].wants_objects.value)

    def test_03_01_get_measurement_columns(self):
        module = cellprofiler.modules.measureimageintensity.MeasureImageIntensity()
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
        self.assertTrue(all([column[0] == cellprofiler.measurement.IMAGE for column in columns]))
        for expected_suffix in expected_suffixes:
            for feature, coltype in ((cellprofiler.modules.measureimageintensity.F_TOTAL_INTENSITY, cellprofiler.measurement.COLTYPE_FLOAT),
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
                self.assertTrue(any([(column[1] == feature_name and
                                      column[2] == coltype)
                                     for column in columns]))

"""test_measureobjectintensity - test the MeasureObjectIntensity module
"""

import base64
import math
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.modules.injectimage as II
import cellprofiler.modules.measureobjectintensity as MOI
import cellprofiler.pipeline as P
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.image as cpi
import cellprofiler.workspace as cpw
import centrosome.outline as cpmo

#
# This is a pipeline consisting of Matlab modules for LoadImages,
# IdentifyPrimAutomatic and MeasureObjectIntensity
#
pipeline_data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBXZWQgRmViIDExIDE2OjU4OjUyIDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAkgIAAHic7VhPb9MwFHfatPwZmsomJtRTjhzGaAUHjlQqgkqsrdg0iaObuMUoiavEnlo+xaR9EY4cOCDxxYjbpHW8UudvAamWLOs57/3eez87z04OAQCNEwDqwXg/6BWwbLVQ1oTO5QtEKXYnfg3o4Gk4/zPoV9DDcGSjK2gz5INVi+Z77phczqerR+fEYjbqQ0dUDlqfOSPk+YNxZBg+HuIZsi/wVwTiLVL7iK6xj4kb2of48uzKL6GS38OgfztY86BJPFSDfizMc/03YK2vb+CtIeg3wn6JZvT52xk0qeFAan7mOC0FTjWGUwXdfgeUaafH7HTQ7Qx73O61wq4u5cvlPjNthJd85bXPmq/Kb03yy+V26/QV95dknR9K9lzuEsMl1GB+uGGLxEnLw6dgr/+PeVRiOBXQJ/v1KDMPFc6BhMPlAfWZ8c4mI2jvPJ6i1keL4WigHdrtOg55n7RO27E40ubxMqGdXO9bZ612GfmrcB5IOFzuuRS5PqZzIZ5ofK/AeyzhcRm7Fr7GFoO2gR04Wd0CyoivrPq2KY4OoyS4UGAzQxzyvjnLwce2OLLgTTw4901oIwEna73Na6+KP+l7kDSOss+/rHy4L2Ah65CXh8j+pr793i7Wgbx1bVE0Jh5h0+JxojFLXSOjL8ikC0AjqHFomiLPTee6gJcYZ9P3zjrPZVi7jCcpzq552vO95/tv8C2PReWr8vOvj6p6ewTiPHCZMGpjF90puCJuQ/vzuSSfj1nPkQ8EWj3hIpsknycSDpd7FnIpHs+HHnbEO1wSvBMJj8vnCPrMQ4PFdhQuyfJ5X1fwUgmko+N7ub7j0vqrParc+W+kstNDnV/NH83b5tLvd5Bu/Z9t0Y/arvR/A0/mSFM='

IMAGE_NAME = "MyImage"
OBJECT_NAME = "MyObjects"


class TestMeasureObjects(unittest.TestCase):
    def error_callback(self, calller, event):
        if isinstance(event, P.RunExceptionEvent):
            self.fail(event.error.message)

    def test_01_01_load(self):
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        fd = StringIO(base64.b64decode(pipeline_data))
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, MOI.MeasureObjectIntensity))
        self.assertEqual(len(module.objects), 1)
        self.assertEqual(module.objects[0].name.value, 'Nuclei')
        self.assertEqual(module.images[0].name.value, 'DNA')

    def test_01_02_load_v2(self):
        data = ('eJztW0Fv2zYUpl0nWFagS08dhhXgoYckiDXZq9E0KFo78boZqB2jNtoNadYx'
                'Mm1zkEVDotK4Re/9qTvmuONEW7IlTolkS4qdVAIE+VH8+L33+PhIyVS90n5V'
                'OYAlSYb1SjvfJSqGTRWxLtUH+1Bju/BQx4jhDqTaPmybGDboGSw+hoXifqm0'
                'X5BhUZafgsWOTK1+z7pcPANg3bp+Y51Z+9aaLWdcJ5dbmDGi9Yw1kAPf2+UX'
                '1vkG6QSdqvgNUk1szCic8prWpe3RcHqrTjumihto4K5sHQ1zcIp146jrAO3b'
                'TXKO1Rb5iAUTnGqv8RkxCNVsvN2+WDrlpUzgbfXph5e6pY7Q/gFiSr/FrB7w'
                'lnO/bf0481tG8FvOOh+6ynn938Csfs7Hz/dd9TdtmWgdckY6JlIhGaDeVGve'
                'nhzQ3h1Pe3dAtVEZ4/YCcOuCHlxumIqKSTTecgBuU+DlZxufs/wv50hhcMC7'
                'Ig79g/BrAp7Lh1hVDbA4vqJYwwaE81/Gg8+AnyPwFuTdx7KND/L/XQHP5aZO'
                'h6iHmDWIxuVh2vlWaIfLVQo1yqBp2OMoznbmjcc/rFEURz9GjaMgvbMefBY0'
                'aDRckL/9+v+IGSb8VaWnSJ36O+r4C8qDDwQ8l6u4i0yVwRpPgrBKdKwwqo8i'
                '9f+8uIIk/w+3LuCcw8Ft2NcouHKAnmH7bZH5R5bk8bFbsH/EYM8y+8sPl/Pg'
                'ctzmQpL2XTY+k+yfMHoU5GT7ddF8FqZ/QuJKqzh+N4C3X7lc0xjWDMJGrnYW'
                'iY8WVqjWQfroyGQq0S5dPyZtz2EfaRpWC/kE/DJvXpBjzOPzrH/jGI9h4rxB'
                'NZykfeL6tLAg7klIXJi8Ead9Yfw5z7xVDsD5zd/tDxQqKjIMe8RGsbcfwP9E'
                '4Ofyn1svms/4iwj8XNrZfs+lt9bS9flxJd88OZbzT08+FT9vvzf4jRaxao3L'
                'tkPZ+53Ax+Wmbj3eurLUovnuLSa9Pn9dcsZfDGiK89gexX+/B+jxSNCDy9LO'
                '8bt3P51w91RtJ04LXpsalx/56RVnXPk9P72kOu7p1NQ60f0SxH/FPFB0zwNx'
                'rAevYx5P+rl/VeycN48XF7Qvznl4FfJ/kvPwKuf7Zc7ftyGPLzP+k1z3ylJp'
                '6Xom/Z4lznXadeNWZX21av2c9Lpp1cftbctL4nqltCQ9v/www2UEnN//hdcZ'
                '3+M/F3mAD8O345cP6enfWGGzhuLU5zrznMsOSLQOHibY3rL9k+Juxji6LX67'
                'bfamuBSX4lJc1Ly4Cbx5kZ+z+WSybLhJ9n5t/g3iT9dxKS4JXBksN+5T3NeJ'
                'K4M07lJcOu+luBSX4lJciktxNwn3b2aGywg4Lrv3s/D6f7l4/Ob5HVf9TVtW'
                'sKoOdcq/a9SlwfjjO0NSKepMvmaTXlk/a64P2zjPMICnLPCUL+MhHawx0h0N'
                '+eZCk9EBYkSRanYp33JYcUo5bz+Ad0/g3QviNZzN11PO6XZszncewHcg8B1c'
                'xjfAyDB1PHmHTpwd0lJ9Unw0LhY2Totxs+HD7+7/rCU9eHh/7ap4A8AbZ7P4'
                'u3ixCF8ul83eA959d3cDcDngjXuO/wfMF+dbV9R3bFzl+vP6OWMdUf0048lN'
                'dZq0v5r1/wNnl0Vu')
        pipeline = P.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, P.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, MOI.MeasureObjectIntensity))
        self.assertEqual(len(module.images), 2)
        for expected, actual in zip(("DNA", "Actin"), [img.name for img in module.images]):
            self.assertEqual(expected, actual)
        self.assertEqual(len(module.objects), 2)
        for expected, actual in zip(("Cells", "Nuclei"), [obj.name for obj in module.objects]):
            self.assertEqual(expected, actual)

    def test_02_01_supplied_measurements(self):
        """Test the get_category / get_measurements, get_measurement_images functions"""

        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.add_object()
        moi.objects[0].name.value = 'MyObjects1'
        moi.objects[1].name.value = 'MyObjects2'

        self.assertEqual(tuple(sorted(moi.get_categories(None, 'MyObjects1'))),
                         tuple(sorted([MOI.INTENSITY, MOI.C_LOCATION])))
        self.assertEqual(moi.get_categories(None, 'Foo'), [])
        measurements = moi.get_measurements(None, 'MyObjects1', MOI.INTENSITY)
        self.assertEqual(len(measurements), len(MOI.ALL_MEASUREMENTS))
        measurements = moi.get_measurements(None, 'MyObjects1', MOI.C_LOCATION)
        self.assertEqual(len(measurements), len(MOI.ALL_LOCATION_MEASUREMENTS))
        self.assertTrue(all([m in MOI.ALL_LOCATION_MEASUREMENTS for m in measurements]))
        self.assertTrue(moi.get_measurement_images(None, 'MyObjects1',
                                                   MOI.INTENSITY,
                                                   MOI.MAX_INTENSITY),
                        ['MyImage'])

    def test_02_02_get_measurement_columns(self):
        '''test the get_measurement_columns method'''
        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.add_object()
        moi.objects[0].name.value = 'MyObjects1'
        moi.objects[1].name.value = 'MyObjects2'
        columns = moi.get_measurement_columns(None)
        self.assertEqual(len(columns), 2 * len(MOI.ALL_MEASUREMENTS) + 2 * len(MOI.ALL_LOCATION_MEASUREMENTS))
        for column in columns:
            self.assertTrue(column[0] in ('MyObjects1', 'MyObjects2'))
            self.assertEqual(column[2], cpmeas.COLTYPE_FLOAT)
            category = column[1].split('_')[0]
            self.assertTrue(category in (MOI.INTENSITY, MOI.C_LOCATION))
            if category == MOI.INTENSITY:
                self.assertTrue(column[1][column[1].find('_') + 1:] in
                                [m + '_MyImage' for m in MOI.ALL_MEASUREMENTS])
            else:
                self.assertTrue(column[1][column[1].find('_') + 1:] in
                                [m + '_MyImage' for m in MOI.ALL_LOCATION_MEASUREMENTS])

    def features_and_columns_match(self, measurements, module):
        object_names = [x for x in measurements.get_object_names()
                        if x not in (cpmeas.IMAGE, cpmeas.EXPERIMENT)]
        features = [[f for f in measurements.get_feature_names(object_name)
                     if f != 'Exit_Status']
                    for object_name in object_names]
        columns = module.get_measurement_columns(None)
        self.assertEqual(sum([len(f) for f in features]), len(columns))
        for column in columns:
            index = object_names.index(column[0])
            self.assertTrue(column[1] in features[index])
            self.assertTrue(column[2] == cpmeas.COLTYPE_FLOAT)

    def make_workspace(self, labels, pixel_data, mask=None):
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(IMAGE_NAME, cpi.Image(pixel_data, mask))
        object_set = cpo.ObjectSet()
        o = cpo.Objects()
        if labels.shape[1] == 3:
            o.ijv = labels
        else:
            o.segmented = labels
        object_set.add_objects(o, OBJECT_NAME)
        pipeline = P.Pipeline()
        module = MOI.MeasureObjectIntensity()
        module.images[0].name.value = IMAGE_NAME
        module.objects[0].name.value = OBJECT_NAME
        module.module_num = 1
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, cpmeas.Measurements(),
                                  image_set_list)
        return workspace, module

    def test_03_01_00_zero(self):
        """Make sure we can process a blank image"""
        ii = II.InjectImage('MyImage', np.zeros((10, 10)))
        ii.module_num = 1
        io = II.InjectObjects('MyObjects', np.zeros((10, 10), int))
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.objects[0].name.value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        for category, features in ((MOI.INTENSITY, MOI.ALL_MEASUREMENTS),
                                   (MOI.C_LOCATION, MOI.ALL_LOCATION_MEASUREMENTS)):
            for meas_name in features:
                feature_name = "%s_%s_%s" % (category, meas_name, 'MyImage')
                data = m.get_current_measurement('MyObjects', feature_name)
                self.assertEqual(np.product(data.shape), 0, "Got data for feature %s" % feature_name)
            self.features_and_columns_match(m, moi)

    def test_03_01_01_masked(self):
        """Make sure we can process a completely masked image

        Regression test of IMG-971
        """
        ii = II.InjectImage('MyImage', np.zeros((10, 10)),
                            np.zeros((10, 10), bool))
        ii.module_num = 1
        io = II.InjectObjects('MyObjects', np.ones((10, 10), int))
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.objects[0].name.value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        for meas_name in MOI.ALL_MEASUREMENTS:
            feature_name = "%s_%s_%s" % (MOI.INTENSITY, meas_name, 'MyImage')
            data = m.get_current_measurement('MyObjects', feature_name)
            self.assertEqual(np.product(data.shape), 1)
            self.assertTrue(np.all(np.isnan(data) | (data == 0)))
        self.features_and_columns_match(m, moi)

    def test_03_02_00_one(self):
        """Check measurements on a 3x3 square of 1's"""
        img = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
        ii = II.InjectImage('MyImage', img.astype(float))
        ii.module_num = 1
        io = II.InjectObjects('MyObjects', img.astype(int))
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.objects[0].name.value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        for category, meas_name, value in (
                (MOI.INTENSITY, MOI.INTEGRATED_INTENSITY, 9),
                (MOI.INTENSITY, MOI.MEAN_INTENSITY, 1),
                (MOI.INTENSITY, MOI.STD_INTENSITY, 0),
                (MOI.INTENSITY, MOI.MIN_INTENSITY, 1),
                (MOI.INTENSITY, MOI.MAX_INTENSITY, 1),
                (MOI.INTENSITY, MOI.INTEGRATED_INTENSITY_EDGE, 8),
                (MOI.INTENSITY, MOI.MEAN_INTENSITY_EDGE, 1),
                (MOI.INTENSITY, MOI.STD_INTENSITY_EDGE, 0),
                (MOI.INTENSITY, MOI.MIN_INTENSITY_EDGE, 1),
                (MOI.INTENSITY, MOI.MAX_INTENSITY_EDGE, 1),
                (MOI.INTENSITY, MOI.MASS_DISPLACEMENT, 0),
                (MOI.INTENSITY, MOI.LOWER_QUARTILE_INTENSITY, 1),
                (MOI.INTENSITY, MOI.MEDIAN_INTENSITY, 1),
                (MOI.INTENSITY, MOI.UPPER_QUARTILE_INTENSITY, 1),
                (MOI.C_LOCATION, MOI.LOC_CMI_X, 3),
                (MOI.C_LOCATION, MOI.LOC_CMI_Y, 2)):
            feature_name = "%s_%s_%s" % (category, meas_name, 'MyImage')
            data = m.get_current_measurement('MyObjects', feature_name)
            self.assertEqual(np.product(data.shape), 1)
            self.assertEqual(data[0], value, "%s expected %f != actual %f" % (meas_name, value, data[0]))

    def test_03_02_01_one_masked(self):
        """Check measurements on a 3x3 square of 1's"""
        img = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
        ii = II.InjectImage('MyImage', img.astype(float), img > 0)
        ii.module_num = 1
        io = II.InjectObjects('MyObjects', img.astype(int))
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.objects[0].name.value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        for meas_name, value in ((MOI.INTEGRATED_INTENSITY, 9),
                                 (MOI.MEAN_INTENSITY, 1),
                                 (MOI.STD_INTENSITY, 0),
                                 (MOI.MIN_INTENSITY, 1),
                                 (MOI.MAX_INTENSITY, 1),
                                 (MOI.INTEGRATED_INTENSITY_EDGE, 8),
                                 (MOI.MEAN_INTENSITY_EDGE, 1),
                                 (MOI.STD_INTENSITY_EDGE, 0),
                                 (MOI.MIN_INTENSITY_EDGE, 1),
                                 (MOI.MAX_INTENSITY_EDGE, 1),
                                 (MOI.MASS_DISPLACEMENT, 0),
                                 (MOI.LOWER_QUARTILE_INTENSITY, 1),
                                 (MOI.MEDIAN_INTENSITY, 1),
                                 (MOI.MAD_INTENSITY, 0),
                                 (MOI.UPPER_QUARTILE_INTENSITY, 1)):
            feature_name = "%s_%s_%s" % (MOI.INTENSITY, meas_name, 'MyImage')
            data = m.get_current_measurement('MyObjects', feature_name)
            self.assertEqual(np.product(data.shape), 1)
            self.assertEqual(data[0], value, "%s expected %f != actual %f" % (meas_name, value, data[0]))

    def test_03_02_02_intensity_location(self):
        image = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 2, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]]).astype(float) / 2.0
        labels = (image != 0).astype(int)
        workspace, module = self.make_workspace(labels, image)
        self.assertTrue(isinstance(module, MOI.MeasureObjectIntensity))
        module.run(workspace)
        for feature, value in ((MOI.LOC_MAX_X, 5),
                               (MOI.LOC_MAX_Y, 2)):
            feature_name = "%s_%s_%s" % (MOI.C_LOCATION, feature, 'MyImage')
            values = workspace.measurements.get_current_measurement(
                    OBJECT_NAME, feature_name)
            self.assertEqual(len(values), 1)
            self.assertEqual(values[0], value)

    def test_03_03_00_mass_displacement(self):
        """Check the mass displacement of three squares"""

        labels = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 2, 2, 2, 2, 2, 0],
                           [0, 2, 2, 2, 2, 2, 0],
                           [0, 2, 2, 2, 2, 2, 0],
                           [0, 2, 2, 2, 2, 2, 0],
                           [0, 2, 2, 2, 2, 2, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 3, 3, 3, 3, 3, 0],
                           [0, 3, 3, 3, 3, 3, 0],
                           [0, 3, 3, 3, 3, 3, 0],
                           [0, 3, 3, 3, 3, 3, 0],
                           [0, 3, 3, 3, 3, 3, 0],
                           [0, 0, 0, 0, 0, 0, 0]])
        image = np.zeros(labels.shape, dtype=float)
        #
        # image # 1 has a single value in one of the corners
        # whose distance is sqrt(8) from the center
        #
        image[1, 1] = 1
        # image # 2 has a single value on the top edge
        # and should have distance 2
        #
        image[7, 3] = 1
        # image # 3 has a single value on the left edge
        # and should have distance 2
        image[15, 1] = 1
        ii = II.InjectImage('MyImage', image)
        ii.module_num = 1
        io = II.InjectObjects('MyObjects', labels)
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.objects[0].name.value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.MASS_DISPLACEMENT, 'MyImage')
        mass_displacement = m.get_current_measurement('MyObjects', feature_name)
        self.assertEqual(np.product(mass_displacement.shape), 3)
        self.assertAlmostEqual(mass_displacement[0], math.sqrt(8.0))
        self.assertAlmostEqual(mass_displacement[1], 2.0)
        self.assertAlmostEqual(mass_displacement[2], 2.0)

    def test_03_03_01_mass_displacement_masked(self):
        """Regression test IMG-766 - mass displacement of a masked image"""

        labels = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 2, 2, 2, 2, 2, 0],
                           [0, 2, 2, 2, 2, 2, 0],
                           [0, 2, 2, 2, 2, 2, 0],
                           [0, 2, 2, 2, 2, 2, 0],
                           [0, 2, 2, 2, 2, 2, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 3, 3, 3, 3, 3, 0],
                           [0, 3, 3, 3, 3, 3, 0],
                           [0, 3, 3, 3, 3, 3, 0],
                           [0, 3, 3, 3, 3, 3, 0],
                           [0, 3, 3, 3, 3, 3, 0],
                           [0, 0, 0, 0, 0, 0, 0]])
        image = np.zeros(labels.shape, dtype=float)
        #
        # image # 1 has a single value in one of the corners
        # whose distance is sqrt(8) from the center
        #
        image[1, 1] = 1
        # image # 2 has a single value on the top edge
        # and should have distance 2
        #
        image[7, 3] = 1
        # image # 3 has a single value on the left edge
        # and should have distance 2
        image[15, 1] = 1
        mask = np.zeros(image.shape, bool)
        mask[labels > 0] = True
        ii = II.InjectImage('MyImage', image, mask)
        ii.module_num = 1
        io = II.InjectObjects('MyObjects', labels)
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.objects[0].name.value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.MASS_DISPLACEMENT, 'MyImage')
        mass_displacement = m.get_current_measurement('MyObjects', feature_name)
        self.assertEqual(np.product(mass_displacement.shape), 3)
        self.assertAlmostEqual(mass_displacement[0], math.sqrt(8.0))
        self.assertAlmostEqual(mass_displacement[1], 2.0)
        self.assertAlmostEqual(mass_displacement[2], 2.0)

    def test_03_04_quartiles(self):
        """test quartile values on a 250x250 square filled with uniform values"""
        labels = np.ones((250, 250), int)
        np.random.seed(0)
        image = np.random.uniform(size=(250, 250))
        ii = II.InjectImage('MyImage', image)
        ii.module_num = 1
        io = II.InjectObjects('MyObjects', labels)
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.objects[0].name.value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.LOWER_QUARTILE_INTENSITY, 'MyImage')
        data = m.get_current_measurement('MyObjects', feature_name)
        self.assertAlmostEqual(data[0], .25, 2)
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.MEDIAN_INTENSITY, 'MyImage')
        data = m.get_current_measurement('MyObjects', feature_name)
        self.assertAlmostEqual(data[0], .50, 2)
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.MAD_INTENSITY, 'MyImage')
        data = m.get_current_measurement('MyObjects', feature_name)
        self.assertAlmostEqual(data[0], 0.25, 2)
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.UPPER_QUARTILE_INTENSITY, 'MyImage')
        data = m.get_current_measurement('MyObjects', feature_name)
        self.assertAlmostEqual(data[0], .75, 2)

    def test_03_05_quartiles(self):
        """Regression test a bug that occurs in an image with one pixel"""
        labels = np.zeros((10, 20))
        labels[2:7, 3:8] = 1
        labels[5, 15] = 2
        np.random.seed(0)
        image = np.random.uniform(size=(10, 20))
        ii = II.InjectImage('MyImage', image)
        ii.module_num = 1
        io = II.InjectObjects('MyObjects', labels)
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.objects[0].name.value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        # Crashes when pipeline runs in measureobjectintensity.py revision 7146
        m = pipeline.run()

    def test_03_06_quartiles(self):
        """test quartile values on a 250x250 square with 4 objects"""
        labels = np.ones((250, 250), int)
        labels[125:, :] += 1
        labels[:, 125:] += 2
        np.random.seed(0)
        image = np.random.uniform(size=(250, 250))
        #
        # Make the distributions center around .5, .25, 1/6 and .125
        #
        image /= labels.astype(float)
        ii = II.InjectImage('MyImage', image)
        ii.module_num = 1
        io = II.InjectObjects('MyObjects', labels)
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.images[0].name.value = 'MyImage'
        moi.objects[0].name.value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.LOWER_QUARTILE_INTENSITY, 'MyImage')
        data = m.get_current_measurement('MyObjects', feature_name)
        self.assertAlmostEqual(data[0], 1.0 / 4.0, 2)
        self.assertAlmostEqual(data[1], 1.0 / 8.0, 2)
        self.assertAlmostEqual(data[2], 1.0 / 12.0, 2)
        self.assertAlmostEqual(data[3], 1.0 / 16.0, 2)
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.MEDIAN_INTENSITY, 'MyImage')
        data = m.get_current_measurement('MyObjects', feature_name)
        self.assertAlmostEqual(data[0], 1.0 / 2.0, 2)
        self.assertAlmostEqual(data[1], 1.0 / 4.0, 2)
        self.assertAlmostEqual(data[2], 1.0 / 6.0, 2)
        self.assertAlmostEqual(data[3], 1.0 / 8.0, 2)
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.UPPER_QUARTILE_INTENSITY, 'MyImage')
        data = m.get_current_measurement('MyObjects', feature_name)
        self.assertAlmostEqual(data[0], 3.0 / 4.0, 2)
        self.assertAlmostEqual(data[1], 3.0 / 8.0, 2)
        self.assertAlmostEqual(data[2], 3.0 / 12.0, 2)
        self.assertAlmostEqual(data[3], 3.0 / 16.0, 2)
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.MAD_INTENSITY, 'MyImage')
        data = m.get_current_measurement('MyObjects', feature_name)

        self.assertAlmostEqual(data[0], 1.0 / 4.0, 2)
        self.assertAlmostEqual(data[1], 1.0 / 8.0, 2)
        self.assertAlmostEqual(data[2], 1.0 / 12.0, 2)
        self.assertAlmostEqual(data[3], 1.0 / 16.0, 2)

    def test_03_07_median_intensity_masked(self):
        np.random.seed(37)
        labels = np.ones((10, 10), int)
        mask = np.ones((10, 10), bool)
        mask[:, :5] = False
        pixel_data = np.random.uniform(size=(10, 10)).astype(np.float32)
        pixel_data[~mask] = 1
        expected = np.sort(pixel_data[mask])[np.sum(mask) / 2]
        self.assertNotEqual(expected, np.median(pixel_data))
        workspace, module = self.make_workspace(labels, pixel_data, mask)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        values = m.get_current_measurement(
                OBJECT_NAME, '_'.join((MOI.INTENSITY, MOI.MEDIAN_INTENSITY, IMAGE_NAME)))
        self.assertEqual(len(values), 1)
        self.assertEqual(expected, values[0])

    def test_03_08_std_intensity(self):
        np.random.seed(38)
        labels = np.ones((40, 30), int)
        labels[:, 15:] = 3
        labels[20:, :] += 1
        image = np.random.uniform(size=(40, 30)).astype(np.float32)
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        values = m.get_current_measurement(
                OBJECT_NAME, '_'.join((MOI.INTENSITY, MOI.STD_INTENSITY, IMAGE_NAME)))
        self.assertEqual(len(values), 4)
        for i in range(1, 5):
            self.assertAlmostEqual(values[i - 1], np.std(image[labels == i]))

    def test_03_09_std_intensity_edge(self):
        np.random.seed(39)
        labels = np.ones((40, 30), int)
        labels[:, 15:] = 3
        labels[20:, :] += 1
        edge_mask = np.zeros((40, 30), bool)
        i, j = np.mgrid[0:40, 0:30]
        for ii in (0, 19, 20, -1):
            edge_mask[ii, :] = True
        for jj in (0, 14, 15, -1):
            edge_mask[:, jj] = True
        elabels = labels * edge_mask
        image = np.random.uniform(size=(40, 30)).astype(np.float32)
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        values = m.get_current_measurement(
                OBJECT_NAME, '_'.join((MOI.INTENSITY, MOI.STD_INTENSITY_EDGE, IMAGE_NAME)))
        self.assertEqual(len(values), 4)
        for i in range(1, 5):
            self.assertAlmostEqual(values[i - 1], np.std(image[elabels == i]))

    def test_03_10_ijv(self):
        #
        # Test the module on overlapping objects
        #
        np.random.seed(310)
        i, j = np.mgrid[0:30, 0:35]
        o1 = np.argwhere((i - 10) ** 2 + (j - 10) ** 2 < 49)  # circle radius 7 at 10,10
        o2 = np.argwhere((i - 12) ** 2 + (j - 15) ** 2 < 25)  # circle radius 5 at 12,15
        o3 = np.argwhere((i - 15) ** 2 + (j - 25) ** 2 < 49)  # circle radius 7 at 15, 25
        labels = np.vstack([np.column_stack([x, n * np.ones(x.shape[0], int)])
                            for n, x in ((1, o1), (2, o2), (3, o3))])
        image = np.random.uniform(size=i.shape)
        workspace0, module0 = self.make_workspace(labels, image)
        module0.run(workspace0)
        measurements0 = workspace0.measurements
        assert isinstance(measurements0, cpmeas.Measurements)
        for i, (workspace, module) in enumerate((
                self.make_workspace(np.column_stack([o1, np.ones(o1.shape[0], int)]), image),
                self.make_workspace(np.column_stack([o2, np.ones(o2.shape[0], int)]), image),
                self.make_workspace(np.column_stack([o3, np.ones(o3.shape[0], int)]), image))):
            module.run(workspace)
            measurements1 = workspace.measurements
            assert isinstance(measurements1, cpmeas.Measurements)
            for cname, fnames in ((MOI.INTENSITY, MOI.ALL_MEASUREMENTS),
                                  (MOI.C_LOCATION, MOI.ALL_LOCATION_MEASUREMENTS)):
                for fname in fnames:
                    mname = "_".join([cname, fname, IMAGE_NAME])
                    m0 = measurements0.get_measurement(OBJECT_NAME, mname)
                    m1 = measurements1.get_measurement(OBJECT_NAME, mname)
                    self.assertEqual(len(m0), 3)
                    self.assertEqual(len(m1), 1)
                    self.assertEqual(m0[i], m1[0])

    def test_04_01_wrong_image_size(self):
        '''Regression test of IMG-961 - object and image size differ'''
        np.random.seed(41)
        labels = np.ones((20, 50), int)
        image = np.random.uniform(size=(30, 40)).astype(np.float32)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add('MyImage', cpi.Image(image))
        object_set = cpo.ObjectSet()
        o = cpo.Objects()
        o.segmented = labels
        object_set.add_objects(o, "MyObjects")
        pipeline = P.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, P.RunExceptionEvent))

        pipeline.add_listener(callback)
        module = MOI.MeasureObjectIntensity()
        module.module_num = 1
        module.images[0].name.value = "MyImage"
        module.objects[0].name.value = "MyObjects"
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set, object_set,
                                  cpmeas.Measurements(), image_set_list)
        module.run(workspace)
        feature_name = '%s_%s_%s' % (MOI.INTENSITY, MOI.INTEGRATED_INTENSITY, "MyImage")
        m = workspace.measurements.get_current_measurement("MyObjects", feature_name)
        self.assertEqual(len(m), 1)
        self.assertAlmostEqual(m[0], np.sum(image[:20, :40]), 4)

    def test_04_02_masked_edge(self):
        # Regression test of issue #1115
        labels = np.zeros((20, 50), int)
        labels[15:25, 15:25] = 1
        image = np.random.uniform(size=labels.shape).astype(np.float32)
        #
        # Mask the edge of the object
        #
        mask = ~ cpmo.outline(labels).astype(bool)
        m = cpmeas.Measurements()
        m.add(IMAGE_NAME, cpi.Image(image, mask=mask))
        object_set = cpo.ObjectSet()
        o = cpo.Objects()
        o.segmented = labels
        object_set.add_objects(o, OBJECT_NAME)
        pipeline = P.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, P.RunExceptionEvent))

        pipeline.add_listener(callback)
        module = MOI.MeasureObjectIntensity()
        module.module_num = 1
        module.images[0].name.value = IMAGE_NAME
        module.objects[0].name.value = OBJECT_NAME
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, m, object_set,
                                  m, None)
        module.run(workspace)

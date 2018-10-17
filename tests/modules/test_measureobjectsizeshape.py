import StringIO
import os
import unittest

import numpy
import skimage.io

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.injectimage
import cellprofiler.modules.measureobjectsizeshape
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()


OBJECTS_NAME = "myobjects"


class TestMeasureObjectSizeShape(unittest.TestCase):
    def make_workspace(self, labels):
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler.object.ObjectSet()
        objects = cellprofiler.object.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
        module.module_num = 1
        module.object_groups[0].name.value = OBJECTS_NAME
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, object_set,
                                                     m, image_set_list)
        return workspace, module

    def test_01_01_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8957

MeasureObjectSizeShape:[module_num:1|svn_version:\'1\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select objects to measure:Nuclei
    Select objects to measure:Cells
    Calculate the Zernike features?:Yes
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureobjectsizeshape.MeasureObjectSizeShape))
        self.assertEqual(len(module.object_groups), 2)
        for og, expected in zip(module.object_groups, ("Nuclei", "Cells")):
            self.assertEqual(og.name, expected)
        self.assertTrue(module.calculate_zernikes)

    def test_01_00_zeros(self):
        """Run on an empty labels matrix"""
        object_set = cellprofiler.object.ObjectSet()
        labels = numpy.zeros((10, 20), int)
        objects = cellprofiler.object.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, "SomeObjects")
        module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
        settings = ["SomeObjects", "Yes"]
        module.set_settings_from_values(settings, 1, module.module_class())
        module.module_num = 1
        image_set_list = cellprofiler.image.ImageSetList()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set_list.get_image_set(0),
                                                     object_set, measurements, image_set_list)
        module.run(workspace)

        for f in (cellprofiler.modules.measureobjectsizeshape.F_AREA, cellprofiler.modules.measureobjectsizeshape.F_CENTER_X, cellprofiler.modules.measureobjectsizeshape.F_CENTER_Y,
                  cellprofiler.modules.measureobjectsizeshape.F_ECCENTRICITY, cellprofiler.modules.measureobjectsizeshape.F_EULER_NUMBER,
                  cellprofiler.modules.measureobjectsizeshape.F_EXTENT, cellprofiler.modules.measureobjectsizeshape.F_FORM_FACTOR,
                  cellprofiler.modules.measureobjectsizeshape.F_MAJOR_AXIS_LENGTH, cellprofiler.modules.measureobjectsizeshape.F_MINOR_AXIS_LENGTH,
                  cellprofiler.modules.measureobjectsizeshape.F_ORIENTATION, cellprofiler.modules.measureobjectsizeshape.F_PERIMETER,
                  cellprofiler.modules.measureobjectsizeshape.F_SOLIDITY, cellprofiler.modules.measureobjectsizeshape.F_COMPACTNESS,
                  cellprofiler.modules.measureobjectsizeshape.F_MAXIMUM_RADIUS, cellprofiler.modules.measureobjectsizeshape.F_MEAN_RADIUS,
                  cellprofiler.modules.measureobjectsizeshape.F_MEDIAN_RADIUS,
                  cellprofiler.modules.measureobjectsizeshape.F_MIN_FERET_DIAMETER, cellprofiler.modules.measureobjectsizeshape.F_MAX_FERET_DIAMETER):
            m = cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE + "_" + f
            a = measurements.get_current_measurement('SomeObjects', m)
            self.assertEqual(len(a), 0)

    def test_01_02_run(self):
        """Run with a rectangle, cross and circle"""
        object_set = cellprofiler.object.ObjectSet()
        labels = numpy.zeros((10, 20), int)
        labels[1:9, 1:5] = 1
        labels[1:9, 11] = 2
        labels[4, 6:19] = 2
        objects = cellprofiler.object.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, "SomeObjects")
        labels = numpy.zeros((115, 115), int)
        x, y = numpy.mgrid[-50:51, -50:51]
        labels[:101, :101][x ** 2 + y ** 2 <= 2500] = 1
        objects = cellprofiler.object.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, "OtherObjects")
        module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
        settings = ["SomeObjects", "OtherObjects", "Yes"]
        module.set_settings_from_values(settings, 1, module.module_class())
        module.module_num = 1
        image_set_list = cellprofiler.image.ImageSetList()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set_list.get_image_set(0),
                                                     object_set, measurements, image_set_list)
        module.run(workspace)
        self.features_and_columns_match(measurements, module, pipeline)

        a = measurements.get_current_measurement('SomeObjects', 'AreaShape_Area')
        self.assertEqual(len(a), 2)
        self.assertEqual(a[0], 32)
        self.assertEqual(a[1], 20)
        #
        # Mini-test of the form factor of a circle
        #
        ff = measurements.get_current_measurement('OtherObjects',
                                                  'AreaShape_FormFactor')
        self.assertEqual(len(ff), 1)
        perim = measurements.get_current_measurement('OtherObjects',
                                                     'AreaShape_Perimeter')
        area = measurements.get_current_measurement('OtherObjects',
                                                    'AreaShape_Area')
        # The perimeter is obtained geometrically and is overestimated.
        expected = 100 * numpy.pi
        diff = abs((perim[0] - expected) / (perim[0] + expected))
        self.assertTrue(diff < .05, "perimeter off by %f" % diff)
        wrongness = (perim[0] / expected) ** 2

        # It's an approximate circle...
        expected = numpy.pi * 50.0 ** 2
        diff = abs((area[0] - expected) / (area[0] + expected))
        self.assertTrue(diff < .05, "area off by %f" % diff)
        wrongness *= expected / area[0]

        self.assertAlmostEqual(ff[0] * wrongness, 1.0)
        for object_name, object_count in (('SomeObjects', 2),
                                          ('OtherObjects', 1)):
            for measurement in module.get_measurements(pipeline, object_name,
                                                       'AreaShape'):
                feature_name = 'AreaShape_%s' % measurement
                m = measurements.get_current_measurement(object_name,
                                                         feature_name)
                self.assertEqual(len(m), object_count)

    def test_02_01_categories(self):
        module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
        settings = ["SomeObjects", "OtherObjects", "Yes"]
        module.set_settings_from_values(settings, 1, module.module_class())
        for object_name in settings[:-1]:
            categories = module.get_categories(None, object_name)
            self.assertEqual(len(categories), 1)
            self.assertEqual(categories[0], "AreaShape")
        self.assertEqual(len(module.get_categories(None, "Bogus")), 0)

    def test_02_02_measurements_zernike(self):
        module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
        settings = ["SomeObjects", "OtherObjects", "Yes"]
        module.set_settings_from_values(settings, 1, module.module_class())
        pipeline = cellprofiler.pipeline.Pipeline()
        for object_name in settings[:-1]:
            measurements = module.get_measurements(pipeline, object_name, 'AreaShape')
            for measurement in cellprofiler.modules.measureobjectsizeshape.F_STANDARD + \
                    cellprofiler.modules.measureobjectsizeshape.F_STD_2D:
                self.assertTrue(measurement in measurements)
            self.assertTrue('Zernike_3_1' in measurements)

    def test_02_03_measurements_no_zernike(self):
        module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
        settings = ["SomeObjects", "OtherObjects", "No"]
        module.set_settings_from_values(settings, 1, module.module_class())
        pipeline = cellprofiler.pipeline.Pipeline()
        for object_name in settings[:-1]:
            measurements = module.get_measurements(pipeline, object_name, 'AreaShape')
            for measurement in cellprofiler.modules.measureobjectsizeshape.F_STANDARD + \
                    cellprofiler.modules.measureobjectsizeshape.F_STD_2D:
                self.assertTrue(measurement in measurements)
            self.assertFalse('Zernike_3_1' in measurements)

    def test_03_01_non_contiguous(self):
        '''make sure MeasureObjectAreaShape doesn't crash if fed non-contiguous objects'''
        module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
        module.object_groups[0].name.value = "SomeObjects"
        module.calculate_zernikes.value = True
        object_set = cellprofiler.object.ObjectSet()
        labels = numpy.zeros((10, 20), int)
        labels[1:9, 1:5] = 1
        labels[4:6, 6:19] = 1
        objects = cellprofiler.object.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, "SomeObjects")
        module.module_num = 1
        image_set_list = cellprofiler.image.ImageSetList()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set_list.get_image_set(0),
                                                     object_set, measurements, image_set_list)
        module.run(workspace)
        values = measurements.get_current_measurement("SomeObjects",
                                                      "AreaShape_Perimeter")
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], 54)

    def test_03_02_zernikes_are_different(self):
        '''Regression test of IMG-773'''

        numpy.random.seed(32)
        labels = numpy.zeros((40, 20), int)
        #
        # Make two "objects" composed of random foreground/background
        #
        labels[1:19, 1:19] = (numpy.random.uniform(size=(18, 18)) > .5).astype(int)
        labels[21:39, 1:19] = (numpy.random.uniform(size=(18, 18)) > .5).astype(int) * 2
        objects = cellprofiler.object.Objects()
        objects.segmented = labels
        object_set = cellprofiler.object.ObjectSet()
        object_set.add_objects(objects, "SomeObjects")
        module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
        module.object_groups[0].name.value = "SomeObjects"
        module.calculate_zernikes.value = True
        module.module_num = 1
        image_set_list = cellprofiler.image.ImageSetList()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set_list.get_image_set(0),
                                                     object_set, measurements, image_set_list)
        module.run(workspace)
        features = [x[1] for x in module.get_measurement_columns(pipeline)
                    if x[0] == "SomeObjects" and
                    x[1].startswith("AreaShape_Zernike")]
        for feature in features:
            values = measurements.get_current_measurement(
                    "SomeObjects", feature)
            self.assertEqual(len(values), 2)
            self.assertNotEqual(values[0], values[1])

    def test_04_01_extent(self):
        module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
        module.object_groups[0].name.value = "SomeObjects"
        module.calculate_zernikes.value = True
        object_set = cellprofiler.object.ObjectSet()
        labels = numpy.zeros((10, 20), int)
        # 3/4 of a square is covered
        labels[5:7, 5:10] = 1
        labels[7:9, 5:15] = 1
        objects = cellprofiler.object.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, "SomeObjects")
        module.module_num = 1
        image_set_list = cellprofiler.image.ImageSetList()
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set_list.get_image_set(0),
                                                     object_set, measurements, image_set_list)
        module.run(workspace)
        values = measurements.get_current_measurement(
                "SomeObjects", "_".join((cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE, cellprofiler.modules.measureobjectsizeshape.F_EXTENT)))
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(values[0], .75)

    def test_05_01_overlapping(self):
        '''Test object measurement with two overlapping objects in ijv format'''

        i, j = numpy.mgrid[0:10, 0:20]
        m = (i > 1) & (i < 9) & (j > 1) & (j < 19)
        m1 = m & (i < j)
        m2 = m & (i < 9 - j)
        mlist = []
        olist = []
        for m in (m1, m2):
            objects = cellprofiler.object.Objects()
            objects.segmented = m.astype(int)
            olist.append(objects)
        ijv = numpy.column_stack((
            numpy.hstack([numpy.argwhere(m)[:, 0] for m in (m1, m2)]),
            numpy.hstack([numpy.argwhere(m)[:, 1] for m in (m1, m2)]),
            numpy.array([1] * numpy.sum(m1) + [2] * numpy.sum(m2))))
        objects = cellprofiler.object.Objects()
        objects.ijv = ijv
        olist.append(objects)
        for objects in olist:
            module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
            module.object_groups[0].name.value = "SomeObjects"
            module.calculate_zernikes.value = True
            object_set = cellprofiler.object.ObjectSet()
            object_set.add_objects(objects, "SomeObjects")
            module.module_num = 1
            image_set_list = cellprofiler.image.ImageSetList()
            measurements = cellprofiler.measurement.Measurements()
            mlist.append(measurements)
            pipeline = cellprofiler.pipeline.Pipeline()
            pipeline.add_module(module)

            def callback(caller, event):
                self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))
                pipeline.add_listener(callback)

            workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                         image_set_list.get_image_set(0),
                                                         object_set, measurements, image_set_list)
            module.run(workspace)

        pipeline = cellprofiler.pipeline.Pipeline()
        for c in module.get_measurement_columns(pipeline):
            oname, feature = c[:2]
            if oname != "SomeObjects":
                continue
            measurements = mlist[0]
            self.assertTrue(isinstance(measurements, cellprofiler.measurement.Measurements))
            v1 = measurements.get_current_measurement(oname, feature)
            self.assertEqual(len(v1), 1)
            v1 = v1[0]
            measurements = mlist[1]
            v2 = measurements.get_current_measurement(oname, feature)
            self.assertEqual(len(v2), 1)
            v2 = v2[0]
            expected = (v1, v2)
            v = mlist[2].get_current_measurement(oname, feature)
            self.assertEqual(tuple(v), expected)

    def test_06_01_max_radius(self):
        labels = numpy.zeros((20, 10), int)
        labels[3:8, 3:6] = 1
        labels[11:19, 2:7] = 2
        workspace, module = self.make_workspace(labels)
        module.run(workspace)
        m = workspace.measurements
        max_radius = m.get_current_measurement(
                OBJECTS_NAME, cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE + "_" + cellprofiler.modules.measureobjectsizeshape.F_MAXIMUM_RADIUS)
        self.assertEqual(len(max_radius), 2)
        self.assertEqual(max_radius[0], 2)
        self.assertEqual(max_radius[1], 3)

    def features_and_columns_match(self, measurements, module, pipeline):
        self.assertEqual(len(measurements.get_object_names()), 3)
        self.assertTrue('SomeObjects' in measurements.get_object_names())
        self.assertTrue('OtherObjects' in measurements.get_object_names())
        features = measurements.get_feature_names('SomeObjects')
        features += measurements.get_feature_names('OtherObjects')
        columns = module.get_measurement_columns(pipeline)
        self.assertEqual(len(features), len(columns))
        for column in columns:
            self.assertTrue(column[0] in ['SomeObjects', 'OtherObjects'])
            self.assertTrue(column[1] in features)
            self.assertTrue(column[2] == cellprofiler.measurement.COLTYPE_FLOAT)

    def test_run_volume(self):
        labels = numpy.zeros((10, 20, 40), dtype=numpy.uint8)
        labels[:, 5:15, 25:35] = 1

        workspace, module = self.make_workspace(labels)
        workspace.pipeline.set_volumetric(True)
        module.run(workspace)

        for feature in [cellprofiler.modules.measureobjectsizeshape.F_VOLUME,
                        cellprofiler.modules.measureobjectsizeshape.F_EXTENT,
                        cellprofiler.modules.measureobjectsizeshape.F_CENTER_X,
                        cellprofiler.modules.measureobjectsizeshape.F_CENTER_Y,
                        cellprofiler.modules.measureobjectsizeshape.F_CENTER_Z,
                        cellprofiler.modules.measureobjectsizeshape.F_SURFACE_AREA]:
            assert workspace.measurements.has_current_measurements(
                OBJECTS_NAME,
                cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE + "_" + feature
            )

            assert len(workspace.measurements.get_current_measurement(
                OBJECTS_NAME,
                cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE + "_" + feature
            )) == 1

        # Assert AreaShape_Center_X and AreaShape_Center_Y aren't flipped. See:
        # https://github.com/CellProfiler/CellProfiler/issues/3352
        center_x = workspace.measurements.get_current_measurement(
            OBJECTS_NAME,
            "_".join([cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE,
                      cellprofiler.modules.measureobjectsizeshape.F_CENTER_X])
        )[0]

        assert center_x == 29.5

        center_y = workspace.measurements.get_current_measurement(
            OBJECTS_NAME,
            "_".join([cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE,
                      cellprofiler.modules.measureobjectsizeshape.F_CENTER_Y])
        )[0]

        assert center_y == 9.5

    # https://github.com/CellProfiler/CellProfiler/issues/2813
    def test_run_without_zernikes(self):
        cells_resource = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "resources",
                "cells.tiff"
            )
        )

        workspace, module = self.make_workspace(skimage.io.imread(cells_resource))

        module.calculate_zernikes.value = False

        module.run(workspace)

        measurements = workspace.measurements

        for feature in measurements.get_feature_names(OBJECTS_NAME):
            assert "Zernike_" not in feature

    def test_run_with_zernikes(self):
        cells_resource = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "resources",
                "cells.tiff"
            )
        )

        workspace, module = self.make_workspace(skimage.io.imread(cells_resource))

        module.calculate_zernikes.value = True

        module.run(workspace)

        measurements = workspace.measurements

        zernikes = [feature for feature in measurements.get_feature_names(OBJECTS_NAME) if "Zernike_" in feature]

        assert len(zernikes) > 0

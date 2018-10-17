'''test_measurecorrelation - test the MeasureCorrelation module'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.measurecolocalization as M

IMAGE1_NAME = 'image1'
IMAGE2_NAME = 'image2'
OBJECTS_NAME = 'objects'


class TestMeasureCorrelation(unittest.TestCase):
    def make_workspace(self, image1, image2, objects=None):
        '''Make a workspace for testing Threshold'''
        module = M.MeasureColocalization()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        for image_group, name, image in zip(module.image_groups,
                                            (IMAGE1_NAME, IMAGE2_NAME),
                                            (image1, image2)):
            image_group.image_name.value = name
            image_set.add(name, image)
        object_set = cpo.ObjectSet()
        if objects is None:
            module.images_or_objects.value = M.M_IMAGES
        else:
            module.images_or_objects.value = M.M_IMAGES_AND_OBJECTS
            module.object_groups[0].object_name.value = OBJECTS_NAME
            object_set.add_objects(objects, OBJECTS_NAME)
        pipeline = cpp.Pipeline()
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  cpmeas.Measurements(),
                                  image_set_list)
        return workspace, module

    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8905

MeasureColocalization:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Hidden:2
    Hidden:2
    Select an image to measure:DNA
    Select an image to measure:Cytoplasm
    Select where to measure correlation:Both
    Select an object to measure:Nuclei
    Select an object to measure:Cells
"""
        fd = StringIO(data)
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertEqual(module.images_or_objects.value, M.M_IMAGES_AND_OBJECTS)
        self.assertEqual(module.image_count.value, 2)
        self.assertEqual(module.thr, 15.0)
        for name in [x.image_name.value for x in module.image_groups]:
            self.assertTrue(name in ["DNA", "Cytoplasm"])

        self.assertEqual(module.object_count.value, 2)
        for name in [x.object_name.value for x in module.object_groups]:
            self.assertTrue(name in ["Nuclei", "Cells"])

    def test_01_04_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20160216135025
GitHash:e55aeba
ModuleCount:1
HasImagePlaneDetails:False

MeasureColocalization:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Hidden:2
    Hidden:2
    Select an image to measure:DNA
    Select an image to measure:Cytoplasm
    Set threshold as percentage of maximum intensity for the images:25.0
    Select where to measure correlation:Both
    Select an object to measure:Nuclei
    Select an object to measure:Cells
"""
        fd = StringIO(data)
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertEqual(module.images_or_objects.value, M.M_IMAGES_AND_OBJECTS)
        self.assertEqual(module.image_count.value, 2)
        self.assertEqual(module.thr, 25.0)
        for name in [x.image_name.value for x in module.image_groups]:
            self.assertTrue(name in ["DNA", "Cytoplasm"])

        self.assertEqual(module.object_count.value, 2)
        for name in [x.object_name.value for x in module.object_groups]:
            self.assertTrue(name in ["Nuclei", "Cells"])

    all_object_measurement_formats = [
        M.F_CORRELATION_FORMAT, M.F_COSTES_FORMAT, M.F_K_FORMAT,
        M.F_MANDERS_FORMAT, M.F_OVERLAP_FORMAT, M.F_RWC_FORMAT]
    all_image_measurement_formats = all_object_measurement_formats + [
        M.F_SLOPE_FORMAT]
    asymmetrical_measurement_formats = [
        M.F_COSTES_FORMAT, M.F_K_FORMAT, M.F_MANDERS_FORMAT, M.F_RWC_FORMAT]

    def test_02_01_get_categories(self):
        '''Test the get_categories function for some different cases'''
        module = M.MeasureColocalization()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = M.M_IMAGES

        def cat(name):
            return module.get_categories(None, name) == ["Correlation"]

        self.assertTrue(cat("Image"))
        self.assertFalse(cat(OBJECTS_NAME))
        module.images_or_objects.value = M.M_OBJECTS
        self.assertFalse(cat("Image"))
        self.assertTrue(cat(OBJECTS_NAME))
        module.images_or_objects.value = M.M_IMAGES_AND_OBJECTS
        self.assertTrue(cat("Image"))
        self.assertTrue(cat(OBJECTS_NAME))

    def test_02_02_get_measurements(self):
        '''Test the get_measurements function for some different cases'''
        module = M.MeasureColocalization()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = M.M_IMAGES

        def meas(name):
            ans = list(module.get_measurements(None, name, "Correlation"))
            ans.sort()
            if name == "Image":
                mf = self.all_image_measurement_formats
            else:
                mf = self.all_object_measurement_formats
            expected = sorted([_.split("_")[1] for _ in mf])
            return ans == expected

        self.assertTrue(meas("Image"))
        self.assertFalse(meas(OBJECTS_NAME))
        module.images_or_objects.value = M.M_OBJECTS
        self.assertFalse(meas("Image"))
        self.assertTrue(meas(OBJECTS_NAME))
        module.images_or_objects.value = M.M_IMAGES_AND_OBJECTS
        self.assertTrue(meas("Image"))
        self.assertTrue(meas(OBJECTS_NAME))

    def test_02_03_get_measurement_images(self):
        '''Test the get_measurment_images function for some different cases'''
        for iocase, names in (
                (M.M_IMAGES, [cpmeas.IMAGE]),
                (M.M_OBJECTS, [OBJECTS_NAME]),
                (M.M_IMAGES_AND_OBJECTS, [cpmeas.IMAGE, OBJECTS_NAME])):
            module = M.MeasureColocalization()
            module.image_groups[0].image_name.value = IMAGE1_NAME
            module.image_groups[1].image_name.value = IMAGE2_NAME
            module.object_groups[0].object_name.value = OBJECTS_NAME
            module.images_or_objects.value = iocase
            for name, mfs in ((cpmeas.IMAGE, self.all_image_measurement_formats),
                              (OBJECTS_NAME, self.all_object_measurement_formats)):
                if name not in names:
                    continue
                for mf in mfs:
                    ftr = mf.split("_")[1]
                    ans = module.get_measurement_images(
                            None, name, "Correlation", ftr)
                    expected = ["%s_%s" % (i1, i2) for i1, i2 in
                                ((IMAGE1_NAME, IMAGE2_NAME),
                                 (IMAGE2_NAME, IMAGE1_NAME))]
                    if mf in self.asymmetrical_measurement_formats:
                        self.assertTrue(all([e in ans for e in expected]))
                    else:
                        self.assertTrue(any([e in ans for e in expected]))

    def test_02_04_01_get_measurement_columns_images(self):
        module = M.MeasureColocalization()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = M.M_IMAGES
        columns = module.get_measurement_columns(None)
        expected = [
                       (cpmeas.IMAGE,
                        ftr % (IMAGE1_NAME, IMAGE2_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.all_image_measurement_formats] + [
                       (cpmeas.IMAGE,
                        ftr % (IMAGE2_NAME, IMAGE1_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.asymmetrical_measurement_formats]
        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cf == ef for cf, ef in zip(column, ex)])
                                 for ex in expected]))

    def test_02_04_02_get_measurement_columns_objects(self):
        module = M.MeasureColocalization()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = M.M_OBJECTS
        columns = module.get_measurement_columns(None)
        expected = [
                       (OBJECTS_NAME,
                        ftr % (IMAGE1_NAME, IMAGE2_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.all_object_measurement_formats] + [
                       (OBJECTS_NAME,
                        ftr % (IMAGE2_NAME, IMAGE1_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.asymmetrical_measurement_formats]
        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cf == ef for cf, ef in zip(column, ex)])
                                 for ex in expected]))

    def test_02_04_03_get_measurement_columns_both(self):
        module = M.MeasureColocalization()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = M.M_IMAGES_AND_OBJECTS
        columns = module.get_measurement_columns(None)
        expected = [
                       (cpmeas.IMAGE,
                        ftr % (IMAGE1_NAME, IMAGE2_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.all_image_measurement_formats] + [
                       (cpmeas.IMAGE,
                        ftr % (IMAGE2_NAME, IMAGE1_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.asymmetrical_measurement_formats] + [
                       (OBJECTS_NAME,
                        ftr % (IMAGE1_NAME, IMAGE2_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.all_object_measurement_formats] + [
                       (OBJECTS_NAME,
                        ftr % (IMAGE2_NAME, IMAGE1_NAME),
                        cpmeas.COLTYPE_FLOAT)
                       for ftr in self.asymmetrical_measurement_formats]

        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cf == ef for cf, ef in zip(column, ex)])
                                 for ex in expected]))

    def test_03_01_correlated(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10))
        i1 = cpi.Image(image)
        i2 = cpi.Image(image)
        workspace, module = self.make_workspace(i1, i2)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, cpmeas.IMAGE, "Correlation", "Correlation")
        corr = m.get_current_measurement(cpmeas.IMAGE, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr, 1)

        self.assertEqual(len(m.get_object_names()), 1)
        self.assertEqual(m.get_object_names()[0], cpmeas.IMAGE)
        columns = module.get_measurement_columns(None)
        features = m.get_feature_names(cpmeas.IMAGE)
        self.assertEqual(len(columns), len(features))
        for column in columns:
            self.assertTrue(column[1] in features)

    def test_03_02_anticorrelated(self):
        '''Test two anticorrelated images'''
        #
        # Make a checkerboard pattern and reverse it for one image
        #
        i, j = np.mgrid[0:10, 0:10]
        image1 = ((i + j) % 2).astype(float)
        image2 = 1 - image1
        i1 = cpi.Image(image1)
        i2 = cpi.Image(image2)
        workspace, module = self.make_workspace(i1, i2)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, cpmeas.IMAGE, "Correlation", "Correlation")
        corr = m.get_current_measurement(cpmeas.IMAGE, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr, -1)

    def test_04_01_slope(self):
        '''Test the slope measurement'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(10, 10)).astype(np.float32)
        image2 = image1 * .5
        i1 = cpi.Image(image1)
        i2 = cpi.Image(image2)
        workspace, module = self.make_workspace(i1, i2)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, cpmeas.IMAGE, "Correlation", "Slope")
        slope = m.get_current_measurement(cpmeas.IMAGE, "Correlation_Slope_%s" % mi[0])
        if mi[0] == "%s_%s" % (IMAGE1_NAME, IMAGE2_NAME):
            self.assertAlmostEqual(slope, .5, 5)
        else:
            self.assertAlmostEqual(slope, 2)

    def test_05_01_crop(self):
        '''Test similarly cropping one image to another'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(20, 20))
        i1 = cpi.Image(image1)
        crop_mask = np.zeros((20, 20), bool)
        crop_mask[5:16, 5:16] = True
        i2 = cpi.Image(image1[5:16, 5:16], crop_mask=crop_mask)
        workspace, module = self.make_workspace(i1, i2)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, cpmeas.IMAGE, "Correlation", "Correlation")
        corr = m.get_current_measurement(cpmeas.IMAGE, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr, 1)

    def test_05_02_mask(self):
        '''Test images with two different masks'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(20, 20))
        mask1 = np.ones((20, 20), bool)
        mask1[5:8, 8:12] = False
        mask2 = np.ones((20, 20), bool)
        mask2[14:18, 2:5] = False
        mask = mask1 & mask2
        image2 = image1.copy()
        #
        # Try to confound the module by making masked points anti-correlated
        #
        image2[~mask] = 1 - image1[~mask]
        i1 = cpi.Image(image1, mask=mask1)
        i2 = cpi.Image(image2, mask=mask2)
        workspace, module = self.make_workspace(i1, i2)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, cpmeas.IMAGE, "Correlation", "Correlation")
        corr = m.get_current_measurement(cpmeas.IMAGE, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr, 1)

    def test_06_01_objects(self):
        '''Test images with two objects'''
        labels = np.zeros((10, 10), int)
        labels[:4, :4] = 1
        labels[6:, 6:] = 2
        i, j = np.mgrid[0:10, 0:10]
        image1 = ((i + j) % 2).astype(float)
        image2 = image1.copy()
        #
        # Anti-correlate the second object
        #
        image2[labels == 2] = 1 - image1[labels == 2]
        i1 = cpi.Image(image1)
        i2 = cpi.Image(image2)
        o = cpo.Objects()
        o.segmented = labels
        workspace, module = self.make_workspace(i1, i2, o)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, OBJECTS_NAME,
                                           "Correlation", "Correlation")
        corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
        self.assertEqual(len(corr), 2)
        self.assertAlmostEqual(corr[0], 1)
        self.assertAlmostEqual(corr[1], -1)

        self.assertEqual(len(m.get_object_names()), 2)
        self.assertTrue(OBJECTS_NAME in m.get_object_names())
        columns = module.get_measurement_columns(None)
        image_features = m.get_feature_names(cpmeas.IMAGE)
        object_features = m.get_feature_names(OBJECTS_NAME)
        self.assertEqual(len(columns), len(image_features) + len(object_features))
        for column in columns:
            if column[0] == cpmeas.IMAGE:
                self.assertTrue(column[1] in image_features)
            else:
                self.assertEqual(column[0], OBJECTS_NAME)
                self.assertTrue(column[1] in object_features)

    def test_06_02_cropped_objects(self):
        '''Test images and objects with a cropping mask'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(20, 20))
        i1 = cpi.Image(image1)
        crop_mask = np.zeros((20, 20), bool)
        crop_mask[5:15, 5:15] = True
        i2 = cpi.Image(image1[5:15, 5:15], crop_mask=crop_mask)
        labels = np.zeros((10, 10), int)
        labels[:4, :4] = 1
        labels[6:, 6:] = 2
        o = cpo.Objects()
        o.segmented = labels
        #
        # Make the objects have the cropped image as a parent
        #
        o.parent_image = i2
        workspace, module = self.make_workspace(i1, i2, o)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, OBJECTS_NAME, "Correlation", "Correlation")
        corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr[0], 1)
        self.assertAlmostEqual(corr[1], 1)

    def test_06_03_no_objects(self):
        '''Test images with no objects'''
        labels = np.zeros((10, 10), int)
        i, j = np.mgrid[0:10, 0:10]
        image1 = ((i + j) % 2).astype(float)
        image2 = image1.copy()
        i1 = cpi.Image(image1)
        i2 = cpi.Image(image2)
        o = cpo.Objects()
        o.segmented = labels
        workspace, module = self.make_workspace(i1, i2, o)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, OBJECTS_NAME,
                                           "Correlation", "Correlation")
        corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
        self.assertEqual(len(corr), 0)
        self.assertEqual(len(m.get_object_names()), 2)
        self.assertTrue(OBJECTS_NAME in m.get_object_names())
        columns = module.get_measurement_columns(None)
        image_features = m.get_feature_names(cpmeas.IMAGE)
        object_features = m.get_feature_names(OBJECTS_NAME)
        self.assertEqual(len(columns), len(image_features) + len(object_features))
        for column in columns:
            if column[0] == cpmeas.IMAGE:
                self.assertTrue(column[1] in image_features)
            else:
                self.assertEqual(column[0], OBJECTS_NAME)
                self.assertTrue(column[1] in object_features)

    def test_06_04_wrong_size(self):
        '''Regression test of IMG-961 - objects and images of different sizes'''
        np.random.seed(0)
        image1 = np.random.uniform(size=(20, 20))
        i1 = cpi.Image(image1)
        labels = np.zeros((10, 30), int)
        labels[:4, :4] = 1
        labels[6:, 6:] = 2
        o = cpo.Objects()
        o.segmented = labels
        workspace, module = self.make_workspace(i1, i1, o)
        module.run(workspace)
        m = workspace.measurements
        mi = module.get_measurement_images(None, OBJECTS_NAME, "Correlation", "Correlation")
        corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
        self.assertAlmostEqual(corr[0], 1)
        self.assertAlmostEqual(corr[1], 1)

    def test_06_05_last_object_masked(self):
        # Regression test of issue #1553
        # MeasureColocalization was truncating the measurements
        # if the last had no pixels or all pixels masked.
        #
        r = np.random.RandomState()
        r.seed(65)
        image1 = r.uniform(size=(20, 20))
        image2 = r.uniform(size=(20, 20))
        labels = np.zeros((20, 20), int)
        labels[3:8, 3:8] = 1
        labels[13:18, 13:18] = 2
        mask = labels != 2
        objects = cpo.Objects()
        objects.segmented = labels

        for mask1, mask2 in ((mask, None), (None, mask), (mask, mask)):
            workspace, module = self.make_workspace(
                    cpi.Image(image1, mask=mask1),
                    cpi.Image(image2, mask=mask2),
                    objects)
            module.run(workspace)
            m = workspace.measurements
            feature = M.F_CORRELATION_FORMAT % (IMAGE1_NAME, IMAGE2_NAME)
            values = m[OBJECTS_NAME, feature]
            self.assertEqual(len(values), 2)
            self.assertTrue(np.isnan(values[1]))

    def test_zero_valued_intensity(self):
        # https://github.com/CellProfiler/CellProfiler/issues/2680
        image1 = np.zeros((10, 10), dtype=np.float32)

        image2 = np.random.rand(10, 10).astype(np.float32)

        labels = np.zeros_like(image1, dtype=np.uint8)

        labels[5, 5] = 1

        objects = cpo.Objects()

        objects.segmented = labels

        workspace, module = self.make_workspace(cpi.Image(image1), cpi.Image(image2), objects)

        module.run(workspace)

        m = workspace.measurements

        feature = M.F_CORRELATION_FORMAT % (IMAGE1_NAME, IMAGE2_NAME)

        values = m[OBJECTS_NAME, feature]

        assert len(values) == 1

        assert np.isnan(values[0])

    def test_non_overlapping_object_intensity(self):
        # https://github.com/CellProfiler/CellProfiler/issues/2764
        image1 = np.random.rand(10, 10)
        image1[:5, :] = 0

        image2 = np.random.rand(10, 10)
        image2[5:, :] = 0

        objects = cpo.Objects()
        objects.segmented = np.ones_like(image1, dtype=np.uint8)

        workspace, module = self.make_workspace(cpi.Image(image1), cpi.Image(image2), objects)

        module.run(workspace)

        m = workspace.measurements

        feature = M.F_OVERLAP_FORMAT % (IMAGE1_NAME, IMAGE2_NAME)

        values = m[OBJECTS_NAME, feature]

        assert len(values) == 1

        assert values[0] == 0.0

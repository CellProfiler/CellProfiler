import StringIO
import unittest

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.loadsingleimage
import cellprofiler.modules.measureimagequality
import cellprofiler.modules.namesandtypes
import cellprofiler.modules.smooth
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import centrosome.threshold
import numpy

cellprofiler.preferences.set_headless()

MY_IMAGE = "my_image"
MY_OBJECTS = "my_objects"


class TestMeasureImageQuality(unittest.TestCase):
    def make_workspace(self, pixel_data, mask=None, objects=None):
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler.region.Set()
        image = cellprofiler.image.Image(pixel_data)
        if not mask is None:
            image.mask = mask
        image_set.add(MY_IMAGE, image)
        if not objects is None:
            o = cellprofiler.region.Region()
            o.segmented = objects
            object_set.add_objects(o, MY_OBJECTS)
        module = cellprofiler.modules.measureimagequality.MeasureImageQuality()
        module.images_choice.value = cellprofiler.modules.measureimagequality.O_SELECT
        module.image_groups[0].include_image_scalings.value = False
        module.image_groups[0].image_names.value = MY_IMAGE
        module.image_groups[0].use_all_threshold_methods.value = False
        module.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                     object_set,
                                                     cellprofiler.measurement.Measurements(), image_set_list)
        return workspace

    def test_00_00_zeros(self):
        workspace = self.make_workspace(numpy.zeros((100, 100)))
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].check_intensity.value = True
        q.image_groups[0].calculate_threshold.value = False
        q.image_groups[0].scale_groups[0].scale.value = 20
        q.run(workspace)
        m = workspace.measurements
        for feature_name, value in (("ImageQuality_FocusScore_my_image", 0),
                                    ("ImageQuality_LocalFocusScore_my_image_20", 0),
                                    ("ImageQuality_Correlation_my_image_20", 0),
                                    ("ImageQuality_PercentMaximal_my_image", 100),
                                    ("ImageQuality_PercentMinimal_my_image", 100),
                                    ("ImageQuality_PowerLogLogSlope_my_image", 0),
                                    ("ImageQuality_TotalIntensity_my_image", 0),
                                    ("ImageQuality_MeanIntensity_my_image", 0),
                                    ("ImageQuality_MedianIntensity_my_image", 0),
                                    ("ImageQuality_StdIntensity_my_image", 0),
                                    ("ImageQuality_MADIntensity_my_image", 0),
                                    ("ImageQuality_MaxIntensity_my_image", 0),
                                    ("ImageQuality_MinIntensity_my_image", 0)):
            self.assertTrue(m.has_current_measurements(cellprofiler.measurement.IMAGE, feature_name),
                            "Missing feature %s" % feature_name)
            m_value = m.get_current_measurement(cellprofiler.measurement.IMAGE, feature_name)
            if not value is None:
                self.assertEqual(m_value, value,
                                 "Measured value, %f, for feature %s was not %f" %
                                 (m_value, feature_name, value))
        self.features_and_columns_match(m, q)

    def features_and_columns_match(self, measurements, module,
                                   object_name=cellprofiler.measurement.IMAGE):
        self.assertTrue(object_name in measurements.get_object_names())
        features = measurements.get_feature_names(object_name)
        columns = filter((lambda x: x[0] == object_name),
                         module.get_measurement_columns(None))
        self.assertEqual(len(features), len(columns))
        for column in columns:
            self.assertTrue(column[1] in features, 'features_and_columns_match, %s not in %s' % (column[1], features))
            self.assertTrue(column[2] == cellprofiler.measurement.COLTYPE_FLOAT,
                            'features_and_columns_match, %s type not %s' % (column[2], cellprofiler.measurement.COLTYPE_FLOAT))

    def test_00_01_zeros_and_mask(self):
        workspace = self.make_workspace(numpy.zeros((100, 100)),
                                        numpy.zeros((100, 100), bool))
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].check_intensity.value = True
        q.image_groups[0].calculate_threshold.value = False
        q.image_groups[0].scale_groups[0].scale.value = 20
        q.run(workspace)
        m = workspace.measurements
        for feature_name, value in (("ImageQuality_FocusScore_my_image", 0),
                                    ("ImageQuality_LocalFocusScore_my_image_20", 0),
                                    ("ImageQuality_Correlation_my_image_20", 0),
                                    ("ImageQuality_PercentMaximal_my_image", 0),
                                    ("ImageQuality_PercentMinimal_my_image", 0),
                                    ("ImageQuality_PowerLogLogSlope_my_image", 0),
                                    ("ImageQuality_TotalIntensity_my_image", 0),
                                    ("ImageQuality_MeanIntensity_my_image", 0),
                                    ("ImageQuality_MedianIntensity_my_image", 0),
                                    ("ImageQuality_StdIntensity_my_image", 0),
                                    ("ImageQuality_MADIntensity_my_image", 0),
                                    ("ImageQuality_MaxIntensity_my_image", 0),
                                    ("ImageQuality_MinIntensity_my_image", 0)):
            self.assertTrue(m.has_current_measurements(cellprofiler.measurement.IMAGE, feature_name),
                            "Missing feature %s" % feature_name)
            m_value = m.get_current_measurement(cellprofiler.measurement.IMAGE, feature_name)
            self.assertEqual(m_value, value,
                             "Measured value, %f, for feature %s was not %f" % (m_value, feature_name, value))

    def test_01_01_image_blur(self):
        '''Test the focus scores of a random image

        The expected variance of a uniform distribution is 1/12 of the
        difference of the extents (=(0,1)). We divide this by the mean
        and the focus_score should be 1/6

        The local focus score is the variance among the 25 focus scores
        divided by the median focus score. This should be low.
        '''
        numpy.random.seed(0)
        workspace = self.make_workspace(numpy.random.uniform(size=(100, 100)))
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = False
        q.image_groups[0].check_intensity.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.image_groups[0].scale_groups[0].scale.value = 20
        q.run(workspace)
        m = workspace.measurements
        for feature_name, value in (("ImageQuality_FocusScore_my_image", 1.0 / 6.0),
                                    ("ImageQuality_LocalFocusScore_my_image_20", 0),
                                    ("ImageQuality_PercentSaturation_my_image", None),
                                    ("ImageQuality_PercentMaximal_my_image", None)):
            if value is None:
                self.assertFalse(m.has_current_measurements(cellprofiler.measurement.IMAGE, feature_name),
                                 "Feature %s should not be present" % feature_name)
            else:
                self.assertTrue(m.has_current_measurements(cellprofiler.measurement.IMAGE, feature_name),
                                "Missing feature %s" % feature_name)

                m_value = m.get_current_measurement(cellprofiler.measurement.IMAGE, feature_name)
                self.assertAlmostEqual(m_value, value, 2,
                                       "Measured value, %f, for feature %s was not %f" % (m_value, feature_name, value))
        self.features_and_columns_match(m, q)

    def test_01_02_local_focus_score(self):
        '''Test the local focus score by creating one deviant grid block

        Create one grid block out of four that has a uniform value. That one
        should have a focus score of zero. The others have a focus score of
        1/6, so the local focus score should be the variance of (1/6,1/6,1/6,0)
        divided by the median local norm variance (=1/6)
        '''
        expected_value = numpy.var([1.0 / 6.0] * 3 + [0]) * 6.0
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(1000, 1000))
        image[:500, :500] = .5
        workspace = self.make_workspace(image)
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = False
        q.image_groups[0].check_intensity.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.image_groups[0].scale_groups[0].scale.value = 500
        q.run(workspace)
        m = workspace.measurements
        value = m.get_current_measurement(cellprofiler.measurement.IMAGE, "ImageQuality_LocalFocusScore_my_image_500")
        self.assertAlmostEqual(value, expected_value, 3)

    def test_01_03_focus_score_with_mask(self):
        '''Test focus score with a mask to block out an aberrant part of the image'''
        numpy.random.seed(0)
        expected_value = 1.0 / 6.0
        image = numpy.random.uniform(size=(1000, 1000))
        mask = numpy.ones(image.shape, bool)
        mask[400:600, 400:600] = False
        image[mask == False] = .5
        workspace = self.make_workspace(image, mask)
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = False
        q.image_groups[0].check_intensity.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.image_groups[0].scale_groups[0].scale.value = 500
        q.run(workspace)
        m = workspace.measurements
        value = m.get_current_measurement(cellprofiler.measurement.IMAGE, "ImageQuality_FocusScore_my_image")
        self.assertAlmostEqual(value, expected_value, 3)

    def test_01_04_local_focus_score_with_mask(self):
        '''Test local focus score and mask'''
        numpy.random.seed(0)
        expected_value = numpy.var([1.0 / 6.0] * 3 + [0]) * 6.0
        image = numpy.random.uniform(size=(1000, 1000))
        image[:500, :500] = .5
        mask = numpy.ones(image.shape, bool)
        mask[400:600, 400:600] = False
        image[mask == False] = .5
        workspace = self.make_workspace(image, mask)
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = False
        q.image_groups[0].check_intensity.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.image_groups[0].scale_groups[0].scale.value = 500
        q.run(workspace)
        m = workspace.measurements
        value = m.get_current_measurement(cellprofiler.measurement.IMAGE, "ImageQuality_LocalFocusScore_my_image_500")
        self.assertAlmostEqual(value, expected_value, 3)

    def test_02_01_saturation(self):
        '''Test percent saturation'''
        image = numpy.zeros((10, 10))
        image[:5, :5] = 1
        workspace = self.make_workspace(image)
        q = workspace.module
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].check_intensity.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.run(workspace)
        m = workspace.measurements
        for feature_name in ("ImageQuality_ThresholdOtsu_my_image",
                             "ImageQuality_FocusScore_my_image",
                             "ImageQuality_LocalFocusScore_my_image_20"):
            self.assertFalse(m.has_current_measurements(cellprofiler.measurement.IMAGE,
                                                        feature_name),
                             "%s should not be present" % feature_name)
        for (feature_name, expected_value) in (("ImageQuality_PercentMaximal_my_image", 25),
                                               ("ImageQuality_PercentMinimal_my_image", 75)):
            self.assertTrue(m.has_current_measurements(cellprofiler.measurement.IMAGE,
                                                       feature_name))
            self.assertAlmostEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE,
                                                             feature_name),
                                   expected_value)
        self.features_and_columns_match(m, q)

    def test_02_02_maximal(self):
        '''Test percent maximal'''
        image = numpy.zeros((10, 10))
        image[:5, :5] = .5
        expected_value = 100.0 / 4.0
        workspace = self.make_workspace(image)
        q = workspace.module
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].check_intensity.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.run(workspace)
        m = workspace.measurements
        self.assertAlmostEqual(expected_value,
                               m.get_current_measurement(cellprofiler.measurement.IMAGE, "ImageQuality_PercentMaximal_my_image"))

    def test_02_03_saturation_mask(self):
        '''Test percent saturation with mask'''
        image = numpy.zeros((10, 10))
        # 1/2 of image is saturated
        # 1/4 of image is saturated but masked
        image[:5, :] = 1
        mask = numpy.ones((10, 10), bool)
        mask[:5, 5:] = False
        workspace = self.make_workspace(image, mask)
        q = workspace.module
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].check_intensity.value = False
        q.image_groups[0].calculate_threshold.value = False

        q.run(workspace)
        m = workspace.measurements
        for feature_name in ("ImageQuality_ThresholdOtsu_my_image",
                             "ImageQuality_FocusScore_my_image",
                             "ImageQuality_LocalFocusScore_my_image_20"):
            self.assertFalse(m.has_current_measurements(cellprofiler.measurement.IMAGE,
                                                        feature_name),
                             "%s should not be present" % feature_name)
        for (feature_name, expected_value) in (("ImageQuality_PercentMaximal_my_image", 100.0 / 3),
                                               ("ImageQuality_PercentMinimal_my_image", 200.0 / 3)):
            self.assertTrue(m.has_current_measurements(cellprofiler.measurement.IMAGE,
                                                       feature_name))
            print feature_name, expected_value, m.get_current_measurement(cellprofiler.measurement.IMAGE,
                                                                          feature_name)
            self.assertAlmostEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE,
                                                             feature_name),
                                   expected_value)

    def test_02_04_maximal_mask(self):
        '''Test percent maximal with mask'''
        image = numpy.zeros((10, 10))
        image[:5, :5] = .5
        mask = numpy.ones((10, 10), bool)
        mask[:5, 5:] = False
        expected_value = 100.0 / 3.0
        workspace = self.make_workspace(image, mask)
        q = workspace.module
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].check_intensity.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.run(workspace)
        m = workspace.measurements
        self.assertAlmostEqual(expected_value,
                               m.get_current_measurement(cellprofiler.measurement.IMAGE, "ImageQuality_PercentMaximal_my_image"))

    def test_03_01_threshold(self):
        '''Test all thresholding methods

        Use an image that has 1/5 of "foreground" pixels to make MOG
        happy and set the object fraction to 1/5 to test this.
        '''
        numpy.random.seed(0)
        image = numpy.random.beta(2, 5, size=(100, 100))
        object_fraction = .2
        mask = numpy.random.binomial(1, object_fraction, size=(100, 100))
        count = numpy.sum(mask)
        image[mask == 1] = 1.0 - numpy.random.beta(2, 20, size=count)
        #
        # Kapur needs to be quantized
        #
        image = numpy.around(image, 2)

        workspace = self.make_workspace(image)
        q = workspace.module

        for tm, idx in zip(centrosome.threshold.TM_GLOBAL_METHODS,
                           range(len(centrosome.threshold.TM_GLOBAL_METHODS))):
            if idx != 0:
                q.add_image_group()
            q.image_groups[idx].image_names.value = "my_image"
            q.image_groups[idx].include_image_scalings.value = False
            q.image_groups[idx].check_blur.value = False
            q.image_groups[idx].check_saturation.value = False
            q.image_groups[idx].check_intensity.value = False
            q.image_groups[idx].calculate_threshold.value = True
            q.image_groups[idx].use_all_threshold_methods.value = False
            t = q.image_groups[idx].threshold_groups[0]
            t.threshold_method.value = tm
            t.object_fraction.value = object_fraction
            t.two_class_otsu.value = cellprofiler.modules.measureimagequality.O_THREE_CLASS
            t.assign_middle_to_foreground.value = cellprofiler.modules.measureimagequality.O_FOREGROUND
            t.use_weighted_variance.value = cellprofiler.modules.measureimagequality.O_WEIGHTED_VARIANCE
        q.run(workspace)
        m = workspace.measurements
        for feature_name in ("ImageQuality_FocusScore_my_image",
                             "ImageQuality_LocalFocusScore_my_image_20",
                             "ImageQuality_PercentSaturation_my_image",
                             "ImageQuality_PercentMaximal_my_image"):
            self.assertFalse(m.has_current_measurements(cellprofiler.measurement.IMAGE,
                                                        feature_name))
        for tm, idx in zip(centrosome.threshold.TM_GLOBAL_METHODS,
                           range(len(centrosome.threshold.TM_GLOBAL_METHODS))):
            if tm == centrosome.threshold.TM_OTSU_GLOBAL:
                feature_name = "ImageQuality_ThresholdOtsu_my_image_3FW"
            elif tm == centrosome.threshold.TM_MOG_GLOBAL:
                feature_name = "ImageQuality_ThresholdMoG_my_image_20"
            else:
                feature_name = "ImageQuality_Threshold%s_my_image" % tm.split(' ')[0]
            self.assertTrue(m.has_current_measurements(cellprofiler.measurement.IMAGE,
                                                       feature_name))
        self.features_and_columns_match(m, q)

    def test_03_02_experiment_threshold(self):
        '''Test experiment-wide thresholds'''
        numpy.random.seed(32)
        workspace = self.make_workspace(numpy.zeros((10, 10)))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module = workspace.module
        self.assertTrue(isinstance(module, cellprofiler.modules.measureimagequality.MeasureImageQuality))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        image_name = module.image_groups[0].image_names.get_selections()[0]
        feature = module.image_groups[0].threshold_groups[0].threshold_feature_name(image_name)
        data = numpy.random.uniform(size=100)
        m.add_all_measurements(cellprofiler.measurement.IMAGE, feature, data.tolist())
        module.post_run(workspace)

        # Check threshold algorithms
        threshold_group = module.image_groups[0].threshold_groups[0]
        threshold_algorithm = threshold_group.threshold_algorithm
        f_mean, f_median, f_std = [
            threshold_group.threshold_feature_name(image_name, agg)
            for agg in cellprofiler.modules.measureimagequality.AGG_MEAN, cellprofiler.modules.measureimagequality.AGG_MEDIAN, cellprofiler.modules.measureimagequality.AGG_STD]

        expected = ((f_mean, numpy.mean(data)),
                    (f_median, numpy.median(data)),
                    (f_std, numpy.std(data)))
        for feature, expected_value in expected:
            value = m.get_experiment_measurement(feature)
            self.assertAlmostEqual(value, expected_value)

    def test_03_03_experiment_threshold_cycle_skipping(self):
        """Regression test of IMG-970: can you handle nulls in measurements?"""

        numpy.random.seed(33)
        workspace = self.make_workspace(numpy.zeros((10, 10)))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module = workspace.module
        self.assertTrue(isinstance(module, cellprofiler.modules.measureimagequality.MeasureImageQuality))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        image_name = module.image_groups[0].image_names.get_selections()[0]
        feature = module.image_groups[0].threshold_groups[0].threshold_feature_name(image_name)
        data = numpy.random.uniform(size=100)
        dlist = data.tolist()
        #
        # Erase 10 randomly
        #
        eraser = numpy.lexsort([numpy.random.uniform(size=100)])[:10]
        mask = numpy.ones(data.shape, bool)
        mask[eraser] = False
        for e in eraser:
            dlist[e] = None

        m.add_all_measurements(cellprofiler.measurement.IMAGE, feature, dlist)
        module.post_run(workspace)
        self.features_and_columns_match(m, module, cellprofiler.measurement.EXPERIMENT)

        # Check threshold algorithms
        threshold_group = module.image_groups[0].threshold_groups[0]
        threshold_algorithm = threshold_group.threshold_algorithm
        image_name = module.image_groups[0].image_names.value
        f_mean, f_median, f_std = [
            threshold_group.threshold_feature_name(image_name, agg)
            for agg in cellprofiler.modules.measureimagequality.AGG_MEAN, cellprofiler.modules.measureimagequality.AGG_MEDIAN, cellprofiler.modules.measureimagequality.AGG_STD]

        expected = ((f_mean, numpy.mean(data[mask])),
                    (f_median, numpy.median(data[mask])),
                    (f_std, numpy.std(data[mask])))
        for feature, expected_value in expected:
            value = m.get_experiment_measurement(feature)
            self.assertAlmostEqual(value, expected_value)

    def test_03_04_use_all_thresholding_methods(self):
        workspace = self.make_workspace(numpy.zeros((100, 100)))
        q = workspace.module
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = False
        q.image_groups[0].check_intensity.value = False
        q.image_groups[0].calculate_threshold.value = True
        q.image_groups[0].use_all_threshold_methods.value = True
        q.run(workspace)
        m = workspace.measurements
        for feature_name in ['ImageQuality_ThresholdOtsu_my_image_2S',
                             'ImageQuality_ThresholdOtsu_my_image_2W',
                             'ImageQuality_ThresholdOtsu_my_image_3BW',
                             'ImageQuality_ThresholdOtsu_my_image_3BS',
                             'ImageQuality_ThresholdOtsu_my_image_3FS',
                             'ImageQuality_ThresholdOtsu_my_image_3FW',
                             'ImageQuality_ThresholdMoG_my_image_5',
                             'ImageQuality_ThresholdMoG_my_image_75',
                             'ImageQuality_ThresholdMoG_my_image_95',
                             'ImageQuality_ThresholdMoG_my_image_25',
                             'ImageQuality_ThresholdBackground_my_image',
                             'ImageQuality_ThresholdRobustBackground_my_image',
                             'ImageQuality_ThresholdKapur_my_image',
                             'ImageQuality_ThresholdRidlerCalvard_my_image']:
            self.assertTrue(m.has_current_measurements(cellprofiler.measurement.IMAGE,
                                                       feature_name))
        self.features_and_columns_match(m, q)

    def check_error(self, caller, event):
        self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

    def test_04_02_load_saturation_blur(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8925
FromMatlab:True

MeasureImageSaturationBlur:[module_num:1|svn_version:\'8913\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D]
    What did you call the image you want to check for saturation?:Image1
    What did you call the image you want to check for saturation?:Image2
    What did you call the image you want to check for saturation?:Image3
    What did you call the image you want to check for saturation?:Do not use
    What did you call the image you want to check for saturation?:Do not use
    What did you call the image you want to check for saturation?:Do not use
    Do you want to also check the above images for image quality (called blur earlier)?:Yes
    If you chose to check images for image quality above, enter the window size of LocalFocusScore measurement (A suggested value is 2 times ObjectSize)?:25
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureimagequality.MeasureImageQuality))
        self.assertEqual(len(module.image_groups), 3)
        for i in range(3):
            group = module.image_groups[i]
            self.assertEqual(group.image_names, "Image%d" % (i + 1))
            self.assertTrue(group.check_blur)
            self.assertEqual(group.scale_groups[0].scale, 25)
            self.assertTrue(group.check_saturation)
            self.assertFalse(group.calculate_threshold)

    def test_04_03_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9207

MeasureImageQuality:[module_num:1|svn_version:\'9143\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select an image to measure:Alpha
    Check for blur?:Yes
    Window size for blur measurements:25
    Check for saturation?:Yes
    Calculate threshold?:Yes
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.2
    Calculate quartiles and sum of radial power spectrum?:Yes
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select an image to measure:Beta
    Check for blur?:No
    Window size for blur measurements:15
    Check for saturation?:No
    Calculate threshold?:No
    Select a thresholding method:MoG Global
    Typical fraction of the image covered by objects:0.3
    Calculate quartiles and sum of radial power spectrum?:No
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Entropy
    Assign pixels in the middle intensity class to the foreground or the background?:Background
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measureimagequality.MeasureImageQuality))
        self.assertEqual(len(module.image_groups), 2)

        group = module.image_groups[0]
        thr = group.threshold_groups[0]
        self.assertEqual(group.image_names, "Alpha")
        self.assertTrue(group.check_blur)
        self.assertEqual(group.scale_groups[0].scale, 25)
        self.assertTrue(group.check_saturation)
        self.assertTrue(group.calculate_threshold)
        self.assertEqual(thr.threshold_method, cellprofiler.modules.measureimagequality.cpthresh.TM_OTSU)
        self.assertAlmostEqual(thr.object_fraction.value, 0.2)
        self.assertEqual(thr.two_class_otsu, cellprofiler.modules.measureimagequality.O_THREE_CLASS)
        self.assertEqual(thr.use_weighted_variance, cellprofiler.modules.measureimagequality.O_WEIGHTED_VARIANCE)
        self.assertEqual(thr.assign_middle_to_foreground, cellprofiler.modules.measureimagequality.O_FOREGROUND)

        group = module.image_groups[1]
        thr = group.threshold_groups[0]
        self.assertEqual(group.image_names, "Beta")
        self.assertFalse(group.check_blur)
        self.assertEqual(group.scale_groups[0].scale, 15)
        self.assertFalse(group.check_saturation)
        self.assertFalse(group.calculate_threshold)
        self.assertEqual(thr.threshold_method, cellprofiler.modules.measureimagequality.cpthresh.TM_MOG)
        self.assertAlmostEqual(thr.object_fraction.value, 0.3)
        self.assertEqual(thr.two_class_otsu, cellprofiler.modules.measureimagequality.O_TWO_CLASS)
        self.assertEqual(thr.use_weighted_variance, cellprofiler.modules.measureimagequality.O_ENTROPY)
        self.assertEqual(thr.assign_middle_to_foreground, cellprofiler.modules.measureimagequality.O_BACKGROUND)

    def test_04_04_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10908

MeasureImageQuality:[module_num:1|svn_version:\'10368\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Calculate metrics for which images?:All loaded images
    Image count:1
    Scale count:1
    Threshold count:0
    Select the images to measure:Alpha
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Window size for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:Yes

MeasureImageQuality:[module_num:2|svn_version:\'10368\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Calculate metrics for which images?:Select...
    Image count:1
    Scale count:1
    Threshold count:1
    Select the images to measure:Alpha
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Window size for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:Yes
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground

MeasureImageQuality:[module_num:3|svn_version:\'10368\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Calculate metrics for which images?:Select...
    Image count:1
    Scale count:1
    Threshold count:1
    Select the images to measure:Delta,Beta
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Window size for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:Yes
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground

MeasureImageQuality:[module_num:4|svn_version:\'10368\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Calculate metrics for which images?:Select...
    Image count:2
    Scale count:1
    Scale count:1
    Threshold count:1
    Threshold count:1
    Select the images to measure:Delta
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Window size for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the images to measure:Epsilon
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Window size for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:Yes
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground

MeasureImageQuality:[module_num:5|svn_version:\'10368\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Calculate metrics for which images?:Select...
    Image count:1
    Scale count:2
    Threshold count:2
    Select the images to measure:Zeta
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Window size for blur measurements:20
    Window size for blur measurements:30
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 5)
        for module in pipeline.modules():
            self.assertTrue(isinstance(module, cellprofiler.modules.measureimagequality.MeasureImageQuality))

        module = pipeline.modules()[0]
        self.assertEqual(len(module.image_groups), 1)
        group = module.image_groups[0]
        self.assertEqual(group.threshold_groups, [])
        self.assertEqual(module.images_choice, cellprofiler.modules.measureimagequality.O_ALL_LOADED)
        self.assertTrue(group.check_blur)
        self.assertEqual(group.scale_groups[0].scale, 20)
        self.assertTrue(group.check_saturation)
        self.assertTrue(group.check_intensity)
        self.assertTrue(group.calculate_threshold)
        self.assertTrue(group.use_all_threshold_methods)

        module = pipeline.modules()[1]
        self.assertEqual(len(module.image_groups), 1)
        group = module.image_groups[0]
        self.assertEqual(module.images_choice, cellprofiler.modules.measureimagequality.O_SELECT)
        self.assertEqual(group.image_names, "Alpha")

        module = pipeline.modules()[2]
        self.assertEqual(len(module.image_groups), 1)
        group = module.image_groups[0]
        self.assertEqual(module.images_choice, cellprofiler.modules.measureimagequality.O_SELECT)
        self.assertEqual(group.image_names, "Delta,Beta")

        module = pipeline.modules()[3]
        self.assertEqual(len(module.image_groups), 2)
        group = module.image_groups[0]
        self.assertEqual(module.images_choice, cellprofiler.modules.measureimagequality.O_SELECT)
        self.assertEqual(group.image_names, "Delta")
        self.assertTrue(group.check_intensity)
        self.assertFalse(group.use_all_threshold_methods)
        thr = group.threshold_groups[0]
        self.assertEqual(thr.threshold_method, cellprofiler.modules.measureimagequality.cpthresh.TM_OTSU)
        self.assertEqual(thr.use_weighted_variance, cellprofiler.modules.measureimagequality.O_WEIGHTED_VARIANCE)
        self.assertEqual(thr.two_class_otsu, cellprofiler.modules.measureimagequality.O_TWO_CLASS)
        group = module.image_groups[1]
        self.assertEqual(group.image_names, "Epsilon")

        module = pipeline.modules()[4]
        self.assertEqual(len(module.image_groups), 1)
        group = module.image_groups[0]
        self.assertEqual(module.images_choice, cellprofiler.modules.measureimagequality.O_SELECT)
        self.assertEqual(group.image_names, "Zeta")
        self.assertFalse(group.use_all_threshold_methods)
        thr = group.threshold_groups[0]
        self.assertEqual(thr.threshold_method, cellprofiler.modules.measureimagequality.cpthresh.TM_OTSU)
        self.assertEqual(thr.use_weighted_variance, cellprofiler.modules.measureimagequality.O_WEIGHTED_VARIANCE)
        self.assertEqual(thr.two_class_otsu, cellprofiler.modules.measureimagequality.O_TWO_CLASS)
        thr = group.threshold_groups[1]
        self.assertEqual(thr.threshold_method, cellprofiler.modules.measureimagequality.cpthresh.TM_OTSU)
        self.assertEqual(thr.use_weighted_variance, cellprofiler.modules.measureimagequality.O_WEIGHTED_VARIANCE)
        self.assertEqual(thr.two_class_otsu, cellprofiler.modules.measureimagequality.O_THREE_CLASS)
        self.assertEqual(thr.assign_middle_to_foreground, cellprofiler.modules.measureimagequality.O_FOREGROUND)

    def test_05_01_intensity_image(self):
        '''Test operation on a single unmasked image'''
        numpy.random.seed(0)
        pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * .99
        pixels[0:2, 0:2] = 1
        workspace = self.make_workspace(pixels, None)
        q = workspace.module
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = False
        q.image_groups[0].check_intensity.value = True
        q.image_groups[0].calculate_threshold.value = False
        q.run(workspace)
        m = workspace.measurements
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "ImageQuality_TotalIntensity_my_image"),
                         numpy.sum(pixels))
        self.assertEqual(m.get_current_measurement(cellprofiler.measurement.IMAGE, "ImageQuality_MeanIntensity_my_image"),
                         numpy.sum(pixels) / 100.0)
        self.assertEqual(m.get_current_image_measurement('ImageQuality_MinIntensity_my_image'),
                         numpy.min(pixels))
        self.assertEqual(m.get_current_image_measurement('ImageQuality_MaxIntensity_my_image'),
                         numpy.max(pixels))

    def test_06_01_check_image_groups(self):
        workspace = self.make_workspace(numpy.zeros((100, 100)))
        image_set_list = workspace.image_set_list
        image_set = image_set_list.get_image_set(0)
        for i in range(1, 5):
            image_set.add("my_image%s" % i, cellprofiler.image.Image(numpy.zeros((100, 100))))

        q = workspace.module
        # Set my_image1 and my_image2 settings: Saturation only
        q.image_groups[0].image_names.value = "my_image1,my_image2"
        q.image_groups[0].include_image_scalings.value = False
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].check_intensity.value = False
        q.image_groups[0].calculate_threshold.value = False

        # Set my_image3 and my_image4's settings: Blur only
        q.add_image_group()
        q.image_groups[1].image_names.value = "my_image3,my_image4"
        q.image_groups[1].include_image_scalings.value = False
        q.image_groups[1].check_blur.value = True
        q.image_groups[1].check_saturation.value = False
        q.image_groups[1].check_intensity.value = False
        q.image_groups[1].calculate_threshold.value = False
        q.run(workspace)
        m = workspace.measurements

        # Make sure each group of settings has (and *doesn't* have) the correct measures
        for i in [1, 2]:
            for feature_name in (("ImageQuality_PercentMaximal_my_image%s" % i),
                                 ("ImageQuality_PercentMinimal_my_image%s" % i)):
                self.assertTrue(m.has_current_measurements(cellprofiler.measurement.IMAGE, feature_name),
                                "Missing feature %s" % feature_name)

            for feature_name in (("ImageQuality_FocusScore_my_image%s" % i),
                                 ("ImageQuality_LocalFocusScore_my_image%s_20" % i),
                                 ("ImageQuality_PowerLogLogSlope_my_image%s" % i)):
                self.assertFalse(m.has_current_measurements(cellprofiler.measurement.IMAGE, feature_name),
                                 "Erroneously present feature %s" % feature_name)
        for i in [3, 4]:
            for feature_name in (("ImageQuality_FocusScore_my_image%s" % i),
                                 ("ImageQuality_LocalFocusScore_my_image%s_20" % i),
                                 ("ImageQuality_PowerLogLogSlope_my_image%s" % i)):
                self.assertTrue(m.has_current_measurements(cellprofiler.measurement.IMAGE, feature_name),
                                "Missing feature %s" % feature_name)
            for feature_name in (("ImageQuality_PercentMaximal_my_image%s" % i),
                                 ("ImageQuality_PercentMinimal_my_image%s" % i)):
                self.assertFalse(m.has_current_measurements(cellprofiler.measurement.IMAGE, feature_name),
                                 "Erroneously present feature %s" % feature_name)

    def test_06_01_images_to_process(self):
        #
        # Test MeasureImageQuality.images_to_process on a pipeline with a
        # variety of image providers.
        #
        expected_names = ["foo", "bar"]
        pipeline = cellprofiler.pipeline.Pipeline()
        module1 = cellprofiler.modules.namesandtypes.NamesAndTypes()
        module1.module_num = 1
        module1.assignment_method.value = \
            cellprofiler.image.ASSIGN_RULES
        module1.add_assignment()
        module1.add_assignment()
        module1.assignments[0].image_name.value = expected_names[0]
        module1.assignments[0].load_as_choice.value = \
            cellprofiler.image.LOAD_AS_GRAYSCALE_IMAGE
        #
        # TO_DO: issue #652
        #    This test should fail at some later date when we can detect
        #    that an illumination function should not be QA measured
        #
        module1.assignments[1].image_name.value = expected_names[1]
        module1.assignments[1].load_as_choice.value = \
            cellprofiler.image.LOAD_AS_ILLUMINATION_FUNCTION
        module1.assignments[2].load_as_choice.value = \
            cellprofiler.image.LOAD_AS_OBJECTS
        pipeline.add_module(module1)

        module2 = cellprofiler.modules.smooth.Smooth()
        module2.module_num = 2
        module2.image_name.value = expected_names[0]
        module2.filtered_image_name.value = "henry"
        pipeline.add_module(module2)

        miq_module = cellprofiler.modules.measureimagequality.MeasureImageQuality()
        miq_module.module_num = 3
        miq_module.images_choice.value = cellprofiler.modules.measureimagequality.O_ALL_LOADED
        image_names = miq_module.images_to_process(
                miq_module.image_groups[0], None, pipeline)
        self.assertEqual(len(image_names), len(expected_names))
        for image_name in image_names:
            self.assertTrue(image_name in expected_names)

import StringIO
import base64
import unittest
import zlib

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.measuregranularity
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

import sys

print sys.path

IMAGE_NAME = 'myimage'
OBJECTS_NAME = 'myobjects'


class TestMeasureGranularity(unittest.TestCase):
    def test_01_02_load_v1(self):
        '''Load a variable_revision_number=1 pipeline'''
        data = ('eJztWc1u2kAQXhNSNYlUpae0tz2GNljG+WmCqgQaVy1qoCigRFGUthtYwkq2'
                'F/knhVaReuxj9dhH6GP0Eep17BhvHAxOSEuFYWVmPN98O7MzsFrKxfpe8RVc'
                'FyVYLtazLaJiWFWR1aKGloe6tQJ3DYws3IRUz8ND517DHZjbcN753Iv82haU'
                'JWkLJLuEUvmRc/v1FIAHzv2hM1Leo1lPFvoGk2vYsoh+Zs6CNHji6X844wAZ'
                'BJ2q+ACpNjYDCl9f0lu03utcPSrTpq3iCtL6jZ2rYmun2DDft3yg97hKulit'
                'kS+YC8E328fnxCRU9/Cef157xUstjpfl4edCkAeBywPLy1Kfntm/BYF9OiJv'
                'j/vsFz2Z6E1yTpo2UiHR0NnVLJg/KcbfTMjfDFAqRRdXiMEtcvNgo467VvZ1'
                'FzUsqCGr0R6GPx3ykwaSKK8PgxNCOAGsDhnvTXxx8c5z8TJZoVCnFrRNPHy+'
                'UyE/KZCTkuEqNCHfxnB1tgTC8TJZwS1kqxYssSKDCjFww6JGz/W3GeNvlvPH'
                '5GLDafuIvD3g8P7l4+fA8Pm+vt6rq0n64sjpqiQ4SVy/3/WV77ee5Ij6Tbp+'
                'o+AKMfOcA+F6Y/JuG+k6VnPZO+BP+r06Lj7++zA35vj4vqpQHd+G71sM3zsQ'
                'Xk8mf1jeqb5kGxu8LT7PfGTSIVbVffp5+7iYrZ5kfM0uVW1N3z6WslsnX3Mr'
                '8sWlcY04SFeZiYx7lPm3Y+a/yc2fyWwORxgZ3sTWLjJZpipT3Wp7OtnTKagX'
                'aP5S38h30Tf/6u8E3z/ymPmm/TOZ/dOdH20/P64+jdqPupv/M4PandvzT3GT'
                'jSuAaf1Mcf8/rgAG13nUOUVQ55DoTdyZpHinuMnGFcC0Xqe45LhF4eb9J38+'
                'w+w/gcH19gyE643JDWfL3zEo+9/AEDX3cNsUVYqal6fL4p7zsdR30Ozui2N4'
                'FI5HuYlHw8i0DezVPNJtFRnE6onlS73L+ybQR+VzLoK/Py8p57U0P3gd+PwH'
                '6/J7JwlfWrjOtxCDS3uZZLjvYLR1Xx5g78eW1P4PGItAGQ==')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        self.assertEqual(len(module.images), 2)
        for image_setting, image_name, subsample_size, bsize, elsize, glen in \
                ((module.images[0], 'DNA', .25, .25, 10, 16),
                 (module.images[1], 'Actin', .33, .50, 12, 20)):
            # self.assertTrue(isinstance(image_setting, M.MeasureGranularity))
            self.assertEqual(image_setting.image_name, image_name)
            self.assertAlmostEqual(image_setting.subsample_size.value, subsample_size)
            self.assertAlmostEqual(image_setting.image_sample_size.value, bsize)
            self.assertEqual(image_setting.element_size.value, elsize)
            self.assertEqual(image_setting.granular_spectrum_length.value, glen)
            self.assertEqual(len(image_setting.objects), 0)
            self.assertEqual(image_setting.object_count.value, 0)

    def test_01_03_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10252

MeasureGranularity:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Image count:2
    Object count:2
    Select an image to measure:DNA
    Subsampling factor for granularity measurements:0.25
    Subsampling factor for background reduction:0.25
    Radius of structuring element:10
    Range of the granular spectrum:16
    Object name:Nuclei
    Object name:Cells
    Object count:3
    Select an image to measure:Actin
    Subsampling factor for granularity measurements:0.33
    Subsampling factor for background reduction:0.5
    Radius of structuring element:12
    Range of the granular spectrum:20
    Object name:Nuclei
    Object name:Cells
    Object name:Cytoplasm
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        self.assertEqual(len(module.images), 2)
        for image_setting, image_name, subsample_size, bsize, elsize, glen, objs, in \
                ((module.images[0], 'DNA', .25, .25, 10, 16, ("Nuclei", "Cells")),
                 (module.images[1], 'Actin', .33, .50, 12, 20, ("Nuclei", "Cells", "Cytoplasm"))):
            # self.assertTrue(isinstance(image_setting, M.MeasureGranularity))
            self.assertEqual(image_setting.image_name, image_name)
            self.assertAlmostEqual(image_setting.subsample_size.value, subsample_size)
            self.assertAlmostEqual(image_setting.image_sample_size.value, bsize)
            self.assertEqual(image_setting.element_size.value, elsize)
            self.assertEqual(image_setting.granular_spectrum_length.value, glen)
            self.assertEqual(len(image_setting.objects), len(objs))
            self.assertEqual(image_setting.object_count.value, len(objs))
            self.assertTrue(all([ob.objects_name.value in objs
                                 for ob in image_setting.objects]))

    def make_pipeline(self, image, mask, subsample_size, image_sample_size,
                      element_size, granular_spectrum_length,
                      labels=None):
        '''Make a pipeline with a MeasureGranularity module

        image - measure granularity on this image
        mask - exclude / include pixels from measurement. None = no mask
        subsample_size, etc. - values for corresponding settings in the module
        returns tuple of module & workspace
        '''
        module = cellprofiler.modules.measuregranularity.MeasureGranularity()
        module.module_num = 1
        image_setting = module.images[0]
        # assert isinstance(image_setting, M.MeasureGranularity)
        image_setting.image_name.value = IMAGE_NAME
        image_setting.subsample_size.value = subsample_size
        image_setting.image_sample_size.value = image_sample_size
        image_setting.element_size.value = element_size
        image_setting.granular_spectrum_length.value = granular_spectrum_length
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        img = cellprofiler.image.Image(image, mask)
        image_set.add(IMAGE_NAME, img)
        pipeline = cellprofiler.pipeline.Pipeline()

        def error_callback(event, caller):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(error_callback)
        pipeline.add_module(module)
        object_set = cellprofiler.region.Set()
        if labels is not None:
            objects = cellprofiler.region.Region()
            objects.segmented = labels
            object_set.add_objects(objects, OBJECTS_NAME)
            image_setting.add_objects()
            image_setting.objects[0].objects_name.value = OBJECTS_NAME
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                     object_set, cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        return module, workspace

    def test_02_00_all_masked(self):
        '''Run on a totally masked image'''
        module, workspace = self.make_pipeline(numpy.zeros((40, 40)),
                                               numpy.zeros((40, 40), bool),
                                               .25, .25, 10, 16)
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertTrue(numpy.isnan(value))

    def test_02_01_zeros(self):
        '''Run on an image of all zeros'''
        module, workspace = self.make_pipeline(numpy.zeros((40, 40)), None,
                                               .25, .25, 10, 16)
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertAlmostEqual(value, 0)

    def test_03_01_no_scaling(self):
        '''Run on an image without subsampling or background scaling'''
        #
        # Make an image with granularity at scale 1
        #
        i, j = numpy.mgrid[0:10, 0:10]
        image = (i % 2 == j % 2).astype(float)
        expected = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        module, workspace = self.make_pipeline(image, None,
                                               1, 1, 10, 16)
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertAlmostEqual(value, expected[i - 1])

    def test_03_02_subsampling(self):
        '''Run on an image with subsampling'''
        #
        # Make an image with granularity at scale 2
        #
        i, j = numpy.mgrid[0:80, 0:80]
        image = ((i / 8).astype(int) % 2 == (j / 8).astype(int) % 2).astype(float)
        #
        # The 4x4 blocks need two erosions before disappearing. The corners
        # need an additional two erosions before disappearing
        #
        expected = [0, 96, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        module, workspace = self.make_pipeline(image, None,
                                               .5, 1, 10, 16)
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertAlmostEqual(value, expected[i - 1])

    def test_03_03_background_sampling(self):
        '''Run on an image with background subsampling'''
        #
        # Make an image with granularity at scale 2
        #
        i, j = numpy.mgrid[0:80, 0:80]
        image = ((i / 4).astype(int) % 2 == (j / 4).astype(int) % 2).astype(float)
        #
        # Add in a background offset
        #
        image = image * .5 + .5
        #
        # The 4x4 blocks need two erosions before disappearing. The corners
        # need an additional two erosions before disappearing
        #
        expected = [0, 99, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        module, workspace = self.make_pipeline(image, None,
                                               1, .5, 10, 16)
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertAlmostEqual(value, expected[i - 1])

    def test_04_01_filter_background(self):
        '''Run on an image, filtering out the background

        This test makes sure that the grey_closing happens correctly
        over the user-specified radius.
        '''
        #
        # Make an image with granularity at scale 2
        #
        i, j = numpy.mgrid[0:80, 0:80]
        image = ((i / 4).astype(int) % 2 == (j / 4).astype(int) % 2).astype(float)
        #
        # Scale the pixels down so we have some dynamic range and offset
        # so the background is .2
        #
        image = image * .5 + .2
        #
        # Paint all background pixels on the edge and 1 in to be 0
        #
        image[:, :2][image[:, :2] < .5] = 0
        #
        # Paint all of the foreground pixels on the edge to be .5
        #
        image[:, 0][image[:, 0] > .5] = .5
        #
        # The pixel at 0,0 doesn't get a background of zero
        #
        image[0, 0] = .7
        #
        # The 4x4 blocks need two erosions before disappearing. The corners
        # need an additional two erosions before disappearing
        #
        expected = [0, 99, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        module, workspace = self.make_pipeline(image, None,
                                               1, 1, 5, 16)
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertAlmostEqual(value, expected[i - 1])

    def test_05_01_all_masked(self):
        '''Run on objects and a totally masked image'''
        labels = numpy.ones((40, 40), int)
        labels[20:, :] = 2
        module, workspace = self.make_pipeline(numpy.zeros((40, 40)),
                                               numpy.zeros((40, 40), bool),
                                               .25, .25, 10, 16,
                                               labels)
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertTrue(numpy.isnan(value))
            values = m.get_current_measurement(OBJECTS_NAME,
                                               feature)
            self.assertEqual(len(values), 2)
            self.assertTrue(numpy.all(numpy.isnan(values)) or numpy.all(values == 0))

    def test_05_02_no_objects(self):
        '''Run on a labels matrix with no objects'''
        module, workspace = self.make_pipeline(numpy.zeros((40, 40)),
                                               None,
                                               .25, .25, 10, 16,
                                               numpy.zeros((40, 40), int))
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertAlmostEqual(value, 0)
            values = m.get_current_measurement(OBJECTS_NAME,
                                               feature)
            self.assertEqual(len(values), 0)

    def test_05_03_zeros(self):
        '''Run on an image of all zeros'''
        labels = numpy.ones((40, 40), int)
        labels[20:, :] = 2
        module, workspace = self.make_pipeline(numpy.zeros((40, 40)), None,
                                               .25, .25, 10, 16, labels)
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertAlmostEqual(value, 0)
            values = m.get_current_measurement(OBJECTS_NAME, feature)
            self.assertEqual(len(values), 2)
            numpy.testing.assert_almost_equal(values, 0)

    def test_06_01_no_scaling(self):
        '''Run on an image without subsampling or background scaling'''
        #
        # Make an image with granularity at scale 1
        #
        i, j = numpy.mgrid[0:40, 0:30]
        image = (i % 2 == j % 2).astype(float)
        expected = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        labels = numpy.ones((40, 30), int)
        labels[20:, :] = 2
        module, workspace = self.make_pipeline(image, None,
                                               1, 1, 10, 16, labels)
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertAlmostEqual(value, expected[i - 1])
            values = m.get_current_measurement(OBJECTS_NAME, feature)
            self.assertEqual(len(values), 2)
            numpy.testing.assert_almost_equal(values, expected[i - 1])

    def test_06_02_subsampling(self):
        '''Run on an image with subsampling'''
        #
        # Make an image with granularity at scale 2
        #
        i, j = numpy.mgrid[0:80, 0:80]
        image = ((i / 8).astype(int) % 2 == (j / 8).astype(int) % 2).astype(float)
        #
        # The 4x4 blocks need two erosions before disappearing. The corners
        # need an additional two erosions before disappearing
        #
        expected = [0, 96, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        labels = numpy.ones((80, 80), int)
        labels[40:, :] = 2
        module, workspace = self.make_pipeline(image, None,
                                               .5, 1, 10, 16, labels)
        self.assertTrue(isinstance(module, cellprofiler.modules.measuregranularity.MeasureGranularity))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        for i in range(1, 16):
            feature = module.images[0].granularity_feature(i)
            self.assertTrue(feature in m.get_feature_names(cellprofiler.measurement.IMAGE))
            value = m.get_current_image_measurement(feature)
            self.assertAlmostEqual(value, expected[i - 1])
            values = m.get_current_measurement(OBJECTS_NAME, feature)
            self.assertEqual(len(values), 2)
            #
            # We rescale the downscaled image to the size of the labels
            # and this throws the images off during interpolation
            #
            numpy.testing.assert_almost_equal(values, expected[i - 1], 0)

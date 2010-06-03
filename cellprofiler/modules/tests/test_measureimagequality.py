'''test_measureimagequality.py - test the MeasureImageQuality module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

from scipy.io.matlab import savemat
import base64
import numpy as np
import StringIO
import unittest

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.modules.measureimagequality as miq
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpmeas
import cellprofiler.cpmath.threshold as cpthresh

MY_IMAGE = "my_image"
class TestMeasureImageQuality(unittest.TestCase):
    def make_workspace(self, pixel_data, mask=None):
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image = cpi.Image(pixel_data)
        if not mask is None:
            image.mask = mask
        image_set.add(MY_IMAGE, image)
        module = miq.MeasureImageQuality()
        module.image_groups[0].image_name.value = MY_IMAGE
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  cpo.ObjectSet(),
                                  cpmeas.Measurements(),image_set_list)
        return workspace
    
    def test_00_00_zeros(self):
        workspace = self.make_workspace(np.zeros((100,100)))
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].calculate_threshold.value = True
        q.image_groups[0].compute_power_spectrum.value = True
        q.image_groups[0].window_size.value = 20
        q.run(workspace)
        m = workspace.measurements
        for feature_name,value in (("ImageQuality_FocusScore_my_image_20",0),
                                   ("ImageQuality_LocalFocusScore_my_image_20",0),
                                   ("ImageQuality_PercentMaximal_my_image",100),
                                   ("ImageQuality_PercentMinimal_my_image",100),
                                   ("ImageQuality_MagnitudeLogLogSlope_my_image", 0),
                                   ("ImageQuality_PowerLogLogSlope_my_image", 0)):
            self.assertTrue(m.has_current_measurements(cpmeas.IMAGE,feature_name),
                            "Missing feature %s"%feature_name)
            m_value =m.get_current_measurement(cpmeas.IMAGE, feature_name)
            if not value is None:
                self.assertEqual(m_value, value,
                                 "Measured value, %f, for feature %s was not %f" %
                                (m_value, feature_name, value))
        self.features_and_columns_match(m, q)
                
    def features_and_columns_match(self, measurements, module):
        self.assertEqual(len(measurements.get_object_names()),1)
        self.assertEqual(measurements.get_object_names()[0],cpmeas.IMAGE)
        features = measurements.get_feature_names(cpmeas.IMAGE)
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(features), len(columns))
        for column in columns:
            self.assertTrue(column[0] == cpmeas.IMAGE, 'features_and_columns_match, %s not %s'%(column[0], cpmeas.IMAGE))
            self.assertTrue(column[1] in features, 'features_and_columns_match, %s not in %s'%(column[1], features))
            self.assertTrue(column[2] == cpmeas.COLTYPE_FLOAT, 'features_and_columns_match, %s type not %s'%(column[2], cpmeas.COLTYPE_FLOAT))
            
    def test_00_01_zeros_and_mask(self):
        workspace = self.make_workspace(np.zeros((100,100)),
                                        np.zeros((100,100),bool))
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].calculate_threshold.value = True
        q.image_groups[0].compute_power_spectrum.value = True
        q.image_groups[0].window_size.value = 20
        q.run(workspace)
        m = workspace.measurements
        for feature_name, value in (("ImageQuality_FocusScore_my_image_20", 0),
                                    ("ImageQuality_LocalFocusScore_my_image_20", 0),
                                    ("ImageQuality_PercentMaximal_my_image", 0),
                                    ("ImageQuality_PercentMinimal_my_image", 0),
                                    ("ImageQuality_MagnitudeLogLogSlope_my_image", 0),
                                   ("ImageQuality_PowerLogLogSlope_my_image", 0)):
            self.assertTrue(m.has_current_measurements(cpmeas.IMAGE,feature_name), 
                            "Missing feature %s"%feature_name)
            m_value =m.get_current_measurement(cpmeas.IMAGE, feature_name)
            self.assertEqual(m_value, value, "Measured value, %f, for feature %s was not %f"%(m_value, feature_name, value))
    
    def test_01_01_image_blur(self):
        '''Test the focus scores of a random image
        
        The expected variance of a uniform distribution is 1/12 of the
        difference of the extents (=(0,1)). We divide this by the mean
        and the focus_score should be 1/6
        
        The local focus score is the variance among the 25 focus scores
        divided by the median focus score. This should be low.
        '''
        np.random.seed(0)
        workspace = self.make_workspace(np.random.uniform(size=(100,100)))
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.image_groups[0].window_size.value = 20
        q.run(workspace)
        m = workspace.measurements
        for feature_name, value in (("ImageQuality_FocusScore_my_image_20", 1.0/6.0),
                                    ("ImageQuality_LocalFocusScore_my_image_20", 0),
                                    ("ImageQuality_PercentSaturation_my_image", None),
                                    ("ImageQuality_PercentMaximal_my_image", None)):
            if value is None:
                self.assertFalse(m.has_current_measurements(cpmeas.IMAGE,feature_name), 
                                 "Feature %s should not be present"%feature_name)
            else:
                self.assertTrue(m.has_current_measurements(cpmeas.IMAGE,feature_name), 
                                 "Missing feature %s"%feature_name)
                
                m_value =m.get_current_measurement(cpmeas.IMAGE, feature_name)
                self.assertAlmostEqual(m_value, value, 2, 
                                       "Measured value, %f, for feature %s was not %f"%(m_value, feature_name, value))
        self.features_and_columns_match(m, q)

    def test_01_02_local_focus_score(self):
        '''Test the local focus score by creating one deviant grid block
        
        Create one grid block out of four that has a uniform value. That one 
        should have a focus score of zero. The others have a focus score of
        1/6, so the local focus score should be the variance of (1/6,1/6,1/6,0)
        divided by the median local norm variance (=1/6)
        '''
        expected_value = np.var([1.0/6.0]*3+[0])*6.0
        np.random.seed(0)
        image = np.random.uniform(size=(1000,1000))
        image[:500,:500] = .5
        workspace = self.make_workspace(image)
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.image_groups[0].window_size.value = 500
        q.run(workspace)
        m = workspace.measurements
        value = m.get_current_measurement(cpmeas.IMAGE, "ImageQuality_LocalFocusScore_my_image_500")
        self.assertAlmostEqual(value, expected_value,3)
    
    def test_01_03_focus_score_with_mask(self):
        '''Test focus score with a mask to block out an aberrant part of the image'''
        np.random.seed(0)
        expected_value = 1.0/6.0
        image = np.random.uniform(size=(1000,1000))
        mask = np.ones(image.shape, bool)
        mask[400:600,400:600] = False
        image[mask==False] = .5
        workspace = self.make_workspace(image, mask)
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.image_groups[0].window_size.value = 500
        q.run(workspace)
        m = workspace.measurements
        value = m.get_current_measurement(cpmeas.IMAGE, "ImageQuality_FocusScore_my_image_500")
        self.assertAlmostEqual(value, expected_value,3)
        
    def test_01_04_local_focus_score_with_mask(self):
        '''Test local focus score and mask'''
        np.random.seed(0)
        expected_value = np.var([1.0/6.0]*3+[0])*6.0
        image = np.random.uniform(size=(1000,1000))
        image[:500,:500] = .5
        mask = np.ones(image.shape, bool)
        mask[400:600,400:600] = False
        image[mask==False] = .5
        workspace = self.make_workspace(image, mask)
        q = workspace.module
        q.image_groups[0].check_blur.value = True
        q.image_groups[0].check_saturation.value = False
        q.image_groups[0].calculate_threshold.value = False
        q.image_groups[0].window_size.value = 500
        q.run(workspace)
        m = workspace.measurements
        value = m.get_current_measurement(cpmeas.IMAGE, "ImageQuality_LocalFocusScore_my_image_500")
        self.assertAlmostEqual(value, expected_value,3)
        
    
    def test_02_01_saturation(self):
        '''Test percent saturation'''
        image = np.zeros((10,10))
        image[:5,:5] = 1
        workspace = self.make_workspace(image)
        q = workspace.module
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].calculate_threshold.value = False
        q.run(workspace)
        m = workspace.measurements
        for feature_name in ("ImageQuality_ThresholdOtsu_my_image",
                             "ImageQuality_FocusScore_my_image_20",
                             "ImageQuality_LocalFocusScore_my_image_20"):
            self.assertFalse(m.has_current_measurements(cpmeas.IMAGE,
                                                        feature_name),
                             "%s should not be present"%feature_name)
        for (feature_name, expected_value) in (("ImageQuality_PercentMaximal_my_image", 25),
                                               ("ImageQuality_PercentMinimal_my_image", 75)):
            self.assertTrue(m.has_current_measurements(cpmeas.IMAGE,
                                                        feature_name))
            self.assertAlmostEqual(m.get_current_measurement(cpmeas.IMAGE, 
                                                             feature_name),
                                   expected_value)
        self.features_and_columns_match(m, q)
    
    def test_02_02_maximal(self):
        '''Test percent maximal'''
        image = np.zeros((10,10))
        image[:5,:5] = .5
        expected_value = 100.0 / 4.0
        workspace = self.make_workspace(image)
        q = workspace.module
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].calculate_threshold.value = False
        q.run(workspace)
        m = workspace.measurements
        self.assertAlmostEqual(expected_value, m.get_current_measurement(cpmeas.IMAGE, "ImageQuality_PercentMaximal_my_image"))
        
    def test_02_03_saturation_mask(self):
        '''Test percent saturation with mask'''
        image = np.zeros((10,10))
        # 1/2 of image is saturated
        # 1/4 of image is saturated but masked
        image[:5,:] = 1
        mask = np.ones((10,10),bool)
        mask[:5,5:] = False
        workspace = self.make_workspace(image, mask)
        q = workspace.module
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].calculate_threshold.value = False
        
        q.run(workspace)
        m = workspace.measurements
        for feature_name in ("ImageQuality_ThresholdOtsu_my_image",
                             "ImageQuality_FocusScore_my_image_20",
                             "ImageQuality_LocalFocusScore_my_image_20"):
            self.assertFalse(m.has_current_measurements(cpmeas.IMAGE,
                                                        feature_name),
                             "%s should not be present"%feature_name)
        for (feature_name, expected_value) in (("ImageQuality_PercentMaximal_my_image", 100.0/3),
                                               ("ImageQuality_PercentMinimal_my_image", 200.0/3)):
            self.assertTrue(m.has_current_measurements(cpmeas.IMAGE,
                                                        feature_name))
            print feature_name, expected_value, m.get_current_measurement(cpmeas.IMAGE, 
                                                             feature_name)
            self.assertAlmostEqual(m.get_current_measurement(cpmeas.IMAGE, 
                                                             feature_name),
                                   expected_value)
    
    def test_02_04_maximal_mask(self):
        '''Test percent maximal with mask'''
        image = np.zeros((10,10))
        image[:5,:5] = .5
        mask = np.ones((10,10),bool)
        mask[:5,5:] = False
        expected_value = 100.0 / 3.0
        workspace = self.make_workspace(image, mask)
        q = workspace.module
        q.image_groups[0].check_blur.value = False
        q.image_groups[0].check_saturation.value = True
        q.image_groups[0].calculate_threshold.value = False
        q.run(workspace)
        m = workspace.measurements
        self.assertAlmostEqual(expected_value, m.get_current_measurement(cpmeas.IMAGE, "ImageQuality_PercentMaximal_my_image"))
    
    def test_03_01_threshold(self):
        '''Test all thresholding methods
        
        Use an image that has 1/5 of "foreground" pixels to make MOG
        happy and set the object fraction to 1/5 to test this.
        '''
        np.random.seed(0)
        image = np.random.beta(2, 5, size=(100,100))
        object_fraction = .2
        mask = np.random.binomial(1, object_fraction, size=(100,100))
        count = np.sum(mask)
        image[mask==1] = 1.0 - np.random.beta(2,20,size=count)
        #
        # Kapur needs to be quantized
        #
        image = np.around(image, 2)
        
        workspace = self.make_workspace(image)
        q = workspace.module
        for tm,idx in zip(cpthresh.TM_GLOBAL_METHODS,
                          range(len(cpthresh.TM_GLOBAL_METHODS))):
            if idx != 0:
                q.add_image_group()
            q.image_groups[idx].image_name.value = "my_image"
            q.image_groups[idx].check_blur.value = False
            q.image_groups[idx].check_saturation.value = False
            q.image_groups[idx].calculate_threshold.value = True
            q.image_groups[idx].compute_power_spectrum.value = False
            q.image_groups[idx].threshold_method.value = tm
            q.image_groups[idx].object_fraction.value = object_fraction
            q.image_groups[idx].two_class_otsu.value = miq.O_THREE_CLASS
            q.image_groups[idx].assign_middle_to_foreground.value = miq.O_FOREGROUND
            q.image_groups[idx].use_weighted_variance.value = miq.O_WEIGHTED_VARIANCE
        q.run(workspace)
        m = workspace.measurements
        for feature_name in ("ImageQuality_FocusScore_my_image_20",
                             "ImageQuality_LocalFocusScore_my_image_20",
                             "ImageQuality_PercentSaturation_my_image",
                             "ImageQuality_PercentMaximal_my_image"):
            self.assertFalse(m.has_current_measurements(cpmeas.IMAGE,
                                                        feature_name)) 
        for tm,idx in zip(cpthresh.TM_GLOBAL_METHODS,
                          range(len(cpthresh.TM_GLOBAL_METHODS))):
            if tm == cpthresh.TM_OTSU_GLOBAL:
                feature_name = "ImageQuality_ThresholdOtsu_my_image_3FW"
            elif tm == cpthresh.TM_MOG_GLOBAL:
                feature_name = "ImageQuality_ThresholdMoG_my_image_20"
            else:
                feature_name = "ImageQuality_Threshold%s_my_image"%tm.split(' ')[0]
            self.assertTrue(m.has_current_measurements(cpmeas.IMAGE,
                                                       feature_name)) 
        self.features_and_columns_match(m, q)

    def check_error(self, caller, event):
        self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

    def test_04_01_load_matlab_pipeline(self):
        p = cpp.Pipeline()
        p.add_listener(self.check_error)
        data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBXZWQgQXByIDE1IDE1OjI2OjU0IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAFwIAAHic7VbNTuMwEJ60aVV2RWDLZSX2kONedpefC8eyArGVaMufkJD24rZusJTGVeJULU/Ao/AYvAMvsY+wR2xIUscCknojKNI6sqxx5vtmxh57bAHAnxWAKh9rvJfgsVUi2ZC6kE8xY8RzggqY8Dmav+X9HPkEdV18jtwQB5C0eL7pDejZdJT8atF+6OI2GsrKvLXDYRf7QWcQA6PfR2SC3VNyhSHdYrUTPCYBoV6Ej/jV2cQuZYpdi/eb2mwdDGUdxLrUpXmh34CZvvnEuq1K+qtRP8MT9m1/gnrMHiLWuxQ8Gxk85RRPGS6473lwZgpnwt7uUTMPrpTClWBr4zHenQxcDdLxCrnjE+cnT4ki8Fnr/UHBC3mP2h5ldhhEibNIfujwtOiBfeDSLnL/jacof/7zzHg+KjxC7rAglDdsEePKug+MFI8B2wsaRxZP3v35lcHzSeERMvH6ZEz6IXJtMkROUl1ec5/Ue7tN3zZPvueMv6hzs6TwCNnx0TToITd+N1jSqFt38+Kf2w9d+94PlML/Lb/8XlmW5q0cYyPDn6feMw/J7vg0HNn8COCRbr7NeOb3Ky9fUfEtun9F+TNvnO89P3Tjfit/b+H586/eP7p2DynqN6WClqc+rik8Qm5hFIQ+fqA65lWSsGnCJ9+D1Yw4SvyrW3r1aFPTXsWYH2fy7/eXu3WBu4b59unrC/px09W/B+9W/qo='
        p.load(StringIO.StringIO(base64.b64decode(data)))
        self.assertEqual(len(p.modules()),2)
        q = p.modules()[1]
        self.assertEqual(len(q.image_groups),1)
        ig = q.image_groups[0]
        self.assertEqual(ig.image_name.value,'OrigBlue')
        self.assertTrue(ig.check_blur.value)
        self.assertEqual(ig.window_size.value, 20)
        self.assertTrue(ig.check_saturation.value)
        self.assertTrue(ig.calculate_threshold.value)
        self.assertEqual(ig.threshold_algorithm, cpthresh.TM_MOG)
        
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
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, miq.MeasureImageQuality))
        self.assertEqual(len(module.image_groups),3)
        for i in range(3):
            group = module.image_groups[i]
            self.assertEqual(group.image_name, "Image%d"%(i+1))
            self.assertTrue(group.check_blur)
            self.assertEqual(group.window_size, 25)
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
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, miq.MeasureImageQuality))
        self.assertEqual(len(module.image_groups),2)
        
        group = module.image_groups[0]
        self.assertEqual(group.image_name, "Alpha")
        self.assertTrue(group.check_blur)
        self.assertEqual(group.window_size, 25)
        self.assertTrue(group.check_saturation)
        self.assertTrue(group.calculate_threshold)
        self.assertEqual(group.threshold_method, miq.cpthresh.TM_OTSU_GLOBAL)
        self.assertAlmostEqual(group.object_fraction.value, 0.2)
        self.assertTrue(group.compute_power_spectrum)
        self.assertEqual(group.two_class_otsu, miq.O_THREE_CLASS)
        self.assertEqual(group.use_weighted_variance, miq.O_WEIGHTED_VARIANCE)
        self.assertEqual(group.assign_middle_to_foreground, miq.O_FOREGROUND)
        
        group = module.image_groups[1]
        self.assertEqual(group.image_name, "Beta")
        self.assertFalse(group.check_blur)
        self.assertEqual(group.window_size, 15)
        self.assertFalse(group.check_saturation)
        self.assertFalse(group.calculate_threshold)
        self.assertEqual(group.threshold_method, miq.cpthresh.TM_MOG_GLOBAL)
        self.assertAlmostEqual(group.object_fraction.value, 0.3)
        self.assertFalse(group.compute_power_spectrum)
        self.assertEqual(group.two_class_otsu, miq.O_TWO_CLASS)
        self.assertEqual(group.use_weighted_variance, miq.O_ENTROPY)
        self.assertEqual(group.assign_middle_to_foreground, miq.O_BACKGROUND)
        
        
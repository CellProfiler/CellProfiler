'''test_calculateimageoverlap - test the CalculateImageOverlap module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision: 1 $"

import base64
import numpy as np
import os
import tempfile
from StringIO import StringIO
import unittest
import zlib

from cellprofiler.preferences import set_headless
set_headless()

from cellprofiler.modules.tests import example_images_directory
import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.calculateimageoverlap as C

GROUND_TRUTH_IMAGE_NAME = 'groundtruth'
TEST_IMAGE_NAME = 'test'
O_IMG = 'Foreground/background segmentation'
O_OBJ = 'Segmented objects'
GROUND_TRUTH_OBJ_IMAGE_NAME  = 'DNA'
ID_OBJ_IMAGE_NAME = 'Protein'
GROUND_TRUTH_OBJ = 'Nuclei'
ID_OBJ = 'Protein'



class TestCalculateImageOverlap(unittest.TestCase):
    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUggpTVXwSsxTMDJTMDSzMrWwMrRQMDIwNFAgGTAwevryMzAwnGJk'
                'YKiY83TqWa8jBgINy++G/WEMdBSwubFY/L/AkcULmpredek1Hd9x9prDrSlm'
                'sWW1TPLH3T/01rAK5+7OjHl7fpX4cp7TS4zj1Sa/fKPE8EaP4UzH/ciFajUd'
                'c2+1Lbq07+CcyQe6czck6MbWC1Z1KzW909znanLQ8MmktfPZa6/Jtv5c7f1M'
                'TWT7xPVW3Jnq/2z7HvnZc/vEx09UXGb+RNA50Gh7to2CYaKj1PxPX6/WxB/b'
                'Wiveanf7sZVElGxKbNiU0pf1LMn12U53p4n8Xxqyu+LEdr/2Fcdl5T/ecxY9'
                '13Rs4jbxjSoFlVM+tNvMvyvUF/ogYoMsW9xkC7GJKx9JPT/tGd/7f+Wu0Kdn'
                's20y5xuoJ9zxTOL5Py+8bL5bY/qBf/rPi8MORV+ruKTW6M12xFegtKN/T43X'
                'FcvlkQ9mXUi7fDVzd9rcZ1uKPW93X9B4p9gl6ne1Joo1jvudS+i/TXK//h3k'
                '0zoiP+v++dytcpe/3a4OPqEo9E9ufY/xy/rUM4Uu3I8Dzq7Om/e/9z9T/Pf+'
                'e/t93/9sUvweI6kR+uzyqmMTvxku2tui1Hc/cS5jnwbXw8/lJsFFfME22mmb'
                'K7dcL7Df7ThHdV5z7/VSye8sX99lfpb+7zPr135GNddNRQBlKwLx')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, C.CalculateImageOverlap))
        self.assertEqual(module.ground_truth, "groundtruth")
        self.assertEqual(module.test_img, "orig")
        
    def test_01_02_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9169

CalculateImageOverlap:[module_num:1|svn_version:\'9000\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Which image do you want to use as the basis for calculating the amount of overlap? :GroundTruth
    Which image do you want to compare for overlap?:Segmentation
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, C.CalculateImageOverlap))
        self.assertEqual(module.ground_truth, "GroundTruth")
        self.assertEqual(module.test_img, "Segmentation")

    def make_workspace(self, ground_truth, test):
        '''Make a workspace with a ground-truth image and a test image
        
        ground_truth and test are dictionaries with the following keys:
        image     - the pixel data
        mask      - (optional) the mask data
        crop_mask - (optional) a cropping mask
        
        returns a workspace and module
        '''
        module = C.CalculateImageOverlap()
        module.module_num = 1
        module.obj_or_img = O_IMG
        module.ground_truth.value = GROUND_TRUTH_IMAGE_NAME
        module.test_img.value = TEST_IMAGE_NAME
        
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        
        for name, d in ((GROUND_TRUTH_IMAGE_NAME, ground_truth),
                        (TEST_IMAGE_NAME, test)):
            image = cpi.Image(d["image"],
                              mask = d.get("mask"),
                              crop_mask = d.get("crop_mask"))
            image_set.add(name, image)
            
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  cpo.ObjectSet(), cpmeas.Measurements(),
                                  image_set_list)
        return workspace, module
    
    def make_obj_workspace(self, ground_truth_obj, id_obj, ground_truth, id):
        '''make a workspace to test comparing objects'''
        ''' ground truth object and ID object  are dictionaires w/ the following keys'''
        '''i - i component of pixel coordinates
        j - j component of pixel coordinates
        l - label '''

        module = C.CalculateImageOverlap()
        module.module_num = 1
        module.obj_or_img.value = O_OBJ
        module.object_name_GT.value = GROUND_TRUTH_OBJ 
        module.img_obj_found_in_GT.value = GROUND_TRUTH_OBJ_IMAGE_NAME
        module.object_name_ID.value = ID_OBJ
        module.img_obj_found_in_ID.value = ID_OBJ_IMAGE_NAME
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        
        for name, d in ((GROUND_TRUTH_OBJ_IMAGE_NAME, ground_truth),
                        (ID_OBJ_IMAGE_NAME, id)):
            image = cpi.Image(d["image"],
                              mask = d.get("mask"),
                              crop_mask = d.get("crop_mask"))
            image_set.add(name, image)
        object_set = cpo.ObjectSet()
        for name, d in ((GROUND_TRUTH_OBJ, ground_truth_obj),
                        (ID_OBJ, id_obj)):
            object = cpo.Objects()
            object.segmented = d
            object.ijv = d
            object_set.add_objects(object, name)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, cpmeas.Measurements(),
                                  image_set_list)
        return workspace, module


    
    def test_03_01_zeros(self):
        '''Test ground-truth of zeros and image of zeros'''
        
        workspace, module = self.make_workspace(
            dict(image = np.ones((20,10), bool)),
            dict(image = np.ones((20,10), bool)))
        
        self.assertTrue(isinstance(module, C.CalculateImageOverlap))
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cpmeas.Measurements))
        self.assertEqual(
            measurements.get_current_image_measurement("Overlap_FalseNegRate_test"),
            0)
        features = measurements.get_feature_names(cpmeas.IMAGE)
        for feature in C.FTR_ALL:
            field = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            self.assertTrue(field in features, 
                            "Missing feature: %s" % feature)
            
    def test_03_02_ones(self):
        '''Test ground-truth of ones and image of ones'''
        
        workspace, module = self.make_workspace(
            dict(image = np.zeros((20,10), bool)),
            dict(image = np.zeros((20,10), bool)))
        
        self.assertTrue(isinstance(module, C.CalculateImageOverlap))
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cpmeas.Measurements))
        for feature, expected in ((C.FTR_FALSE_POS_RATE, 0),
                                  (C.FTR_FALSE_NEG_RATE, 0),
                                  (C.FTR_RECALL, 1),
                                  (C.FTR_PRECISION, 1),
                                  (C.FTR_F_FACTOR, 1)):
            mname = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertEqual(expected, value)
            
    def test_03_03_masked(self):
        '''Test ground-truth of a masked image'''
        
        workspace, module = self.make_workspace(
            dict(image = np.zeros((20,10), bool)),
            dict(image = np.zeros((20,10), bool),
                 mask = np.zeros((20,10), bool)))
        
        self.assertTrue(isinstance(module, C.CalculateImageOverlap))
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cpmeas.Measurements))
        for feature, expected in ((C.FTR_FALSE_POS_RATE, 0),
                                  (C.FTR_FALSE_NEG_RATE, 0),
                                  (C.FTR_RECALL, 1),
                                  (C.FTR_PRECISION, 1),
                                  (C.FTR_F_FACTOR, 1)):
            mname = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertEqual(expected, value)
            
    def test_03_04_all_right(self):
        np.random.seed(34)
        image = np.random.uniform(size=(10,20)) > .5
        workspace, module = self.make_workspace(
            dict(image=image), dict(image=image))
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cpmeas.Measurements))
        for feature, expected in ((C.FTR_FALSE_POS_RATE, 0),
                                  (C.FTR_FALSE_NEG_RATE, 0),
                                  (C.FTR_RECALL, 1),
                                  (C.FTR_PRECISION, 1),
                                  (C.FTR_F_FACTOR, 1)):
            mname = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertEqual(expected, value)
            
    def test_03_05_one_false_positive(self):
        i,j = np.mgrid[0:10,0:20]
        ground_truth = ((i+j) % 2) == 0
        test = ground_truth.copy()
        test[0,1] = True
        workspace, module = self.make_workspace(
            dict(image=ground_truth), dict(image=test))
        module.run(workspace)
        measurements = workspace.measurements
        precision = 100.0 / 101.0
        f_factor = 2 * precision / (1 + precision)
        for feature, expected in ((C.FTR_FALSE_POS_RATE, 0.01),
                                  (C.FTR_FALSE_NEG_RATE, 0),
                                  (C.FTR_RECALL, 1),
                                  (C.FTR_PRECISION, precision),
                                  (C.FTR_F_FACTOR, f_factor)):
            mname = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value, 
                                   msg = "%s is wrong" % feature)
        
    def test_03_05_one_false_negative(self):
        i,j = np.mgrid[0:10,0:20]
        ground_truth = ((i+j) % 2) == 0
        test = ground_truth.copy()
        test[0,0] = False
        workspace, module = self.make_workspace(
            dict(image=ground_truth), dict(image=test))
        module.run(workspace)
        measurements = workspace.measurements
        recall = 0.99
        f_factor = 2 * recall / (1 + recall)
        for feature, expected in ((C.FTR_FALSE_POS_RATE, 0),
                                  (C.FTR_FALSE_NEG_RATE, 0.01),
                                  (C.FTR_RECALL, recall),
                                  (C.FTR_PRECISION, 1),
                                  (C.FTR_F_FACTOR, f_factor)):
            mname = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value, 
                                   msg = "%s is wrong" % feature)
            
    def test_03_06_one_false_positive_and_mask(self):
        i,j = np.mgrid[0:10,0:20]
        ground_truth = ((i+j) % 2) == 0
        test = ground_truth.copy()
        test[0,1] = True
        mask = j < 10
        workspace, module = self.make_workspace(
            dict(image=ground_truth), dict(image=test, mask=mask))
        module.run(workspace)
        measurements = workspace.measurements
        precision = 50.0 / 51.0
        f_factor = 2 * precision / (1 + precision)
        for feature, expected in ((C.FTR_FALSE_POS_RATE, 0.02),
                                  (C.FTR_FALSE_NEG_RATE, 0),
                                  (C.FTR_RECALL, 1),
                                  (C.FTR_PRECISION, precision),
                                  (C.FTR_F_FACTOR, f_factor)):
            mname = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value, 
                                   msg = "%s is wrong" % feature)
        
    def test_03_07_one_false_negative_and_mask(self):
        i,j = np.mgrid[0:10,0:20]
        ground_truth = ((i+j) % 2) == 0
        test = ground_truth.copy()
        test[0,0] = False
        mask = j < 10
        workspace, module = self.make_workspace(
            dict(image=ground_truth), dict(image=test, mask=mask))
        module.run(workspace)
        measurements = workspace.measurements
        recall = 0.98
        f_factor = 2 * recall / (1 + recall)
        for feature, expected in ((C.FTR_FALSE_POS_RATE, 0),
                                  (C.FTR_FALSE_NEG_RATE, 0.02),
                                  (C.FTR_RECALL, recall),
                                  (C.FTR_PRECISION, 1),
                                  (C.FTR_F_FACTOR, f_factor)):
            mname = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value, 
                                   msg = "%s is wrong" % feature)

    def test_03_08_masked_errors(self):
        np.random.seed(38)
        ground_truth = np.random.uniform(size=(20,10)) > .5
        test = ground_truth.copy()
        mask = np.random.uniform(size=(20,10)) > .5
        test[~ mask] = np.random.uniform(size=np.sum(~ mask)) > .5
        workspace, module = self.make_workspace(
            dict(image=ground_truth), dict(image=test, mask=mask))
        module.run(workspace)
        measurements = workspace.measurements
        for feature, expected in ((C.FTR_FALSE_POS_RATE, 0),
                                  (C.FTR_FALSE_NEG_RATE, 0),
                                  (C.FTR_RECALL, 1),
                                  (C.FTR_PRECISION, 1),
                                  (C.FTR_F_FACTOR, 1)):
            mname = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value, 
                                   msg = "%s is wrong" % feature)
            
    def test_03_09_cropped(self):
        np.random.seed(39)
        i,j = np.mgrid[0:10,0:20]
        ground_truth = ((i+j) % 2) == 0
        test = ground_truth.copy()
        test[0,1] = True
        cropping = np.zeros((20,40),bool)
        cropping[10:20, 10:30] = True
        big_ground_truth = np.random.uniform(size=(20,40)) > .5
        big_ground_truth[10:20, 10:30] = ground_truth
        workspace, module = self.make_workspace(
            dict(image=big_ground_truth), 
            dict(image=test, crop_mask = cropping))
        module.run(workspace)
        measurements = workspace.measurements
        precision = 100.0 / 101.0
        f_factor = 2 * precision / (1 + precision)
        for feature, expected in ((C.FTR_FALSE_POS_RATE, 0.01),
                                  (C.FTR_FALSE_NEG_RATE, 0),
                                  (C.FTR_RECALL, 1),
                                  (C.FTR_PRECISION, precision),
                                  (C.FTR_F_FACTOR, f_factor)):
            mname = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            value = measurements.get_current_image_measurement(mname)
            self.assertAlmostEqual(expected, value, 
                                   msg = "%s is wrong. Expected %f, got %f" % 
                                   (feature, expected, value))
        
        
    def test_04_01_get_measurement_columns(self):
        workspace, module = self.make_workspace(
            dict(image = np.zeros((20,10), bool)),
            dict(image = np.zeros((20,10), bool)))
        
        columns = module.get_measurement_columns(workspace.pipeline)
        # All columns should be unique
        self.assertEqual(len(columns), len(set([x[1] for x in columns])))
        # All columns should be floats and done on images
        self.assertTrue(all([x[0] == cpmeas.IMAGE]))
        self.assertTrue(all([x[2] == cpmeas.COLTYPE_FLOAT]))
        for feature in C.FTR_ALL:
            field = '_'.join((C.C_IMAGE_OVERLAP, feature, TEST_IMAGE_NAME))
            self.assertTrue(field in [x[1] for x in columns])

    def test_04_02_get_categories(self):
        workspace, module = self.make_workspace(
            dict(image = np.zeros((20,10), bool)),
            dict(image = np.zeros((20,10), bool)))
        categories = module.get_categories(workspace.pipeline, "Foo")
        self.assertEqual(len(categories), 0)
        categories = module.get_categories(workspace.pipeline, cpmeas.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], C.C_IMAGE_OVERLAP)

    def test_04_03_get_measurements(self):
        workspace, module = self.make_workspace(
            dict(image = np.zeros((20,10), bool)),
            dict(image = np.zeros((20,10), bool)))
        mnames = module.get_measurements(workspace.pipeline, 
                                         cpmeas.IMAGE, C.C_IMAGE_OVERLAP)
        self.assertEqual(len(mnames), len(C.FTR_ALL))
        self.assertTrue(all(n in C.FTR_ALL for n in mnames))
        self.assertTrue(all(f in mnames for f in C.FTR_ALL))
        mnames = module.get_measurements(workspace.pipeline, "Foo",
                                         C.C_IMAGE_OVERLAP)
        self.assertEqual(len(mnames), 0)
        mnames = module.get_measurements(workspace.pipeline, cpmeas.IMAGE,
                                         "Foo")
        self.assertEqual(len(mnames), 0)
        
    def test_04_04_get_measurement_images(self):
        workspace, module = self.make_workspace(
            dict(image = np.zeros((20,10), bool)),
            dict(image = np.zeros((20,10), bool)))
        
        for feature in C.FTR_ALL:
            imnames = module.get_measurement_images(workspace.pipeline,
                                                    cpmeas.IMAGE,
                                                    C.C_IMAGE_OVERLAP,
                                                    feature)
            self.assertEqual(len(imnames), 1)
            self.assertEqual(imnames[0], TEST_IMAGE_NAME)
        imnames = module.get_measurement_images(workspace.pipeline,
                                                cpmeas.IMAGE,
                                                C.C_IMAGE_OVERLAP,
                                                "Foo")
        self.assertEqual(len(imnames), 0)
        imnames = module.get_measurement_images(workspace.pipeline,
                                                cpmeas.IMAGE,
                                                "Foo",
                                                C.FTR_FALSE_NEG_RATE)
        self.assertEqual(len(imnames), 0)
        imnames = module.get_measurement_images(workspace.pipeline,
                                                "Foo",
                                                C.C_IMAGE_OVERLAP,
                                                C.FTR_FALSE_NEG_RATE)
        self.assertEqual(len(imnames), 0)
        
    def test_05_01_test_measure_overlap_objects(self):
        workspace, module = self.make_obj_workspace(
            np.ones((200,3), int),
            np.ones((200,3), int),
            dict(image = np.zeros((20,10), bool)),
            dict(image = np.zeros((20,10), bool)))
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cpmeas.Measurements))

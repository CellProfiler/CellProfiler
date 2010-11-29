'''test_straightenworms - test the StraightenWorms module'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision: 10732 $"

import numpy as np
from StringIO import StringIO
import unittest

import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw
import cellprofiler.modules.straightenworms as S
import cellprofiler.modules.identify as I

OBJECTS_NAME = "worms"
STRAIGHTENED_OBJECTS_NAME = "straightenedworms"
IMAGE_NAME = "wormimage"
STRAIGHTENED_IMAGE_NAME = "straightenedimage"

class TestStraightenWorms(unittest.TestCase):
    def test_01_01_load_v1(self):
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10732

StraightenWorms:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Untangled worms name?:OverlappingWorms
    Straightened worms name?:StraightenedWorms
    Worm width\x3A:20
    Training set file location:Default Output Folder\x7CNone
    Training set file name:TrainingSet.xml
    Image count:2
    Image name:Brightfield
    Straightened image name:StraightenedBrightfield
    Image name:Fluorescence
    Straightened image name:StraightenedFluorescence
'''
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, S.StraightenWorms))
        self.assertEqual(module.objects_name, "OverlappingWorms")
        self.assertEqual(module.straightened_objects_name, "StraightenedWorms")
        self.assertEqual(module.width, 20)
        self.assertEqual(module.training_set_directory.dir_choice,
                         cps.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.training_set_file_name, "TrainingSet.xml")
        self.assertEqual(module.image_count.value, 2)
        for group, input_name, output_name in (
            (module.images[0], "Brightfield", "StraightenedBrightfield"),
            (module.images[1], "Fluorescence", "StraightenedFluorescence")):
            self.assertEqual(group.image_name, input_name)
            self.assertEqual(group.straightened_image_name, output_name)
            
    def make_workspace(self, control_points, lengths, radii, image, mask = None):
        '''Create a workspace containing the control point measurements
        
        control_points - an n x 2 x m array where n is the # of control points,
                         and m is the number of objects.
        lengths - the length of each object
        radii - the radii_from_training defining the radius at each control pt
        '''
        module = S.StraightenWorms()
        module.objects_name.value = OBJECTS_NAME
        module.straightened_objects_name.value = STRAIGHTENED_OBJECTS_NAME
        module.images[0].image_name.value = IMAGE_NAME
        module.images[0].straightened_image_name.value = STRAIGHTENED_IMAGE_NAME
        module.module_num = 1
        
        # Trick the module into thinking it's read the data file
        
        class P:
            def __init__(self):
                self.radii_from_training = radii
        
        module.training_set_directory.dir_choice == cps.URL_FOLDER_NAME
        module.training_set_directory.custom_path = "http://www.cellprofiler.org"
        module.training_set_file_name.value = "TrainingSet.xml"
        module.training_params = {
            "TrainingSet.xml": ( P(), "URL")
        }
        
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        
        m = cpmeas.Measurements()
        for i, (y,x) in enumerate(control_points):
            for v, f in ((x, S.F_CONTROL_POINT_X),
                         (y, S.F_CONTROL_POINT_Y)):
                feature = "_".join((S.C_WORM, f, str(i+1)))
                m.add_measurement(OBJECTS_NAME, feature, v)
        feature = "_".join((S.C_WORM, S.F_LENGTH))
        m.add_measurement(OBJECTS_NAME, feature, lengths)
        
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(IMAGE_NAME, cpi.Image(image, mask))
        
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), OBJECTS_NAME)
        
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, m, image_set_list)
        return workspace, module
    
    def test_02_01_straighten_nothing(self):
        workspace, module = self.make_workspace(np.zeros((5,2,0)), 
                                                np.zeros(0),
                                                np.zeros(5),
                                                np.zeros((20,10)))
        module.run(workspace)
        
        image = workspace.image_set.get_image(STRAIGHTENED_IMAGE_NAME)
        self.assertFalse(np.any(image.mask))
        objectset = workspace.object_set
        self.assertTrue(isinstance(objectset, cpo.ObjectSet))
        labels = objectset.get_objects(STRAIGHTENED_OBJECTS_NAME).segmented
        self.assertTrue(np.all(labels == 0))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        self.assertEqual(m.get_current_image_measurement(
            "_".join((I.C_COUNT,STRAIGHTENED_OBJECTS_NAME))), 0)
        self.assertEqual(len(m.get_current_measurement(
            STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_X)), 0)
        self.assertEqual(len(m.get_current_measurement(
            STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_Y)), 0)
        
    def test_02_02_straighten_straight_worm(self):
        '''Do a "straightening" that is a 1-1 mapping'''
        r = np.random.RandomState()
        r.seed(0)
        image = r.uniform(size=(60, 30))
        control_points = np.array([[[21],[15]],
                                   [[23],[15]],
                                   [[25],[15]],
                                   [[27],[15]],
                                   [[29],[15]]])
        lengths = np.array([8])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 11)
        self.assertEqual(pixels.shape[0], 19)
        np.testing.assert_almost_equal(pixels, image[16:35,10:21])
        
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        self.assertEqual(m.get_current_image_measurement(
            "_".join((I.C_COUNT,STRAIGHTENED_OBJECTS_NAME))), 1)
        v = m.get_current_measurement(
            STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_X)
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(v[0], 5)
        v = m.get_current_measurement(
            STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_Y)
        self.assertEqual(len(v), 1)
        self.assertAlmostEqual(v[0], 9)
        expected = np.zeros(pixels.shape, int)
        for i in range(9):
            expected[i+5, abs(i - 4):11-abs(i-4)] = 1
        object_set = workspace.object_set
        objects = object_set.get_objects(STRAIGHTENED_OBJECTS_NAME)
        self.assertTrue(np.all(objects.segmented == expected))
        
    def test_02_03_straighten_diagonal_worm(self):
        '''Do a straightening on a worm on the 3x4x5 diagonal'''
        r = np.random.RandomState()
        r.seed(23)
        image = r.uniform(size=(60, 30))
        control_points = np.array([[[10],[10]],
                                   [[13],[14]],
                                   [[16],[18]],
                                   [[19],[22]],
                                   [[22],[26]]])
        lengths = np.array([20])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 11)
        self.assertEqual(pixels.shape[0], 31)
        expected = image[control_points[:,0,0],control_points[:,1,0]]
        samples = pixels[5:26:5,5]
        np.testing.assert_almost_equal(expected, samples)
        
    def test_02_04_straighten_two_worms(self):
        '''Straighten the worms from tests 02_02 and 02_03 together'''
        r = np.random.RandomState()
        r.seed(0)
        image = r.uniform(size=(60, 30))
        control_points = np.array([[[21,10],[15,10]],
                                   [[23,13],[15,14]],
                                   [[25,16],[15,18]],
                                   [[27,19],[15,22]],
                                   [[29,22],[15,26]]])
        lengths = np.array([8, 20])
        radii = np.array([1, 3, 5, 3, 1])
        workspace, module = self.make_workspace(control_points, lengths,
                                                radii, image)
        self.assertTrue(isinstance(module, S.StraightenWorms))
        module.width.value = 11
        module.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
        self.assertEqual(pixels.shape[1], 22)
        self.assertEqual(pixels.shape[0], 31)
        np.testing.assert_almost_equal(pixels[0:19,0:11], image[16:35,10:21])
        expected = image[control_points[:,0,1],control_points[:,1,1]]
        samples = pixels[5:26:5,16]
        np.testing.assert_almost_equal(expected, samples)
        
        
    def test_03_01_get_measurement_columns(self):
        workspace, module = self.make_workspace(np.zeros((5,2,0)), 
                                                np.zeros(0),
                                                np.zeros(5),
                                                np.zeros((20,10)))
        self.assertTrue(isinstance(module, S.StraightenWorms))
        columns = module.get_measurement_columns(workspace.pipeline)
        for object_name, feature_name in (
            ( cpmeas.IMAGE, "_".join((I.C_COUNT, STRAIGHTENED_OBJECTS_NAME))),
            ( STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_X),
            ( STRAIGHTENED_OBJECTS_NAME, I.M_LOCATION_CENTER_Y),
            ( STRAIGHTENED_OBJECTS_NAME, I.M_NUMBER_OBJECT_NUMBER)):
            self.assertTrue(any([(o == object_name) and (f == feature_name)
                                 for o, f, t in columns]))
            
        categories = module.get_categories(workspace.pipeline, cpmeas.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], I.C_COUNT)
        
        categories = module.get_categories(workspace.pipeline, STRAIGHTENED_OBJECTS_NAME)
        self.assertEqual(len(categories), 2)
        self.assertTrue(I.C_LOCATION in categories)
        self.assertTrue(I.C_NUMBER in categories)
        
        f = module.get_measurements(workspace.pipeline, cpmeas.IMAGE, I.C_COUNT)
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0], STRAIGHTENED_OBJECTS_NAME)
        
        f = module.get_measurements(workspace.pipeline, STRAIGHTENED_OBJECTS_NAME, I.C_NUMBER)
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0], I.FTR_OBJECT_NUMBER)
        
        f = module.get_measurements(workspace.pipeline, STRAIGHTENED_OBJECTS_NAME, I.C_LOCATION)
        self.assertEqual(len(f), 2)
        self.assertTrue(I.FTR_CENTER_X in f)
        self.assertTrue(I.FTR_CENTER_Y in f)
import unittest

import numpy as np
from centrosome.cpmorphology import skeletonize_labels
from scipy.ndimage import label, distance_transform_edt

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.modules.identify as I
import cellprofiler.objects as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
from cellprofiler.modules import instantiate_module

MODULE_NAME = "Example4b"
INPUT_OBJECTS_NAME = "inputobjects"
OUTPUT_OBJECTS_NAME = "outputobjects"

class TestExample4b(unittest.TestCase):
    def test_01_01_instantiate(self):
        try:
            instantiate_module(MODULE_NAME)
        except:
            self.fail("CellProfiler could not create your module. "
                      "Is it named, " + MODULE_NAME + "?")
    
    def test_01_02_run(self):
        module = instantiate_module(MODULE_NAME)
        module.input_objects_name.value = INPUT_OBJECTS_NAME
        module.output_objects_name.value = OUTPUT_OBJECTS_NAME
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        
        object_set = cpo.ObjectSet()
        #
        # Pick a bunch of random points, dilate them using the distance
        # transform and then label the result.
        #
        r = np.random.RandomState()
        r.seed(12)
        bimg = np.ones((100, 100), bool)
        bimg[r.randint(0,100, 50), r.randint(0, 100, 50)] = False
        labels, count = label(distance_transform_edt(bimg) <= 5)
        #
        # Make the input objects
        #
        input_objects = cpo.Objects()
        input_objects.segmented = labels
        expected = skeletonize_labels(labels)
        object_set.add_objects(input_objects, INPUT_OBJECTS_NAME)
        #
        # Make the workspace
        #
        measurements = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, None, object_set,
                                  measurements, None)
        module.run(workspace)
        #
        # Calculate the centers using Numpy. Scipy can do this too.
        # But maybe it's instructive to show you how to go at the labels
        # matrix using Numpy.
        #
        # We're going to get the centroids by taking the average value
        # of x and y per object.
        #
        y, x = np.mgrid[0:labels.shape[0], 0:labels.shape[1]].astype(float)
        #
        # np.bincount counts the number of occurrences of each integer value.
        # You need to operate on a 1d array - if you flatten the labels
        # and weights, their pixels still align.
        #
        # We do [1:] to discard the background which is labeled 0
        #
        # The optional second argument to np.bincount is the "weight". For
        # each label value, maintain a running sum of the weights.
        #
        areas = np.bincount(expected.flatten())[1:]
        total_x = np.bincount(expected.flatten(), weights=x.flatten())[1:]
        total_y = np.bincount(expected.flatten(), weights=y.flatten())[1:]
        expected_location_x = total_x / areas
        expected_location_y = total_y / areas
        #
        # Now check against the measurements.
        #
        count_feature = I.C_COUNT + "_" + OUTPUT_OBJECTS_NAME
        self.assertTrue(measurements.has_feature(cpmeas.IMAGE, count_feature),
                        "Your module did not produce a %s measurement" %
                        count_feature)
        count = measurements.get_measurement(cpmeas.IMAGE, count_feature)
        self.assertEqual(count, len(areas))
        for ftr, expected in ((I.M_LOCATION_CENTER_X, expected_location_x),
                              (I.M_LOCATION_CENTER_Y, expected_location_y)):
            self.assertTrue(measurements.has_feature(
                OUTPUT_OBJECTS_NAME, ftr))
            location = measurements.get_measurement(OUTPUT_OBJECTS_NAME, ftr)
            np.testing.assert_almost_equal(location, expected)
            
    def test_02_01_maybe_you_implemented_get_categories(self):
        module = instantiate_module(MODULE_NAME)
        module.output_objects_name.value = OUTPUT_OBJECTS_NAME
        if module.get_categories.im_func is not cpm.CPModule.get_categories.im_func:
            c_image = module.get_categories(None, cpmeas.IMAGE)
            self.assertTrue(I.C_COUNT in c_image)
            c_objects = module.get_categories(None, OUTPUT_OBJECTS_NAME)
            self.assertTrue(I.C_LOCATION in c_objects)
            print "+3 for you!"
    
    def test_02_02_maybe_you_implemented_get_measurements(self):
        module = instantiate_module(MODULE_NAME)
        module.output_objects_name.value = OUTPUT_OBJECTS_NAME
        if module.get_measurements.im_func is not cpm.CPModule.get_measurements.im_func:
            ftr_image = module.get_measurements(None, cpmeas.IMAGE, I.C_COUNT)
            self.assertTrue(OUTPUT_OBJECTS_NAME in ftr_image)
            ftr_objects = module.get_measurements(None, OUTPUT_OBJECTS_NAME,
                                                  I.C_LOCATION)
            self.assertTrue(I.FTR_CENTER_X in ftr_objects)
            self.assertTrue(I.FTR_CENTER_Y in ftr_objects)
            print "+3 for you!"
        
        
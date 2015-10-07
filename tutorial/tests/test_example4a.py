from scipy.ndimage import label, distance_transform_edt
import numpy as np
import unittest

import cellprofiler.pipeline as cpp
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw

from cellprofiler.modules import instantiate_module
from centrosome.cpmorphology import skeletonize_labels

MODULE_NAME = "Example4a"
INPUT_OBJECTS_NAME = "inputobjects"
OUTPUT_OBJECTS_NAME = "outputobjects"

class TestExample4a(unittest.TestCase):
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
        workspace = cpw.Workspace(pipeline, module, None, object_set,
                                  cpmeas.Measurements(), None)
        module.run(workspace)
        
        self.assertTrue(OUTPUT_OBJECTS_NAME in object_set.object_names,
                        "Could not find the output objects in the object set")
        output_objects = object_set.get_objects(OUTPUT_OBJECTS_NAME)
        np.testing.assert_array_equal(expected, output_objects.segmented)
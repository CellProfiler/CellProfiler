import sys
import unittest

import numpy as np
import scipy.ndimage

import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
from cellprofiler.modules import instantiate_module

MODULE_NAME = "Example5a"

OBJECTS_NAME = "objects"

class TestExample5a(unittest.TestCase):
    def test_00_00_instantiate(self):
        instantiate_module(MODULE_NAME)

    def make_fg_bg(self, seed):
        #
        # Pick a bunch of random points, dilate them using the distance
        # transform and return the result.
        #
        r = np.random.RandomState()
        r.seed(seed)
        bimg = np.ones((100, 100), bool)
        # pick random points, but not near the edges
        bimg[r.randint(6, 94, 50), r.randint(6, 94, 50)] = False
        return scipy.ndimage.distance_transform_edt(bimg) <= 5
    
    def run_tteesstt(self, objects, level):
        module = instantiate_module(MODULE_NAME)
        module.objects_name.value = OBJECTS_NAME
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        
        object_set = cpo.ObjectSet()
        object_set.add_objects(objects, OBJECTS_NAME)
        #
        # Make the workspace
        #
        measurements = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, None, object_set,
                                  measurements, None)
        module.run(workspace)
        values = measurements.get_measurement(OBJECTS_NAME, "Example5_MeanDistance")
        self.assertEqual(len(values), objects.count)
        for labels, indices in objects.get_labels():
            for l in np.unique(labels):
                #
                # Test very slowly = 1 object per pass
                #
                if l == 0:
                    continue
                d = scipy.ndimage.distance_transform_edt(labels == l)
                value = np.sum(d) / np.sum(labels == l)
                if abs(value - values[l-1]) > .0001:
                    if level == 1:
                        self.fail("You got the wrong answer (label = %d, mine = %f, yours = %f" %
                                  l, value, values[l-1])
                    else:
                        sys.stderr.write("Oh too bad, did not pass level %d\n" % level)
                        return
        if level > 1:
            print "+%d for you!" % level
            
    def test_01_01_run_empty(self):
        #
        # Test on an image devoid of objects
        #
        objects = cpo.Objects()
        objects.segmented = np.zeros((35, 43), int)
        self.run_tteesstt(objects, 1)
        
    def test_01_02_run_simple(self):
        #
        # Make a foreground / background mask and label it. There will
        # always be at least one pixel separation because otherwise the
        # objects would be connected and would be joined. We use an 8-connected
        # structuring element here - that's CellProfiler's standard.
        #
        fg_bg = self.make_fg_bg(12)
        labels, count = scipy.ndimage.label(fg_bg, structure = np.ones((3,3), bool))
        #
        # Make the objects
        #
        objects = cpo.Objects()
        objects.segmented = labels
        self.run_tteesstt(objects, 1)
        
    def test_01_03_run_overlapped(self):
        planes = [self.make_fg_bg(13*(i+1)) for i in range(4)]
        i, j = np.mgrid[0:planes[0].shape[0], 0:planes[0].shape[1]]
        ijv = np.vstack([
            np.column_stack((i[plane!=0], j[plane!=0], plane[plane!=0]))
            for plane in planes])
        objects = cpo.Objects()
        objects.set_ijv(ijv, planes[0].shape)
        self.run_tteesstt(objects, 3)
        
    def test_01_04_run_touching(self):
        fg_bg = self.make_fg_bg(14)
        labels, count = scipy.ndimage.label(fg_bg, structure = np.ones((3,3), bool))
        #
        # This returns the i and j coordinates of the closest foreground pixel
        # for each background pixel. So we label each background pixel and
        # voila! everything is touching.
        #
        i, j = scipy.ndimage.distance_transform_edt(labels == 0,
                                                    return_distances=False,
                                                    return_indices = True)
        labels[labels != 0] = labels[i[labels!=0], j[labels!=0]]
        objects = cpo.Objects()
        objects.segmented= labels
        self.run_tteesstt(objects, 3)
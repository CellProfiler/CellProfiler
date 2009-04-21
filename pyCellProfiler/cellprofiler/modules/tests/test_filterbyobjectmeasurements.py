'''test_filterbyobjectmeasurements.py: Test FilterByObjectMeasurements module
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import numpy as np
import unittest

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.objects as cpo
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpm
import cellprofiler.modules.filterbyobjectmeasurement as F

class TestFilterByObjectMeasurement(unittest.TestCase):
    def make_workspace(self, object_dict= {}, image_dict = {}):
        '''Make a workspace for testing FilterByObjectMeasurement'''
        module = F.FilterByObjectMeasurement()
        pipeline = cpp.Pipeline()
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  cpm.Measurements(),
                                  image_set_list)
        for key in image_dict.keys():
            image_set.add(key, cpi.Image(image_dict[key]))
        for key in object_dict.keys():
            o = cpo.Objects()
            o.segmented = object_dict[key]
            object_set.add_objects(o, key)
        return workspace, module
        
    def test_00_01_zeros_single(self):
        '''Test keep single object on an empty labels matrix'''
        workspace, module = self.make_workspace({ "my_objects": np.zeros((10,10),int) })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_MAXIMAL
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",np.zeros((0,)))
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==0))
    
    def test_00_02_zeros_per_object(self):
        '''Test keep per object filtering on an empty labels matrix'''
        workspace, module = self.make_workspace(  
            {"my_objects": np.zeros((10,10),int),
             "my_enclosing_objects": np.zeros((10,10),int)})
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.enclosing_object_name.value = "my_enclosing_objects"
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_MAXIMAL_PER_OBJECT
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",np.zeros((0,)))
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==0))
    
    def test_00_03_zeros_filter(self):
        '''Test object filtering on an empty labels matrix'''
        workspace, module = self.make_workspace({ "my_objects": np.zeros((10,10),int) })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_LIMITS
        module.limits.min = 0
        module.limits.max = 1000
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",np.zeros((0,)))
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==0))
    
    def test_01_01_keep_single_min(self):
        '''Keep a single object (min) from among two'''
        labels = np.zeros((10,10), int)
        labels[2:4,3:5] = 1
        labels[6:9,5:8] = 2
        expected = labels.copy()
        expected[labels == 1] = 0
        expected[labels == 2] = 1
        workspace, module = self.make_workspace({ "my_objects": labels })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_MINIMAL
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",np.array([2,1]))
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==expected))

    def test_01_02_keep_single_max(self):
        '''Keep a single object (max) from among two'''
        labels = np.zeros((10,10), int)
        labels[2:4,3:5] = 1
        labels[6:9,5:8] = 2
        expected = labels.copy()
        expected[labels == 1] = 0
        expected[labels == 2] = 1
        workspace, module = self.make_workspace({ "my_objects": labels })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_MAXIMAL
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",np.array([1,2]))
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==expected))
    
    def test_02_01_keep_one_min(self):
        '''Keep two sub-objects (min) from among four enclosed by two'''
        sub_labels = np.zeros((20,20), int)
        expected = np.zeros((20,20), int)
        for i,j,k,e in ((0,0,1,0),(10,0,2,1),(0,10,3,2),(10,10,4,0)):
            sub_labels[i+2:i+5,j+3:j+7] = k
            expected[i+2:i+5,j+3:j+7] = e
        labels = np.zeros((20,20), int)
        labels[:,:10] = 1
        labels[:,10:] = 2
        workspace, module = self.make_workspace({ "my_objects": sub_labels,
                                                 "my_enclosing_objects": labels })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.enclosing_object_name.value = 'my_enclosing_objects'
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_MINIMAL_PER_OBJECT
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",np.array([2,1,3,4]))
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==expected))
    
    def test_02_02_keep_one_max(self):
        '''Keep two sub-objects (max) from among four enclosed by two'''
        sub_labels = np.zeros((20,20), int)
        expected = np.zeros((20,20), int)
        for i,j,k,e in ((0,0,1,0),(10,0,2,1),(0,10,3,2),(10,10,4,0)):
            sub_labels[i+2:i+5,j+3:j+7] = k
            expected[i+2:i+5,j+3:j+7] = e
        labels = np.zeros((20,20), int)
        labels[:,:10] = 1
        labels[:,10:] = 2
        workspace, module = self.make_workspace({ "my_objects": sub_labels,
                                                 "my_enclosing_objects": labels })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.enclosing_object_name.value = 'my_enclosing_objects'
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_MAXIMAL_PER_OBJECT
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",np.array([1,2,4,3]))
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==expected))
    
    def test_03_01_filter(self):
        '''Filter objects by limits'''
        n = 40
        labels = np.zeros((10,n*10),int)
        for i in range(40):
            labels[2:5,i*10+3:i*10+7] = i+1
        np.random.seed(0)
        values = np.random.uniform(size=n)
        idx = 1
        my_min = .3
        my_max = .7
        expected = np.zeros(labels.shape, int)
        for i, value in zip(range(n), values):
            if value >= my_min and value <= my_max:
                expected[labels == i+1] = idx
                idx += 1 
        workspace, module = self.make_workspace({ "my_objects": labels })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_LIMITS
        module.limits.min = my_min
        module.limits.max = my_max
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",values)
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==expected))
    
    def test_04_01_renumber_other(self):
        '''Renumber an associated object'''
        n = 40
        labels = np.zeros((10,n*10),int)
        alternates = np.zeros((10,n*10), int)
        for i in range(40):
            labels[2:5,i*10+3:i*10+7] = i+1
            alternates[3:7,i*10+2:i*10+5] = i+1
        np.random.seed(0)
        values = np.random.uniform(size=n)
        idx = 1
        my_min = .3
        my_max = .7
        expected = np.zeros(labels.shape, int)
        expected_alternates = np.zeros(alternates.shape, int)
        for i, value in zip(range(n), values):
            if value >= my_min and value <= my_max:
                expected[labels == i+1] = idx
                expected_alternates[alternates == i+1] = idx
                idx += 1 
        workspace, module = self.make_workspace({ "my_objects": labels,
                                                 "my_alternates": alternates })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_LIMITS
        module.limits.min = my_min
        module.limits.max = my_max
        module.add_additional_object()
        module.additional_objects[0].object_name.value="my_alternates"
        module.additional_objects[0].target_name.value = "my_additional_result"
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",values)
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        alternates = workspace.object_set.get_objects("my_additional_result")
        self.assertTrue(np.all(labels.segmented==expected))
        self.assertTrue(np.all(alternates.segmented==expected_alternates))
        
'''test_filterbyobjectmeasurements.py: Test FilterByObjectMeasurements module
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import base64
import numpy as np
import os
import StringIO
import tempfile
import zlib
import unittest

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.objects as cpo
import cellprofiler.cpimage as cpi
import cellprofiler.preferences as cpprefs
import cellprofiler.measurements as cpm
import cellprofiler.modules.filterobjects as F

class TestFilterObjects(unittest.TestCase):
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
        module.measurements[0].measurement.value = "my_measurement"
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
        module.measurements[0].measurement.value = "my_measurement"
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
        module.measurements[0].measurement.value = "my_measurement"
        module.filter_choice = F.FI_LIMITS
        module.measurements[0].min_limit.value = 0
        module.measurements[0].max_limit.value = 1000
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
        module.measurements[0].measurement.value = "my_measurement"
        module.filter_choice = F.FI_MINIMAL
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",np.array([2,1]))
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==expected))
        parents = m.get_current_measurement("my_result","Parent_my_objects")
        self.assertEqual(len(parents),1)
        self.assertEqual(parents[0],2)
        self.assertEqual(m.get_current_image_measurement("Count_my_result"),1)
        feature = F.FF_CHILDREN_COUNT % "my_result"
        child_count = m.get_current_measurement("my_objects", feature)
        self.assertEqual(len(child_count), 2)
        self.assertEqual(child_count[0], 0)
        self.assertEqual(child_count[1], 1)

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
        module.measurements[0].measurement.value = "my_measurement"
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
        module.measurements[0].measurement.value = "my_measurement"
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
        module.measurements[0].measurement.value = "my_measurement"
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
        module.measurements[0].measurement.value = "my_measurement"
        module.filter_choice = F.FI_LIMITS
        module.measurements[0].wants_minimum.value = True
        module.measurements[0].min_limit.value = my_min
        module.measurements[0].wants_maximum.value = True
        module.measurements[0].max_limit.value = my_max
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",values)
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==expected))
    
    def test_03_02_filter(self):
        '''Filter objects by min limits'''
        n = 40
        labels = np.zeros((10,n*10),int)
        for i in range(40):
            labels[2:5,i*10+3:i*10+7] = i+1
        np.random.seed(0)
        values = np.random.uniform(size=n)
        idx = 1
        my_min = .3
        expected = np.zeros(labels.shape, int)
        for i, value in zip(range(n), values):
            if value >= my_min:
                expected[labels == i+1] = idx
                idx += 1 
        workspace, module = self.make_workspace({ "my_objects": labels })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.measurements[0].measurement.value = "my_measurement"
        module.filter_choice = F.FI_LIMITS
        module.measurements[0].min_limit.value = my_min
        module.measurements[0].max_limit.value = .7
        module.measurements[0].wants_maximum.value = False
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",values)
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==expected))

    def test_03_03_filter(self):
        '''Filter objects by maximum limits'''
        n = 40
        labels = np.zeros((10,n*10),int)
        for i in range(40):
            labels[2:5,i*10+3:i*10+7] = i+1
        np.random.seed(0)
        values = np.random.uniform(size=n)
        idx = 1
        my_max = .7
        expected = np.zeros(labels.shape, int)
        for i, value in zip(range(n), values):
            if value <= my_max:
                expected[labels == i+1] = idx
                idx += 1 
        workspace, module = self.make_workspace({ "my_objects": labels })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.measurements[0].measurement.value = "my_measurement"
        module.filter_choice = F.FI_LIMITS
        module.measurements[0].min_limit.value = .3
        module.measurements[0].wants_minimum.value = False
        module.measurements[0].max_limit.value = my_max
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",values)
        module.run(workspace)
        labels = workspace.object_set.get_objects("my_result")
        self.assertTrue(np.all(labels.segmented==expected))
        
    def test_03_04_filter_two(self):
        '''Filter objects by two measurements'''
        n = 40
        labels = np.zeros((10,n*10),int)
        for i in range(40):
            labels[2:5,i*10+3:i*10+7] = i+1
        np.random.seed(0)
        values = np.zeros((n,2))
        values = np.random.uniform(size=(n,2))
        idx = 1
        my_max = np.array([.7,.5])
        expected = np.zeros(labels.shape, int)
        for i, v1,v2 in zip(range(n), values[:,0],values[:,1]):
            if v1 <= my_max[0] and v2 <= my_max[1]:
                expected[labels == i+1] = idx
                idx += 1 
        workspace, module = self.make_workspace({ "my_objects": labels })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.add_measurement()
        m = workspace.measurements
        for i in range(2):
            measurement_name = "measurement%d" % (i+1)
            module.measurements[i].measurement.value = measurement_name
            module.filter_choice = F.FI_LIMITS
            module.measurements[i].min_limit.value = .3
            module.measurements[i].wants_minimum.value = False
            module.measurements[i].max_limit.value = my_max[i]
            m.add_measurement("my_objects",measurement_name,values[:,i])
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
        module.measurements[0].measurement.value = "my_measurement"
        module.filter_choice = F.FI_LIMITS
        module.measurements[0].min_limit.value = my_min
        module.measurements[0].max_limit.value = my_max
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
    
    def test_05_00_load_matlab_v5(self):
        data = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:1234
FromMatlab:True

FilterByObjectMeasurement:[module_num:1|svn_version:\'8913\'|variable_revision_number:5|show_window:False|notes:\x5B\x5D]
    Which object would you like to filter by, or if using a Ratio, what is the numerator object?:MyObjects
    What do you want to call the filtered objects?:MyFilteredObjects
    Which category of measurements would you want to filter by?:Texture
    Which feature do you want to use? (Enter the feature number or name - see help for details):Granulectomy
    For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements do you want to use (for other measurements, this will only affect the display)?:MyImage
    For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?:15
    Minimum value required:No minimum
    Maximum value allowed:0.85
    What do you want to call the outlines of the identified objects (optional)?:MyOutlines
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, F.FilterByObjectMeasurement))
        self.assertEqual(module.object_name, "MyObjects")
        self.assertEqual(module.target_name, "MyFilteredObjects")
        self.assertEqual(module.measurements[0].measurement, "Texture_Granulectomy")
        self.assertEqual(module.filter_choice, F.FI_LIMITS)
        self.assertAlmostEqual(module.measurements[0].max_limit.value, 0.85)
        self.assertEqual(module.outlines_name, "MyOutlines")

    def test_05_01_load_matlab(self):
        '''Test loading a Matlab pipeline
        
Saved Pipeline, in file fbom_pipe.txt, Saved on 22-Apr-2009

SVN version number: 7297
Pixel Size: 1

Pipeline:
    KeepLargestObject
    FilterByObjectMeasurement
    FilterByObjectMeasurement

Module #1: KeepLargestObject revision - 1
     What did you call the primary objects?    FilteredNuclei
     What did you call the secondary objects?    TargetObjects
     What do you want to call the largest primary objects?    TargetObjects

Module #2: FilterByObjectMeasurement revision - 6
     What do you want to call the filtered objects?    FilteredNuclei
     Which object would you like to filter by, or if using a Ratio, what is the numerator object?    LargestObjects
     Which category of measurements would you want to filter by?    AreaShape
     Which feature do you want to use? (Enter the feature number or name - see help for details)    Perimeter
     For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements do you want to use (for other measurements, this will only affect the display)?    
     For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?    1
     Minimum value required:    200
     Maximum value allowed:    No maximum
     What do you want to call the outlines of the identified objects? Type "Do not use" to ignore.    Do not use

Module #3: FilterByObjectMeasurement revision - 6
     What do you want to call the filtered objects?    FilteredNuclei
     Which object would you like to filter by, or if using a Ratio, what is the numerator object?    TargetObjects
     Which category of measurements would you want to filter by?    Intensity
     Which feature do you want to use? (Enter the feature number or name - see help for details)    MeanIntensity
     For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements do you want to use (for other measurements, this will only affect the display)?    
     For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?    1
     Minimum value required:    No minimum
     Maximum value allowed:    .25
     What do you want to call the outlines of the identified objects? Type "Do not use" to ignore.    OutlineObjects
        '''
        data = ('TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBD' +
                'cmVhdGVkIG9uOiBXZWQgQXByIDIyIDEyOjM2OjQ3IDIwMDkgICAg' +
                'ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg' +
                'ICAgICAgIAABSU0PAAAACQIAAHic1ZZPT9swFMAd0lYURAU7gbRD' +
                'j5xQQZq045gmBGK0FUW9u/RRPCV2FNuo5VPtuOM+Csd9jNkkaR1T' +
                'SNKEanuRFT3nvd/7I+clLYTQzx2EGuq+qdYGiqQe646xtD4AIQid' +
                '8Dqqof14/7daQxwSPPJgiD0JHM0l2b+gd+xmFswfXbGx9KCLfdNY' +
                'SVf6Iwh57y5xjB/3yRS8AXkElJbE7BoeCCeMxv4x396dx2XCittS' +
                '66m+6INj9cFVq2nsa/svaGFfW9K3lmG/G+tnxBMQwrgrbz0g/ydn' +
                'x+Jo/QaHExC90Q+4FbxEf75rDl+A1plPVZymxdH6aQh4cI+D5Pyu' +
                'yrmgAignYmZwyvD6EBIf1BEowFnWpyvA1MzNzKvzBi8Rk1fWL4+/' +
                'Odcifwcdl/TLG9dN+bvopNPJ1fctq16td1nbJ5T40n9Z/6o8PE3z' +
                'itZzdPKpdB7fWJsy0ZYc8nOWzZOeFB6hkJonf9xq5/y2FVfr7Dng' +
                'JGQymHPOMzh7FmcvzWkTOobAyKtqXlV1/mucqvtU9J5Vx7LzT3w8' +
                'gUUZlXKqvmf194OVl9ZZ9F6+aPB78A6d1993e36tel4uAYLUv4vB' +
                '62fwDiye1qP/sq+zCKa+rlyG4ANdB9ec942Mfm0ozW02K/ve5onn' +
                'NBrPfp8L+NVim4OPQ3VFcX+hYufi8A37RNZl/xf1ffWE')
        pipeline = cpp.Pipeline()
        def handle_error(caller, event):
            if isinstance(event, cpp.LoadExceptionEvent):
                self.fail(event.error.message)
                
        pipeline.add_listener(handle_error)
        fd = StringIO.StringIO(base64.b64decode(data))
        pipeline.load(fd)
        
        self.assertEqual(len(pipeline.modules()), 3)
        klo, fbom1, fbom2 = pipeline.modules()
        self.assertEqual(klo.object_name.value, 'TargetObjects')
        self.assertEqual(klo.enclosing_object_name.value, 'FilteredNuclei')
        self.assertEqual(klo.target_name.value, 'TargetObjects')
        self.assertEqual(klo.filter_choice.value, F.FI_MAXIMAL_PER_OBJECT)
        self.assertEqual(klo.measurements[0].measurement.value, 'AreaShape_Area')
        self.assertFalse(klo.wants_outlines.value)
        
        self.assertEqual(fbom1.object_name.value,'LargestObjects')
        self.assertEqual(fbom1.target_name.value,'FilteredNuclei')
        self.assertEqual(fbom1.filter_choice.value, F.FI_LIMITS)
        self.assertEqual(fbom1.measurements[0].measurement.value, 'AreaShape_Perimeter')
        self.assertTrue(fbom1.measurements[0].wants_minimum.value)
        self.assertEqual(fbom1.measurements[0].min_limit.value, 200)
        self.assertFalse(fbom1.measurements[0].wants_maximum.value)
        self.assertFalse(fbom1.wants_outlines.value)
        
        self.assertEqual(fbom2.object_name.value,'TargetObjects')
        self.assertEqual(fbom2.target_name.value,'FilteredNuclei')
        self.assertEqual(fbom2.filter_choice.value, F.FI_LIMITS)
        self.assertEqual(fbom2.measurements[0].measurement.value, 'Intensity_MeanIntensity')
        self.assertFalse(fbom2.measurements[0].wants_minimum.value)
        self.assertTrue(fbom2.measurements[0].wants_maximum.value)
        self.assertEqual(fbom2.measurements[0].max_limit.value, .25)
        self.assertTrue(fbom2.wants_outlines.value)
        self.assertEqual(fbom2.outlines_name.value, 'OutlineObjects')
        
    def test_05_02_load(self):
        '''Load a pipeline saved by pyCP'''
        data = ('TUFUTEFCIDUuMCBNQVQtZmlsZSBQbGF0Zm9ybTogbnQsIENyZWF0ZW' +
                'Qgb246IFdlZCBBcHIgMjIgMTM6MzA6MTQgMjAwOQAAAAAAAAAAAAAA' +
                'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA' +
                'AAAAABSU0OAAAAmDoAAAYAAAAIAAAAAgAAAAAAAAAFAAAACAAAAAEA' +
                'AAABAAAAAQAAAAgAAABTZXR0aW5ncwUABAAYAAAAAQAAAMAAAABWYX' +
                'JpYWJsZVZhbHVlcwAAAAAAAAAAAABWYXJpYWJsZUluZm9UeXBlcwAA' +
                'AAAAAABNb2R1bGVOYW1lcwAAAAAAAAAAAAAAAABOdW1iZXJzT2ZWYX' +
                'JpYWJsZXMAAAAAAABQaXhlbFNpemUAAAAAAAAAAAAAAAAAAABWYXJp' +
                'YWJsZVJldmlzaW9uTnVtYmVycwBNb2R1bGVSZXZpc2lvbk51bWJlcn' +
                'MAAABNb2R1bGVOb3RlcwAAAAAAAAAAAAAAAAAOAAAAQBsAAAYAAAAI' +
                'AAAAAQAAAAAAAAAFAAAACAAAAAUAAAAWAAAAAQAAAAAAAAAOAAAASA' +
                'AAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAARAAAAAQAAAAAA' +
                'AAAQAAAAEQAAAGluZGl2aWR1YWwgaW1hZ2VzAAAAAAAAAA4AAAAwAA' +
                'AABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAA' +
                'ABAAAwBETkEADgAAADgAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAA' +
                'ABAAAABgAAAAEAAAAAAAAAEAAAAAYAAABOdWNsZWkAAA4AAAAwAAAA' +
                'BgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAAAB' +
                'AAAwBETkEADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAAB' +
                'AAAADgAAAAEAAAAAAAAAEAAAAA4AAABGaWx0ZXJlZE51Y2xlaQAADg' +
                'AAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAEAAAAAEA' +
                'AAAAAAAAEAAAABAAAABUZXh0LUV4YWN0IG1hdGNoDgAAADgAAAAGAA' +
                'AACAAAAAQAAAAAAAAABQAAAAgAAAABAAAABgAAAAEAAAAAAAAAEAAA' +
                'AAYAAABOdWNsZWkAAA4AAAA4AAAABgAAAAgAAAAEAAAAAAAAAAUAAA' +
                'AIAAAAAQAAAAUAAAABAAAAAAAAABAAAAAFAAAAQ2VsbHMAAAAOAAAA' +
                'QAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAKAAAAAQAAAA' +
                'AAAAAQAAAACgAAAERvIG5vdCB1c2UAAAAAAAAOAAAAOAAAAAYAAAAI' +
                'AAAABAAAAAAAAAAFAAAACAAAAAEAAAAGAAAAAQAAAAAAAAAQAAAABg' +
                'AAAE51Y2xlaQAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgA' +
                'AAABAAAAAQAAAAEAAAAAAAAAEAABADMAAAAOAAAAOAAAAAYAAAAIAA' +
                'AABAAAAAAAAAAFAAAACAAAAAEAAAAFAAAAAQAAAAAAAAAQAAAABQAA' +
                'ADEwLDQwAAAADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAA' +
                'ABAAAACwAAAAEAAAAAAAAAEAAAAAsAAABQcm9wYWdhdGlvbgAAAAAA' +
                'DgAAADgAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAABgAAAA' +
                'EAAAAAAAAAEAAAAAYAAABOdWNsZWkAAA4AAABYAAAABgAAAAgAAAAE' +
                'AAAAAAAAAAUAAAAIAAAAAQAAACEAAAABAAAAAAAAABAAAAAhAAAASW' +
                '50ZW5zaXR5X0ludGVncmF0ZWRJbnRlbnNpdHlfRE5BAAAAAAAAAA4A' +
                'AABAAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAoAAAABAA' +
                'AAAAAAABAAAAAKAAAARG8gbm90IHVzZQAAAAAAAA4AAAAwAAAABgAA' +
                'AAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAAABAAAw' +
                'BZZXMADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAA' +
                'CQAAAAEAAAAAAAAAEAAAAAkAAABDeXRvcGxhc20AAAAAAAAADgAAAD' +
                'AAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAA' +
                'AAAACQAAAAAAAAAOAAAAOAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACA' +
                'AAAAEAAAAGAAAAAQAAAAAAAAAQAAAABgAAAExpbWl0cwAADgAAADAA' +
                'AAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAgAAAAEAAAAAAA' +
                'AAEAACAE5vAAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAA' +
                'AAEAAAACAAAAAQAAAAAAAAAQAAIATm8AAA4AAABAAAAABgAAAAgAAA' +
                'AEAAAAAAAAAAUAAAAIAAAAAQAAAAsAAAABAAAAAAAAABAAAAALAAAA' +
                'T3RzdSBHbG9iYWwAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAA' +
                'UAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAG' +
                'AAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAABAAAAAEAAAAAAAAAEA' +
                'AEAE5vbmUOAAAASAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEA' +
                'AAAXAAAAAQAAAAAAAAAQAAAAFwAAAERlZmF1bHQgSW1hZ2UgRGlyZW' +
                'N0b3J5AA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAA' +
                'AAMAAAABAAAAAAAAABAAAwBZZXMADgAAADAAAAAGAAAACAAAAAQAAA' +
                'AAAAAABQAAAAgAAAABAAAAAwAAAAEAAAAAAAAAEAADADEuMAAOAAAA' +
                'MAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAA' +
                'AAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAI' +
                'AAAAAQAAAAMAAAABAAAAAAAAABAAAwBZZXMADgAAADAAAAAGAAAACA' +
                'AAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAA' +
                'AAAOAAAAQAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAALAA' +
                'AAAQAAAAAAAAAQAAAACwAAAE90c3UgR2xvYmFsAAAAAAAOAAAASAAA' +
                'AAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAARAAAAAQAAAAAAAA' +
                'AQAAAAEQAAADAuMDAwMDAwLDEuMDAwMDAwAAAAAAAAAA4AAAAwAAAA' +
                'BgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAA' +
                'kAAAAAAAAADgAAADgAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAAB' +
                'AAAABQAAAAEAAAAAAAAAEAAAAAUAAAAzMDAuMAAAAA4AAAAwAAAABg' +
                'AAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAAABAA' +
                'AwBZZXMADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAA' +
                'AAAwAAAAEAAAAAAAAAEAADADEuMAAOAAAAMAAAAAYAAAAIAAAABAAA' +
                'AAAAAAAFAAAACAAAAAEAAAAEAAAAAQAAAAAAAAAQAAQAMC4wMQ4AAA' +
                'AwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAA' +
                'AAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAA' +
                'gAAAABAAAAAwAAAAEAAAAAAAAAEAADAFllcwAOAAAAMAAAAAYAAAAI' +
                'AAAABAAAAAAAAAAFAAAACAAAAAEAAAADAAAAAQAAAAAAAAAQAAMAWW' +
                'VzAA4AAABIAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAABEA' +
                'AAABAAAAAAAAABAAAAARAAAAMC4wMDAwMDAsMS4wMDAwMDAAAAAAAA' +
                'AADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAgAA' +
                'AAEAAAAAAAAAEAACADEwAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAA' +
                'AFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAA4AAAA' +
                'BgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAUAAAABAAAAAAAAAB' +
                'AAAAAFAAAANTAwLjAAAAAOAAAAOAAAAAYAAAAIAAAABAAAAAAAAAAF' +
                'AAAACAAAAAEAAAAIAAAAAQAAAAAAAAAQAAAACAAAAENoYW5uZWwyDg' +
                'AAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAABAAAAAEA' +
                'AAAAAAAAEAAEADAuMDEOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAA' +
                'AACAAAAAEAAAAEAAAAAQAAAAAAAAAQAAQAMC4wNQ4AAAAwAAAABgAA' +
                'AAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAA' +
                'AAAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAA' +
                'AgAAAAEAAAAAAAAAEAACAE5vAAAOAAAAMAAAAAYAAAAIAAAABAAAAA' +
                'AAAAAFAAAACAAAAAEAAAADAAAAAQAAAAAAAAAQAAMARE5BAA4AAABA' +
                'AAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAkAAAABAAAAAA' +
                'AAABAAAAAJAAAASW50ZW5zaXR5AAAAAAAAAA4AAABIAAAABgAAAAgA' +
                'AAAEAAAAAAAAAAUAAAAIAAAAAQAAABEAAAABAAAAAAAAABAAAAARAA' +
                'AAU2Vjb25kYXJ5T3V0bGluZXMAAAAAAAAADgAAADAAAAAGAAAACAAA' +
                'AAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAA' +
                'AOAAAAQAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAMAAAA' +
                'AQAAAAAAAAAQAAAADAAAAEZpbHRlcmVkQmx1ZQAAAAAOAAAAMAAAAA' +
                'YAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAABAAAAAQAAAAAAAAAQ' +
                'AAEAMQAAAA4AAABAAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQ' +
                'AAAAkAAAABAAAAAAAAABAAAAAJAAAASW50ZW5zaXR5AAAAAAAAAA4A' +
                'AAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAIAAAABAA' +
                'AAAAAAABAAAgBObwAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAA' +
                'AAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAOAAAAAYAAA' +
                'AIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAFAAAAAQAAAAAAAAAQAAAA' +
                'BQAAAENlbGxzAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAA' +
                'gAAAABAAAABAAAAAEAAAAAAAAAEAAEAE5vbmUOAAAAMAAAAAYAAAAI' +
                'AAAABAAAAAAAAAAFAAAACAAAAAEAAAACAAAAAQAAAAAAAAAQAAIAMT' +
                'AAAA4AAAAwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAMA' +
                'AAABAAAAAAAAABAAAwAwLjAADgAAADAAAAAGAAAACAAAAAYAAAAAAA' +
                'AABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAQAAA' +
                'AAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAANAAAAAQAAAAAAAA' +
                'AQAAAADQAAAEZpbHRlcmVkQ2VsbHMAAAAOAAAAgAAAAAYAAAAIAAAA' +
                'BAAAAAAAAAAFAAAACAAAAAEAAABLAAAAAQAAAAAAAAAQAAAASwAAAF' +
                '4oP1A8UGxhdGU+LispXyg/UDxXZWxsUm93PltBLVBdKSg/UDxXZWxs' +
                'Q29sdW1uPlswLTldezEsMn0pXyg/UDxTaXRlPlswLTldKQAAAAAADg' +
                'AAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAAAQAAAAEA' +
                'AAAAAAAAEAABADcAAAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAA' +
                'AACAAAAAEAAAAEAAAAAQAAAAAAAAAQAAQATm9uZQ4AAAAwAAAABgAA' +
                'AAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAA' +
                'AAAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAA' +
                'AgAAAAEAAAAAAAAAEAACAE5vAAAOAAAAaAAAAAYAAAAIAAAABAAAAA' +
                'AAAAAFAAAACAAAAAEAAAA4AAAAAQAAAAAAAAAQAAAAOAAAACg/UDxZ' +
                'ZWFyPlswLTldezR9KS0oP1A8TW9udGg+WzAtOV17Mn0pLSg/UDxEYX' +
                'k+WzAtOV17Mn0pDgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgA' +
                'AAABAAAAAwAAAAEAAAAAAAAAEAADAFllcwAOAAAAMAAAAAYAAAAIAA' +
                'AABAAAAAAAAAAFAAAACAAAAAEAAAACAAAAAQAAAAAAAAAQAAIATm8A' +
                'AA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAA' +
                'ABAAAAAAAAAAkAAAAAAAAADgAAAEgAAAAGAAAACAAAAAQAAAAAAAAA' +
                'BQAAAAgAAAABAAAAFQAAAAEAAAAAAAAAEAAAABUAAABPdXRsaW5lc0' +
                'ZpbHRlcmVkR3JlZW4AAAAOAAAAOAAAAAYAAAAIAAAABAAAAAAAAAAF' +
                'AAAACAAAAAEAAAAIAAAAAQAAAAAAAAAQAAAACAAAAENoYW5uZWwxDg' +
                'AAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACgAAAAEA' +
                'AAAAAAAAEAAAAAoAAABEbyBub3QgdXNlAAAAAAAADgAAADAAAAAGAA' +
                'AACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAA' +
                'AAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAA' +
                'AAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAA' +
                'AAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAAE' +
                'AAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACQAAAAEAAAAA' +
                'AAAAEAAAAAkAAABDeXRvcGxhc20AAAAAAAAADgAAADAAAAAGAAAACA' +
                'AAAAQAAAAAAAAABQAAAAgAAAABAAAAAwAAAAEAAAAAAAAAEAADAFll' +
                'cwAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAA' +
                'AAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAA' +
                'AAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAA' +
                'AGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAA' +
                'CQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAA' +
                'EAAAABAAAAAQAAAAAAAAAQAAEAMgAAAA4AAAAwAAAABgAAAAgAAAAE' +
                'AAAAAAAAAAUAAAAIAAAAAQAAAAMAAAABAAAAAAAAABAAAwBZZXMADg' +
                'AAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEA' +
                'AAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAA' +
                'AACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAA' +
                'AAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAA' +
                'AAAAAADgAAADAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAA' +
                'BAAAAAEAAAAAAAAAEAAEAE5vbmUOAAAAMAAAAAYAAAAIAAAABAAAAA' +
                'AAAAAFAAAACAAAAAEAAAADAAAAAQAAAAAAAAAQAAMAWWVzAA4AAAAw' +
                'AAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAA' +
                'AAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgA' +
                'AAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAA' +
                'AABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAA' +
                'AA4AAACAAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAEsAAA' +
                'ABAAAAAAAAABAAAABLAAAAXig/UDxQbGF0ZT4uKylfKD9QPFdlbGxS' +
                'b3c+W0EtUF0pKD9QPFdlbGxDb2x1bW4+WzAtOV17MSwyfSlfKD9QPF' +
                'NpdGU+WzAtOV0pAAAAAAAOAAAAMAAAAAYAAAAIAAAABAAAAAAAAAAF' +
                'AAAACAAAAAEAAAADAAAAAQAAAAAAAAAQAAMAMC4wAA4AAAAwAAAABg' +
                'AAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkA' +
                'AAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAA' +
                'AAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAA' +
                'AAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAA' +
                'BoAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAADgAAAABAAAA' +
                'AAAAABAAAAA4AAAAKD9QPFllYXI+WzAtOV17NH0pLSg/UDxNb250aD' +
                '5bMC05XXsyfSktKD9QPERheT5bMC05XXsyfSkOAAAAMAAAAAYAAAAI' +
                'AAAABAAAAAAAAAAFAAAACAAAAAEAAAAEAAAAAQAAAAAAAAAQAAQATm' +
                '9uZQ4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAA' +
                'AAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAA' +
                'AABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAA' +
                'AAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAA' +
                'AJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAA' +
                'AAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAA' +
                'QAAAAAAAAABQAAAAgAAAABAAAAAgAAAAEAAAAAAAAAEAACAE5vAAAO' +
                'AAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQ' +
                'AAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUA' +
                'AAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAA' +
                'AACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAA' +
                'AAAAAAAOAAAAqBkAAAYAAAAIAAAAAQAAAAAAAAAFAAAACAAAAAUAAA' +
                'AWAAAAAQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAA' +
                'CAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAABAAAAABgAAAA' +
                'gAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAAoAAAABAAAAAAAAABAAAAAK' +
                'AAAAaW1hZ2Vncm91cAAAAAAAAA4AAABAAAAABgAAAAgAAAAEAAAAAA' +
                'AAAAUAAAAIAAAAAQAAAAsAAAABAAAAAAAAABAAAAALAAAAb2JqZWN0' +
                'Z3JvdXAAAAAAAA4AAABAAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAA' +
                'AAAQAAAAoAAAABAAAAAAAAABAAAAAKAAAAaW1hZ2Vncm91cAAAAAAA' +
                'AA4AAABIAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAABEAAA' +
                'ABAAAAAAAAABAAAAARAAAAb2JqZWN0Z3JvdXAgaW5kZXAAAAAAAAAA' +
                'DgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAA' +
                'EAAAAAAAAACQAAAAAAAAAOAAAASAAAAAYAAAAIAAAABAAAAAAAAAAF' +
                'AAAACAAAAAEAAAARAAAAAQAAAAAAAAAQAAAAEQAAAG9iamVjdGdyb3' +
                'VwIGluZGVwAAAAAAAAAA4AAABIAAAABgAAAAgAAAAEAAAAAAAAAAUA' +
                'AAAIAAAAAQAAABEAAAABAAAAAAAAABAAAAARAAAAb2JqZWN0Z3JvdX' +
                'AgaW5kZXAAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAA' +
                'AAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAQAAAAAYAAA' +
                'AIAAAABAAAAAAAAAAFAAAACAAAAAEAAAALAAAAAQAAAAAAAAAQAAAA' +
                'CwAAAG9iamVjdGdyb3VwAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAA' +
                'AAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAw' +
                'AAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAA' +
                'AAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgA' +
                'AAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAQAAAAAYAAAAIAA' +
                'AABAAAAAAAAAAFAAAACAAAAAEAAAALAAAAAQAAAAAAAAAQAAAACwAA' +
                'AG9iamVjdGdyb3VwAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAA' +
                'AFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAA' +
                'BgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAA' +
                'kAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAA' +
                'AAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAQAAAAAYAAAAIAAAABA' +
                'AAAAAAAAAFAAAACAAAAAEAAAAKAAAAAQAAAAAAAAAQAAAACgAAAGlt' +
                'YWdlZ3JvdXAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAA' +
                'AACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAA' +
                'AAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAA' +
                'AAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAA' +
                'AAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAA' +
                'AAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAw' +
                'AAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAA' +
                'AAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgA' +
                'AAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAQAAAAAYAAAAIAA' +
                'AABAAAAAAAAAAFAAAACAAAAAEAAAALAAAAAQAAAAAAAAAQAAAACwAA' +
                'AG9iamVjdGdyb3VwAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAA' +
                'AFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAA' +
                'BgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAA' +
                'kAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAA' +
                'AAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABg' +
                'AAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4A' +
                'AAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAA' +
                'AAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAA' +
                'AAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAA' +
                'AIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAA' +
                'AAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAA' +
                'AAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAA' +
                'AAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMA' +
                'AAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAA' +
                'AAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAA' +
                'AAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAA' +
                'AAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAA' +
                'AOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAA' +
                'AQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAA' +
                'UAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAG' +
                'AAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQ' +
                'AAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAA' +
                'AAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAA' +
                'AAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAA' +
                'ADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAA' +
                'AAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAA' +
                'CAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAA' +
                'gAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAA' +
                'AAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAA' +
                'AAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAA' +
                'AAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAA' +
                'AABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAA' +
                'AAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAA' +
                'AAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAA' +
                'BgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA' +
                '4AAABAAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAABAAAAAB' +
                'AAAAAAAAABAAAAAQAAAAaW1hZ2Vncm91cCBpbmRlcA4AAAAwAAAABg' +
                'AAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkA' +
                'AAAAAAAADgAAAEgAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAA' +
                'AAEgAAAAEAAAAAAAAAEAAAABIAAABvdXRsaW5lZ3JvdXAgaW5kZXAA' +
                'AAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAA' +
                'AAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAABAAAAABgAAAAgAAAAEAAAA' +
                'AAAAAAUAAAAIAAAAAQAAABAAAAABAAAAAAAAABAAAAAQAAAAaW1hZ2' +
                'Vncm91cCBpbmRlcA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAI' +
                'AAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACA' +
                'AAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAA' +
                'AAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAA' +
                'AAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAA' +
                'AAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAAEAAAA' +
                'AGAAAACAAAAAQAAAAAAAAABQAAAAgAAAABAAAACwAAAAEAAAAAAAAA' +
                'EAAAAAsAAABvYmplY3Rncm91cAAAAAAADgAAADAAAAAGAAAACAAAAA' +
                'YAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAO' +
                'AAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQ' +
                'AAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUA' +
                'AAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAA' +
                'AACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAA' +
                'AAAAAAAOAAAASAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAA' +
                'ARAAAAAQAAAAAAAAAQAAAAEQAAAG9iamVjdGdyb3VwIGluZGVwAAAA' +
                'AAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAA' +
                'AAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAA' +
                'AAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAQA' +
                'AAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAKAAAAAQAAAAAA' +
                'AAAQAAAACgAAAGltYWdlZ3JvdXAAAAAAAAAOAAAAMAAAAAYAAAAIAA' +
                'AABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAA' +
                'AA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAA' +
                'ABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAA' +
                'BQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAA' +
                'YAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJ' +
                'AAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAA' +
                'AAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYA' +
                'AAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAA' +
                'AAQAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAAAQAAAAAQAA' +
                'AAAAAAAQAAAAEAAAAGltYWdlZ3JvdXAgaW5kZXAOAAAAMAAAAAYAAA' +
                'AIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAA' +
                'AAAAAA4AAABIAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAAAB' +
                'IAAAABAAAAAAAAABAAAAASAAAAb3V0bGluZWdyb3VwIGluZGVwAAAA' +
                'AAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAA' +
                'AAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAA' +
                'AAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAA' +
                'AABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAA' +
                'AAkAAAAAAAAADgAAAEAAAAAGAAAACAAAAAQAAAAAAAAABQAAAAgAAA' +
                'ABAAAAEAAAAAEAAAAAAAAAEAAAABAAAABpbWFnZWdyb3VwIGluZGVw' +
                'DgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAA' +
                'EAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAF' +
                'AAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABg' +
                'AAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkA' +
                'AAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAA' +
                'AAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAA' +
                'AAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAA' +
                'AwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAA' +
                'AAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAA' +
                'gAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAI' +
                'AAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAA' +
                'AAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAA' +
                'AAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAA' +
                'AABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAMAAA' +
                'AAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQAAAAAAAA' +
                'AJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUAAAAIAAAA' +
                'AAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAAAACAAAAA' +
                'YAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAAAAAAAAAO' +
                'AAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAAAAAAAAAQ' +
                'AAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAAAAUA' +
                'AAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAAAGAA' +
                'AACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAACQAA' +
                'AAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAAAAAA' +
                'AAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAA' +
                'AAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAAD' +
                'AAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAA' +
                'AAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACA' +
                'AAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAABAAAAABgAAAAgA' +
                'AAAEAAAAAAAAAAUAAAAIAAAAAQAAAAoAAAABAAAAAAAAABAAAAAKAA' +
                'AAaW1hZ2Vncm91cAAAAAAAAA4AAAAwAAAABgAAAAgAAAAGAAAAAAAA' +
                'AAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADgAAADAAAA' +
                'AGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEAAAAAAAAA' +
                'CQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAAAACAAAAA' +
                'AAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAAAAgAAAAG' +
                'AAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAAAAAAAADg' +
                'AAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAAAAAAAAEA' +
                'AAAAAAAACQAAAAAAAAAOAAAAMAAAAAYAAAAIAAAABgAAAAAAAAAFAA' +
                'AACAAAAAAAAAAAAAAAAQAAAAAAAAAJAAAAAAAAAA4AAAAwAAAABgAA' +
                'AAgAAAAGAAAAAAAAAAUAAAAIAAAAAAAAAAAAAAABAAAAAAAAAAkAAA' +
                'AAAAAADgAAADAAAAAGAAAACAAAAAYAAAAAAAAABQAAAAgAAAAAAAAA' +
                'AAAAAAEAAAAAAAAACQAAAAAAAAAOAAAAeAIAAAYAAAAIAAAAAQAAAA' +
                'AAAAAFAAAACAAAAAEAAAAFAAAAAQAAAAAAAAAOAAAAYAAAAAYAAAAI' +
                'AAAABAAAAAAAAAAFAAAACAAAAAEAAAAqAAAAAQAAAAAAAAAQAAAAKg' +
                'AAAGNlbGxwcm9maWxlci5tb2R1bGVzLmxvYWRpbWFnZXMuTG9hZElt' +
                'YWdlcwAAAAAAAA4AAABwAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAA' +
                'AAAQAAAEAAAAABAAAAAAAAABAAAABAAAAAY2VsbHByb2ZpbGVyLm1v' +
                'ZHVsZXMuaWRlbnRpZnlwcmltYXV0b21hdGljLklkZW50aWZ5UHJpbU' +
                'F1dG9tYXRpYw4AAABoAAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAA' +
                'AQAAADgAAAABAAAAAAAAABAAAAA4AAAAY2VsbHByb2ZpbGVyLm1vZH' +
                'VsZXMuaWRlbnRpZnlzZWNvbmRhcnkuSWRlbnRpZnlTZWNvbmRhcnkO' +
                'AAAAeAAAAAYAAAAIAAAABAAAAAAAAAAFAAAACAAAAAEAAABCAAAAAQ' +
                'AAAAAAAAAQAAAAQgAAAGNlbGxwcm9maWxlci5tb2R1bGVzLm1lYXN1' +
                'cmVvYmplY3RpbnRlbnNpdHkuTWVhc3VyZU9iamVjdEludGVuc2l0eQ' +
                'AAAAAAAA4AAAB4AAAABgAAAAgAAAAEAAAAAAAAAAUAAAAIAAAAAQAA' +
                'AEgAAAABAAAAAAAAABAAAABIAAAAY2VsbHByb2ZpbGVyLm1vZHVsZX' +
                'MuZmlsdGVyYnlvYmplY3RtZWFzdXJlbWVudC5GaWx0ZXJCeU9iamVj' +
                'dE1lYXN1cmVtZW50DgAAADgAAAAGAAAACAAAAAkAAAAAAAAABQAAAA' +
                'gAAAABAAAABQAAAAEAAAAAAAAAAgAAAAUAAAAVFg8DDwAAAA4AAAAw' +
                'AAAABgAAAAgAAAAGAAAAAAAAAAUAAAAAAAAAAQAAAAAAAAAJAAAACA' +
                'AAAAAAAAAAAPA/DgAAADgAAAAGAAAACAAAAAkAAAAAAAAABQAAAAgA' +
                'AAABAAAABQAAAAEAAAAAAAAAAgAAAAUAAAACAQECAQAAAA4AAABAAA' +
                'AABgAAAAgAAAALAAAAAAAAAAUAAAAIAAAAAQAAAAUAAAABAAAAAAAA' +
                'AAQAAAAKAAAAAAAAAAAAAAAAAAAAAAAAAA4AAAAYAQAABgAAAAgAAA' +
                'ABAAAAAAAAAAUAAAAIAAAAAQAAAAUAAAABAAAAAAAAAA4AAAAoAAAA' +
                'BgAAAAgAAAABAAAAAAAAAAUAAAAIAAAAAAAAAAEAAAABAAAAAAAAAA' +
                '4AAAAoAAAABgAAAAgAAAABAAAAAAAAAAUAAAAIAAAAAAAAAAEAAAAB' +
                'AAAAAAAAAA4AAAAoAAAABgAAAAgAAAABAAAAAAAAAAUAAAAIAAAAAA' +
                'AAAAEAAAABAAAAAAAAAA4AAAAoAAAABgAAAAgAAAABAAAAAAAAAAUA' +
                'AAAIAAAAAAAAAAEAAAABAAAAAAAAAA4AAAAoAAAABgAAAAgAAAABAA' +
                'AAAAAAAAUAAAAIAAAAAAAAAAEAAAABAAAAAAAAAA==')
        pipeline = cpp.Pipeline()
        def handle_error(caller, event):
            if isinstance(event, cpp.LoadExceptionEvent):
                self.fail(event.error.message)
                
        pipeline.add_listener(handle_error)
        fd = StringIO.StringIO(base64.b64decode(data))
        pipeline.load(fd)
        module = pipeline.modules()[4]
        self.assertEqual(module.target_name.value, 'FilteredNuclei')
        self.assertEqual(module.object_name.value, 'Nuclei')
        self.assertEqual(module.measurements[0].measurement.value, 'Intensity_IntegratedIntensity_DNA')
        self.assertEqual(module.filter_choice.value, F.FI_LIMITS)
        self.assertTrue(module.measurements[0].wants_minimum.value)
        self.assertEqual(module.measurements[0].min_limit.value, 300)
        self.assertTrue(module.measurements[0].wants_maximum.value)
        self.assertEqual(module.measurements[0].max_limit.value, 500)
        self.assertEqual(len(module.additional_objects), 1)
        self.assertEqual(module.additional_objects[0].object_name.value, 'Cells')
        self.assertEqual(module.additional_objects[0].target_name.value, 'FilteredCells')

    def test_05_03_test_load_v2(self):
        data = ('eJztW91u2zYUll0naNYtS4FdDBsK6KIXTRCrclKjbTa0tuNlMRA7Rhw0G9Is'
                'ky065kCJhkSl8Ya94x5jl32EkZZsSZxcyYr/hEqAIB2K3/nOOTw6ov7q5fOT'
                'ckUsSrJYL5/nuxABsYkU0sWGdiDqZFc8NIBCgCpi/UC8oNsq6Ih7+2LhxcF+'
                '8UAuinuy/FqIt2Rq9U26+ecHQVin24d0zTqH1hw541mZ3AKEQP3GXBNywrdO'
                '+0e6vlMMqLQReKcgC5guxai9pnfx+aA/PlTHqoVAQ9G8nenSsLQ2MMzT7gjo'
                'HG7CO4Ba8E/AuTDqdgZuoQmx7uAd/XzrmBcTjrfVwx+ODGoOp7+ikE6vRegI'
                '+NtZ3I6/d+OW4eKWo+sTT/uwv+D2zwXE+bGn/5YjQ12Ft1C1FCRCTbkZW830'
                'ySH6Hvj0PRCqjfIQ9yoEt87ZweSG1UEA2rylEPyXHJ7JRxARYAC1QtNjZH+Y'
                'ni1OD1vPwR3J/3SndIiosaGZhT/TxvFXOgqL4M348Blh34lbGO8ax8vkgrz7'
                'Qhb+z7vO4UfLCL/hbKOM1ybHy+QyrV6tntIH12wvmp4vOD1MrmJRx0S0TODa'
                'E3fc4vofZ7xPoAaJGc3erA+fFRp4drhp/Azjy/n4cpRPB1Hq2zdcfJhcBV3F'
                'QkSsseImHmGkAiOqHbMe30XzlUL4HnHxYvIpMS3xZ4TbCgq0e57+ypI8szhF'
                'wRUC+BY9ntOcb3Gu7zSmw2W34OxMsH+efgfFeZb1ia8X1OfCPP2bdT0shfBt'
                'CP5xZXJNJ0A3IRlMsHuW/LOabwX5cdhTdB2gvfwKxSPO/OfMsm8l7jNvnjbv'
                'CvJ8x/0rzk8mH1omwZrYHV1I48wzCzFxLwNw8zyvpef3m68sc34xDa4XYudL'
                'wZ8HTP7t2dvmj+xBAngj7WxfM+kCIPTmspxvXl3K+ddXf+39vX1tsgMtSHsN'
                '27Yj5d3XHB+Tmwa9PTUGpxZBUF9SnH4JsfspZzeTpZ3L9++fX7EwVJ1gjRvO'
                'LJ3JT4PsWuV8Scr8N/Vvuf4FzeeT7F+UOp9k/+Z5n78K/n1+519x6XbO+3nF'
                '+QcsdpBims6T6yT5exzib9D9/AWANz32uuaWvZjQO8CjLyl+l0L8Dno+e4QN'
                'cGNgS1eT5+/nXpf4+7jikux89Z2Ly3C4oPdpi8zv4cs3luD96HqC6iFu/wE6'
                'xFUUt8549IhQV0F/BnFZFTtKIXZEjWsSz88UlzxcSUjzNcWluBSX4hZRT7cE'
                'fz1lqzs/s6chSfI3jVOKS3GTcSUhzfMUl+JS3GrUm6jPh5Lib4pLcSkuxaW4'
                'FJdk3MOsi+PfHzHZ+30I6/+7hyfoOr/j6b/lyB2AUN/A7D89Q9KGP5OZEsKK'
                'av+dJZ3Q3ZrnRy3G0w/hKXE8pUk8UAU6gd1Bn31sZxGsKQR2pJrTyj7BK49a'
                'Ge9dCG+F461M4tWAYloGsJ9ZKwaV2B89Ut1uPh02j3/0cf0O4z/m+I8n8XeH'
                'HzO3B7YBjjUa9VqyP3OuDGwb6u4RPo82Avi9+ZCl0uMn2c1P5Z8g+PPOzceP'
                'b+Pw5XKZ7PC9pwf3KASXE/znAcP/K0yX988+0X/k4yr3nzbOGbrcN04uT25s'
                'k61/Nfv/B37E+Yc=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, F.FilterByObjectMeasurement))
        self.assertEqual(module.target_name.value, 'FilteredBlue')
        self.assertEqual(module.object_name.value, 'Nuclei')
        self.assertEqual(module.rules_or_measurement, F.ROM_RULES)
        self.assertEqual(module.rules_file_name, 'rules.txt')
        self.assertEqual(module.rules_directory.dir_choice, 
                         cpprefs.DEFAULT_INPUT_SUBFOLDER_NAME )
        self.assertEqual(module.rules_directory.custom_path, "./")
        
    def test_05_04_load_matlab_v7(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUggH0l6JeQoGZgqGhlbGllZGhgpGBoYGCiQDBkZPX34GBoafTAwM'
                'FXPezridd9lA4vhdvdhGR1OzRgZO0+kxPZO4Wso8Ay1DFmUeXeYZs2vtbUnV'
                'wqkWwYdN/Q8FF5x8/IGjoP3usrXZl57dOCwwbc2L4j9///yUlw9mYvhTxXLA'
                'rvpegiW/2soHty0Vi3/zBHatbQxm4Kp4MP1/wk+mWxJlPCWMvOKWCdOWPeVd'
                '7X9u78M3oZO/xDC/vCCivXnJ7jT2m36r7Uu/xG//yCGZ/oObqTsz6q3yrkN8'
                'atPev//89FLk7havt9Mi/sXNej1hcklAhJ3Yqcq/4Qlx8ZwX3gasqH/xvNhk'
                'kf/9noTpq9SW3PU+d9HJopp7jm3A9WyF3viIO6qaeXn7j7K5mr16mXfdQcIl'
                'Kz7ygcaCuGzzST82ms2N/RK97H1mR7zjpem3Q98+aWVb1Ls6epL99tcuWzlC'
                'Y4/PVwi0MFfl2y4t5jqteSX7ouJFx4TDz/vn9OUF9VsLtd/2ZdENnOz7amL8'
                'gyeNftt+2y5aJ3891aZl6/zoB08cRHPK6uQ8ZPK2r3i86j3PpUa3FL+wjyyF'
                'JYtm3Ti0LHvZ45SftZe+sfdtuyxfuSdb/eqVvyedrj8X/PukNmR1xlWLj+y9'
                '5Zc336w8u/KC4NmYdgO/K38NV64Tvf++tuN+rCZfXvLq9vUF51vbunwtZsxN'
                'u35x/Yq/3rviWY8GF2smsqfc8RU8b7T3uqi/0LsDbfMut7IJWXh+P7rj15rq'
                '+l7p87vOW2d+3TjvoM9+/6jYlHPnH736vXpHn8rV0j03eHIe8ZVPn5xiqzr3'
                '/Arv4wpT167NkHBevqDW+8HbmpjZj1j+8s+s3+/5/dI88aZHv8Irbl/S75l8'
                '9c+/9f9f/Tg0L/K4fW3b2mh3oem3jyxl23Yhnjt8zr+Y7893le4+o3v//dPf'
                'EZcq/zQeiHPekeDy8rpt5+G3bvyRIt/3buv1/agj21oe/6hwd+eCx7dWmM7T'
                'O/S09fMtxT3tQv23J/+/vX4v54Xls3XSHO0/ztkmv6h8XqT78aPNJv8NH20L'
                '3zFXv2j/ZYm6rvh/m/5s/fP24/tPP2Q4H/WuVtd+5X953bX/zPOlJoQDAOEw'
                'rqc=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, F.FilterObjects))
        self.assertEqual(module.object_name, "Nucs")
        self.assertEqual(module.target_name, "FilteredNuclei")
        self.assertEqual(module.rules_or_measurement, F.ROM_RULES)
        self.assertEqual(module.rules_directory.dir_choice, 
                         cpprefs.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.rules_file_name, "myrules.txt")
        self.assertEqual(module.measurements[0].measurement, "Intensity_MeanIntensity_DNA_1")
    
    def test_05_05_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8973

FilterObjects:[module_num:1|svn_version:\'8955\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Name the output objects:FilteredThings
    Select the object to filter:Things
    Select the measurement to filter by:Intensity_MeanIntensity_DNA
    Select the filtering method:Minimal
    What did you call the objects that contain the filtered objects?:Nuclei
    Filter using a minimum measurement value?:No
    Minimum value:0
    Filter using a maximum measurement value?:No
    Maximum value:1
    Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?:No
    Name the outline image:None
    Filter using classifier rules or measurements?:Measurements
    Rules file location:Default output folder
    Rules folder name:.
    Rules file name:myrules.txt
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, F.FilterObjects))
        self.assertEqual(module.object_name, "Things")
        self.assertEqual(module.target_name, "FilteredThings")
        self.assertEqual(module.rules_or_measurement, F.ROM_MEASUREMENTS)
        self.assertEqual(module.rules_directory.dir_choice, 
                         cpprefs.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.rules_file_name, "myrules.txt")
        self.assertEqual(module.measurements[0].measurement, "Intensity_MeanIntensity_DNA")
        self.assertEqual(module.filter_choice, F.FI_MINIMAL)
    
    def test_05_06_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9025

LoadImages:[module_num:1|svn_version:\'9020\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    What type of files are you loading?:individual images
    How do you want to load these files?:Text-Exact match
    How many images are there in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:No
    Image location:Default Image Folder
    Enter the full path to the images:
    Do you want to check image sets for missing or duplicate files?:Yes
    Do you want to group image sets by metadata?:No
    Do you want to exclude certain files?:No
    What metadata fields do you want to group by?:
    Type the text that these images have in common (case-sensitive):
    What do you want to call this image in CellProfiler?:DNA
    What is the position of this image in each group?:1
    Do you want to extract metadata from the file name, the subfolder path or both?:None
    Type the regular expression that finds metadata in the file name\x3A:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path\x3A:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$

IdentifyPrimaryObjects:[module_num:2|svn_version:\'9010\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Select the input image:DNA
    Name the identified primary objects:MyObjects
    Typical diameter of objects, in pixel units (Min,Max)\x3A:10,40
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Select the thresholding method:Otsu Global
    Threshold correction factor:1.0
    Lower and upper bounds on threshold\x3A:0.000000,1.000000
    Approximate fraction of image covered by objects?:0.01
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter\x3A:10
    Suppress local maxima within this distance\x3A:7
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:PrimaryOutlines
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum size of local maxima?:Yes
    Enter manual threshold\x3A:0.0
    Select binary image\x3A:None
    Save outlines of the identified objects?:No
    Calculate the Laplacian of Gaussian threshold automatically?:Yes
    Enter Laplacian of Gaussian threshold\x3A:0.5
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter\x3A :5
    How do you want to handle images with large numbers of objects?:No action
    Maximum # of objects\x3A:500

IdentifySecondaryObjects:[module_num:3|svn_version:\'9007\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select the input objects:MyObjects
    Name the identified objects:Cells
    Select the method to identify the secondary objects:Propagation
    Select the input image:DNA
    Select the thresholding method:Otsu Global
    Threshold correction factor:1.0
    Lower and upper bounds on threshold\x3A:0.000000,1.000000
    Approximate fraction of image covered by objects?:0.01
    Number of pixels by which to expand the primary objects\x3A:10
    Regularization factor\x3A:0.05
    Name the outline image:SecondaryOutlines
    Enter manual threshold\x3A:0.0
    Select binary image\x3A:None
    Save outlines of the identified objects?:No
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Do you want to discard objects that touch the edge of the image?:No
    Do you want to discard associated primary objects?:No
    New primary objects name\x3A:FilteredNuclei

IdentifyTertiaryObjects:[module_num:4|svn_version:\'8957\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select the larger identified objects:Cells
    Select the smaller identified objects:MyObjects
    Name the identified subregion objects:Cytoplasm
    Name the outline image:CytoplasmOutlines
    Retain the outlines for use later in the pipeline (for example, in SaveImages)?:No

MeasureObjectIntensity:[module_num:5|svn_version:\'9000\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Hidden:1
    Select an image to use for intensity measurements:DNA
    Select objects to measure:MyObjects

FilterObjects:[module_num:6|svn_version:\'9000\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Name the output objects:MyFilteredObjects
    Select the object to filter:MyObjects
    Filter using classifier rules or measurements?:Measurements
    Select the filtering method:Limits
    What did you call the objects that contain the filtered objects?:None
    Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?:No
    Name the outline image:FilteredObjects
    Rules file location:Default input folder
    Rules folder name:./rules
    Rules file name:myrules.txt
    Hidden:2
    Hidden:2
    Select the measurement to filter by:Intensity_LowerQuartileIntensity_DNA
    Filter using a minimum measurement value?:Yes
    Minimum value:0.2
    Filter using a maximum measurement value?:No
    Maximum value:1.5
    Select the measurement to filter by:Intensity_UpperQuartileIntensity_DNA
    Filter using a minimum measurement value?:No
    Minimum value:0.9
    Filter using a maximum measurement value?:Yes
    Maximum value:1.8
    Select additional object to relabel:Cells
    Name the relabeled objects:FilteredCells
    Save outlines of relabeled objects?:No
    Name the outline image:OutlinesFilteredCells
    Select additional object to relabel:Cytoplasm
    Name the relabeled objects:FilteredCytoplasm
    Save outlines of relabeled objects?:No
    Name the outline image:OutlinesFilteredCytoplasm
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 6)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, F.FilterObjects))
        self.assertEqual(module.target_name, "MyFilteredObjects")
        self.assertEqual(module.object_name, "MyObjects")
        self.assertEqual(module.rules_or_measurement, F.ROM_MEASUREMENTS)
        self.assertEqual(module.filter_choice, F.FI_LIMITS)
        self.assertEqual(module.rules_directory.dir_choice, cpprefs.DEFAULT_INPUT_FOLDER_NAME)
        self.assertEqual(module.rules_directory.custom_path, "./rules")
        self.assertEqual(module.rules_file_name, "myrules.txt")
        self.assertEqual(module.measurement_count.value, 2)
        self.assertEqual(module.additional_object_count.value, 2)
        self.assertEqual(module.measurements[0].measurement, 
                         "Intensity_LowerQuartileIntensity_DNA")
        self.assertTrue(module.measurements[0].wants_minimum)
        self.assertFalse(module.measurements[0].wants_maximum)
        self.assertAlmostEqual(module.measurements[0].min_limit.value, 0.2)
        self.assertAlmostEqual(module.measurements[0].max_limit.value, 1.5)
        self.assertEqual(module.measurements[1].measurement,
                         "Intensity_UpperQuartileIntensity_DNA")
        self.assertFalse(module.measurements[1].wants_minimum)
        self.assertTrue(module.measurements[1].wants_maximum)
        self.assertAlmostEqual(module.measurements[1].min_limit.value, 0.9)
        self.assertAlmostEqual(module.measurements[1].max_limit.value, 1.8)
        for group, name in zip(module.additional_objects,('Cells','Cytoplasm')):
            self.assertEqual(group.object_name, name)
            self.assertEqual(group.target_name, "Filtered%s" % name)
            self.assertEqual(group.outlines_name, "OutlinesFiltered%s" % name)
            self.assertFalse(group.wants_outlines)
        
    def test_05_07_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9025

FilterObjects:[module_num:6|svn_version:\'9000\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Name the output objects:MyFilteredObjects
    Select the object to filter:MyObjects
    Filter using classifier rules or measurements?:Measurements
    Select the filtering method:Limits
    What did you call the objects that contain the filtered objects?:None
    Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?:No
    Name the outline image:FilteredObjects
    Rules file location:Default input folder\x7C./rules
    Rules file name:myrules.txt
    Hidden:2
    Hidden:2
    Select the measurement to filter by:Intensity_LowerQuartileIntensity_DNA
    Filter using a minimum measurement value?:Yes
    Minimum value:0.2
    Filter using a maximum measurement value?:No
    Maximum value:1.5
    Select the measurement to filter by:Intensity_UpperQuartileIntensity_DNA
    Filter using a minimum measurement value?:No
    Minimum value:0.9
    Filter using a maximum measurement value?:Yes
    Maximum value:1.8
    Select additional object to relabel:Cells
    Name the relabeled objects:FilteredCells
    Save outlines of relabeled objects?:No
    Name the outline image:OutlinesFilteredCells
    Select additional object to relabel:Cytoplasm
    Name the relabeled objects:FilteredCytoplasm
    Save outlines of relabeled objects?:No
    Name the outline image:OutlinesFilteredCytoplasm
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, F.FilterObjects))
        self.assertEqual(module.target_name, "MyFilteredObjects")
        self.assertEqual(module.object_name, "MyObjects")
        self.assertEqual(module.rules_or_measurement, F.ROM_MEASUREMENTS)
        self.assertEqual(module.filter_choice, F.FI_LIMITS)
        self.assertEqual(module.rules_directory.dir_choice, cpprefs.DEFAULT_INPUT_FOLDER_NAME)
        self.assertEqual(module.rules_directory.custom_path, "./rules")
        self.assertEqual(module.rules_file_name, "myrules.txt")
        self.assertEqual(module.measurement_count.value, 2)
        self.assertEqual(module.additional_object_count.value, 2)
        self.assertEqual(module.measurements[0].measurement, 
                         "Intensity_LowerQuartileIntensity_DNA")
        self.assertTrue(module.measurements[0].wants_minimum)
        self.assertFalse(module.measurements[0].wants_maximum)
        self.assertAlmostEqual(module.measurements[0].min_limit.value, 0.2)
        self.assertAlmostEqual(module.measurements[0].max_limit.value, 1.5)
        self.assertEqual(module.measurements[1].measurement,
                         "Intensity_UpperQuartileIntensity_DNA")
        self.assertFalse(module.measurements[1].wants_minimum)
        self.assertTrue(module.measurements[1].wants_maximum)
        self.assertAlmostEqual(module.measurements[1].min_limit.value, 0.9)
        self.assertAlmostEqual(module.measurements[1].max_limit.value, 1.8)
        for group, name in zip(module.additional_objects,('Cells','Cytoplasm')):
            self.assertEqual(group.object_name, name)
            self.assertEqual(group.target_name, "Filtered%s" % name)
            self.assertEqual(group.outlines_name, "OutlinesFiltered%s" % name)
            self.assertFalse(group.wants_outlines)
        
    def test_06_01_get_measurement_columns(self):
        '''Test the get_measurement_columns function'''
        workspace, module = self.make_workspace({ "my_objects": np.zeros((10,10),int) })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.measurements[0].measurement.value = "my_measurement"
        module.filter_choice = F.FI_MAXIMAL
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",np.zeros((0,)))
        module.run(workspace)
        image_features = m.get_feature_names(cpm.IMAGE)
        result_features = m.get_feature_names("my_result")
        object_features = m.get_feature_names("my_objects")
        columns = module.get_measurement_columns(workspace.pipeline)
        self.assertEqual(len(columns), 6)
        for feature in image_features:
            self.assertTrue(any([(column[0] == cpm.IMAGE and 
                                  column[1] == feature)
                                 for column in columns]))
        for feature in result_features:
            self.assertTrue(any([(column[0] == "my_result" and
                                  column[1] == feature)
                                 for column in columns]))
        for feature in object_features:
            if feature != 'my_measurement':
                self.assertTrue(any([(column[0] == "my_objects" and
                                      column[1] == feature)
                                     for column in columns]))
        
        for column in columns:
            self.assertTrue(column[0] in (cpm.IMAGE, "my_result", "my_objects"))
            if column[0] == cpm.IMAGE:
                self.assertTrue(column[1] in image_features)
            elif column[0] == "my_result":
                self.assertTrue(column[1] in result_features)
        
        for feature, coltype in (("Location_Center_X", cpm.COLTYPE_FLOAT),
                                 ("Location_Center_Y", cpm.COLTYPE_FLOAT),
                                 ("Number_Object_Number", cpm.COLTYPE_INTEGER),
                                 ("Parent_my_objects", cpm.COLTYPE_INTEGER),
                                 ("Children_my_result_Count", cpm.COLTYPE_INTEGER),
                                 ("Count_my_result", cpm.COLTYPE_INTEGER)):
            fcolumns = [x for x in columns if x[1] == feature]
            self.assertEqual(len(fcolumns),1,"Missing or duplicate column: %s"%feature)
            self.assertEqual(fcolumns[0][2], coltype)
            
        for object_name, category in (("Image",dict(Count=["my_result"])),
                                      ("my_objects",
                                       dict(Children=["my_result_Count"])),
                                      ("my_result",
                                       dict(Location=["Center_X","Center_Y"],
                                            Parent=["my_objects"],
                                            Number=["Object_Number"]))):
            categories = module.get_categories(None, object_name)
            for c in category.keys():
                self.assertTrue(c in categories)
                ff = module.get_measurements(None, object_name, c)
                for f in ff:
                    self.assertTrue(f in category[c])
                                            
    def test_08_01_filter_by_rule(self):
        labels = np.zeros((10,20),int)
        labels[3:5,4:9] = 1
        labels[7:9,6:12] = 2
        labels[4:9, 14:18] = 3
        workspace, module = self.make_workspace({ "MyObjects": labels })
        self.assertTrue(isinstance(module, F.FilterByObjectMeasurement))
        m = workspace.measurements
        m.add_measurement("MyObjects","MyMeasurement", 
                          np.array([ 1.5, 2.3,1.8]))
        rules_file_contents = "IF (MyObjects_MyMeasurement > 2.0, [1.0,-1.0], [-1.0,1.0])\n"
        rules_path = tempfile.mktemp()
        fd = open(rules_path, 'wt')
        try:
            fd.write(rules_file_contents)
            fd.close()
            rules_dir, rules_file = os.path.split(rules_path)
            module.object_name.value = "MyObjects"
            module.rules_or_measurement.value = F.ROM_RULES
            module.rules_file_name.value = rules_file
            module.rules_directory.dir_choice = cpprefs.ABSOLUTE_FOLDER_NAME
            module.rules_directory.custom_path = rules_dir
            module.target_name.value = "MyTargetObjects"
            module.run(workspace)
            target_objects = workspace.object_set.get_objects("MyTargetObjects")
            target_labels = target_objects.segmented
            self.assertTrue(np.all(target_labels[labels == 2] > 0))
            self.assertTrue(np.all(target_labels[labels != 2] == 0))
        finally:
            os.remove(rules_path)

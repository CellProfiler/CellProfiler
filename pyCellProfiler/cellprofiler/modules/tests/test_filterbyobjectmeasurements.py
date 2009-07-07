'''test_filterbyobjectmeasurements.py: Test FilterByObjectMeasurements module
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import base64
import numpy as np
import StringIO
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
        module.min_limit.value = 0
        module.max_limit.value = 1000
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
        parents = m.get_current_measurement("my_result","Parent_my_objects")
        self.assertEqual(len(parents),1)
        self.assertEqual(parents[0],2)
        self.assertEqual(m.get_current_image_measurement("Count_my_result"),1)

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
        module.min_limit.value = my_min
        module.max_limit.value = my_max
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
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_LIMITS
        module.min_limit.value = my_min
        module.max_limit.value = .7
        module.wants_maximum.value = False
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
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_LIMITS
        module.min_limit.value = .3
        module.wants_minimum.value = False
        module.max_limit.value = my_max
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
        module.min_limit.value = my_min
        module.max_limit.value = my_max
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
        self.assertEqual(klo.measurement.value, 'AreaShape_Area')
        self.assertFalse(klo.wants_outlines.value)
        
        self.assertEqual(fbom1.object_name.value,'LargestObjects')
        self.assertEqual(fbom1.target_name.value,'FilteredNuclei')
        self.assertEqual(fbom1.filter_choice.value, F.FI_LIMITS)
        self.assertEqual(fbom1.measurement.value, 'AreaShape_Perimeter')
        self.assertTrue(fbom1.wants_minimum.value)
        self.assertEqual(fbom1.min_limit.value, 200)
        self.assertFalse(fbom1.wants_maximum.value)
        self.assertFalse(fbom1.wants_outlines.value)
        
        self.assertEqual(fbom2.object_name.value,'TargetObjects')
        self.assertEqual(fbom2.target_name.value,'FilteredNuclei')
        self.assertEqual(fbom2.filter_choice.value, F.FI_LIMITS)
        self.assertEqual(fbom2.measurement.value, 'Intensity_MeanIntensity')
        self.assertFalse(fbom2.wants_minimum.value)
        self.assertTrue(fbom2.wants_maximum.value)
        self.assertEqual(fbom2.max_limit.value, .25)
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
        self.assertEqual(module.measurement.value, 'Intensity_IntegratedIntensity_DNA')
        self.assertEqual(module.filter_choice.value, F.FI_LIMITS)
        self.assertTrue(module.wants_minimum.value)
        self.assertEqual(module.min_limit.value, 300)
        self.assertTrue(module.wants_maximum.value)
        self.assertEqual(module.max_limit.value, 500)
        self.assertEqual(len(module.additional_objects), 1)
        self.assertEqual(module.additional_objects[0].object_name.value, 'Cells')
        self.assertEqual(module.additional_objects[0].target_name.value, 'FilteredCells')

    def test_06_01_get_measurement_columns(self):
        '''Test the get_measurement_columns function'''
        workspace, module = self.make_workspace({ "my_objects": np.zeros((10,10),int) })
        module.object_name.value = "my_objects"
        module.target_name.value = "my_result"
        module.measurement.value = "my_measurement"
        module.filter_choice = F.FI_MAXIMAL
        m = workspace.measurements
        m.add_measurement("my_objects","my_measurement",np.zeros((0,)))
        module.run(workspace)
        image_features = m.get_feature_names(cpm.IMAGE)
        object_features = m.get_feature_names("my_result")
        columns = module.get_measurement_columns()
        self.assertEqual(len(columns), 4)
        for feature in image_features:
            self.assertTrue(any([(column[0] == cpm.IMAGE and 
                                  column[1] == feature)
                                 for column in columns]))
        for feature in object_features:
            self.assertTrue(any([(column[0] == "my_result" and
                                  column[1] == feature)
                                 for column in columns]))
        
        for column in columns:
            self.assertTrue(column[0] in (cpm.IMAGE, "my_result"))
            if column[0] == cpm.IMAGE:
                self.assertTrue(column[1] in image_features)
            elif column[0] == "my_result":
                self.assertTrue(column[1] in object_features)
        
        for feature, coltype in (("Location_Center_X", cpm.COLTYPE_FLOAT),
                                 ("Location_Center_Y", cpm.COLTYPE_FLOAT),
                                 ("Parent_my_objects", cpm.COLTYPE_INTEGER),
                                 ("Count_my_result", cpm.COLTYPE_INTEGER)):
            fcolumns = [x for x in columns if x[1] == feature]
            self.assertEqual(len(fcolumns),1,"Missing or duplicate column: %s"%feature)
            self.assertEqual(fcolumns[0][2], coltype)
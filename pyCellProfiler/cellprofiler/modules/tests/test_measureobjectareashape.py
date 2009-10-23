"""test_measureobjectareashape.py - test the MeasureObjectAreaShape module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision: 1 $"

import base64
import numpy as np
import unittest
import StringIO

import cellprofiler.pipeline as cpp
import cellprofiler.modules.measureobjectareashape as cpmoas
import cellprofiler.modules.injectimage as ii
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw
import cellprofiler.objects as cpo
import cellprofiler.cpimage as cpi

class TestMeasureObjectAreaShape(unittest.TestCase):
    def test_01_01_load_matlab(self):
        b64data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBUdWUgTWFyIDEwIDExOjMwOjAyIDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAtwQAAHic7FlPb9MwHHW77k9BTENcOHDoZbdSOduycSx0tOywbuqqSRzd1AtGaVwlztTy6fgIfAQ+CjEkbWJls+OkXUFYsiynfu9nPz//nGX7AIAfJgA7YbsX1ir4U7ajfiVRef8GM0Zc298GNfA6ev49rLfII2jk4FvkBNgHixI/v3Dv6HA+Xfx0SceBg/tokhwcln4wGWHPv7qLgdHP12SGnRvyDYN0iYcN8D3xCXUjfMQvPl3EpUyIu89jvFnqUBF0qIf1VeI5H98Gy/G1DN0OEuMPojrEM/b24wxZrDFBzPrCed5JePYEHt6/8oj9IZRaZR51AV+P8D0PYzexHtk8dgWe3YhngMdK+Kx1dDw6jdchw+8IeN7vB5aDidr8twU873ew4/iK6y+KL7r+rPjnreFFt6T4bQk+y0ccX4aPOI+qj57aB0XjPzV+FflkgC2GXNuJ87NufuQ+mIYXTB5fqvLo+MKAzROouJ7nAp73r8P4yEYsvISUdck8Z3NGpw7yJzn0VeUpel6gBL+Vwm+BbmuorWeHUm9MXBTf35vIk1ePzyG2zPwr43km8PD+OW24lDUCH6+fZ5V+13nPyfJ7l3g+2xC8zF/VFL4K+lTf31fMDxo9h46Qs9AvbmV86/ZZ3nM3iPKQzvuKCY3mGYTa+GMIm6cl4XXzjQxXSeEqwACbvf+r/vslC39kGs0TczN8oKPjJe0lj7eWn2DT2GhflMWj5Y9wf8wC+2uWiM+7r0brVA8HD1N+0L2vYnxbgl/V/XCiqdu6cfH5y6uzAdM6i23e++FYU+cjzXWvH3eohKulcDUAW9B8TOe2hK8sfz90ztaNk6036z3+wmXY9Qmbr0C3h9pPEv6XAj/vE3dM7sk4QE6DTJC9+Mq8yve0snEyXVX3p2jeF9uy+XTW+T5gdIIYsR7hzTtvMY+2JHzrmrdOHNtDc99Cqe+Auro85M+ivGX7pGie+ZfmvSPEiUscp5rA6eqg64uicfPqH7c/Xzz+f8XkPVJ0/35fOrZHg+l/nqzvWXT0FVtsSfQ38qjqo8L3CwAA///tltFOwjAUhjsYiLIYxYhhV156qW+g8UYSjST4AoVVrJF2Kd0Fz8EzeO+liT6BiW/kha3ZslLALmTIJB05+Tlb+U7POe2KAwDYEyYVxN+l4SEcoAGjUXiMSYDCXXHvXFhVWE2YG4+vxP46OFcGzr7GkT7tPaI+V0Dx81XwTHnWNV59mvdvOVn7l/BMuul9yaqmeDtaPOmndc9ez4bGkT6N+BMmaKagRayniZP3+ixqHZbVouVh4th+5pvXquq5bs37vWe12PrX/X4ppfESrhpvW7m/7Hl+TWHQlntxlHJODRx3iuOCS0bDdfzO1I9DLV/ptwNEOL4fdxgeXkScDiHHfZCtv/P+Fya8LupTEkA2VvrRMfBaGq+l8O4Q41jgulGPoQGmROGa5tnUuNK/QXAUMXT7cwxcMAS7DzBEOfHahCMywnw8uw6rc3jqui3FfsPzvAPPrW0peZrWhQPUeTngDCwfvyIur+yUSwvqbOLI+cn9/uW/HcnPu//R+mxO/Imf8l6rv+/nZzC9n0/A4vHJZccXZ7xVq1atbqJ+A7xdhz4='
        data = base64.b64decode(b64data)
        pipeline = cpp.Pipeline()
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()),9)
        module = pipeline.modules()[7]
        self.assertTrue(isinstance(module, cpmoas.MeasureObjectAreaShape))
        self.assertEqual(len(module.object_groups), 3)
        for og,expected in zip(module.object_groups,
                               ["Cells","Nuclei","Cytoplasm"]):
            self.assertEqual(og[cpmoas.OG_NAME].value,expected)
        self.assertFalse(module.calculate_zernikes.value)

    def test_01_00_zeros(self):
        """Run on an empty labels matrix"""
        object_set = cpo.ObjectSet()
        labels = np.zeros((10,20),int)
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, "SomeObjects")
        module = cpmoas.MeasureObjectAreaShape()
        settings = ["SomeObjects","Yes"]
        module.set_settings_from_values(settings, 1, module.module_name)
        module.module_num = 1
        image_set_list = cpi.ImageSetList()
        measurements = cpmeas.Measurements()
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, 
                                  image_set_list.get_image_set(0),
                                  object_set, measurements, image_set_list)
        module.run(workspace)
        
        a = measurements.get_current_measurement('SomeObjects','AreaShape_Area')
        self.assertEqual(len(a), 0)

    def test_01_02_run(self):
        """Run with a rectangle, cross and circle"""
        object_set = cpo.ObjectSet()
        labels = np.zeros((10,20),int)
        labels[1:9,1:5] = 1
        labels[1:9,11] = 2
        labels[4,6:19] = 2
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, "SomeObjects")
        labels = np.zeros((115,115),int)
        x,y = np.mgrid[-50:51,-50:51]
        labels[x**2+y**2<=2500] = 1
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, "OtherObjects")
        module = cpmoas.MeasureObjectAreaShape()
        settings = ["SomeObjects","OtherObjects","Yes"]
        module.set_settings_from_values(settings, 1, module.module_name)
        module.module_num = 1
        image_set_list = cpi.ImageSetList()
        measurements = cpmeas.Measurements()
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, 
                                  image_set_list.get_image_set(0),
                                  object_set, measurements, image_set_list)
        module.run(workspace)
        self.features_and_columns_match(measurements, module)
        
        a = measurements.get_current_measurement('SomeObjects','AreaShape_Area')
        self.assertEqual(len(a),2)
        self.assertEqual(a[0],32)
        self.assertEqual(a[1],20)
        #
        # Mini-test of the form factor of a circle
        #
        ff = measurements.get_current_measurement('OtherObjects',
                                                  'AreaShape_FormFactor')
        self.assertEqual(len(ff),1)
        self.assertAlmostEqual(np.sqrt(ff[0]),1,1)
        for object_name, object_count in (('SomeObjects',2),
                                          ('OtherObjects',1)):
            for measurement in module.get_measurements(pipeline,object_name,
                                                       'AreaShape'):
                feature_name = 'AreaShape_%s'%(measurement)
                m = measurements.get_current_measurement(object_name,
                                                         feature_name)
                self.assertEqual(len(m),object_count)
    
    def test_02_01_categories(self):
        module = cpmoas.MeasureObjectAreaShape()
        settings = ["SomeObjects","OtherObjects","Yes"]
        module.set_settings_from_values(settings, 1, module.module_name)
        for object_name in settings[:-1]:
            categories = module.get_categories(None, object_name)
            self.assertEqual(len(categories),1)
            self.assertEqual(categories[0],"AreaShape")
        self.assertEqual(len(module.get_categories(None,"Bogus")),0)
    
    def test_02_02_measurements_zernike(self):
        module = cpmoas.MeasureObjectAreaShape()
        settings = ["SomeObjects","OtherObjects","Yes"]
        module.set_settings_from_values(settings, 1, module.module_name)
        for object_name in settings[:-1]:
            measurements = module.get_measurements(None,object_name,'AreaShape')
            for measurement in cpmoas.F_STANDARD:
                self.assertTrue(measurement in measurements)
            self.assertTrue('Zernike_3_1' in measurements)
    
    def test_02_03_measurements_no_zernike(self):
        module = cpmoas.MeasureObjectAreaShape()
        settings = ["SomeObjects","OtherObjects","No"]
        module.set_settings_from_values(settings, 1, module.module_name)
        for object_name in settings[:-1]:
            measurements = module.get_measurements(None,object_name,'AreaShape')
            for measurement in cpmoas.F_STANDARD:
                self.assertTrue(measurement in measurements)
            self.assertFalse('Zernike_3_1' in measurements)
            
    def features_and_columns_match(self, measurements, module):
        self.assertEqual(len(measurements.get_object_names()), 2)
        self.assertTrue('SomeObjects' in measurements.get_object_names())
        self.assertTrue('OtherObjects' in measurements.get_object_names())
        features = measurements.get_feature_names('SomeObjects')
        features += measurements.get_feature_names('OtherObjects')
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(features), len(columns))
        for column in columns:
            self.assertTrue(column[0] in ['SomeObjects', 'OtherObjects'])
            self.assertTrue(column[1] in features)
            self.assertTrue(column[2] == cpmeas.COLTYPE_FLOAT)
        
                

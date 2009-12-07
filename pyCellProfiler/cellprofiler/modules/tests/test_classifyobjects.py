'''test_classifyobjects - test the ClassifyObjects module

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
from StringIO import StringIO
import unittest
import zlib

import cellprofiler.workspace as cpw
import cellprofiler.cpgridinfo as cpg
import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.modules.classifyobjects as C

OBJECTS_NAME = "myobjects"
MEASUREMENT_NAME_1 = "Measurement1"
MEASUREMENT_NAME_2 = "Measurement2"
IMAGE_NAME = "image"

class TestClassifyObjects(unittest.TestCase):
    def test_01_01_load_matlab_classify_objects(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylRwSU1WMDBRMDSzMja3MjRQMDIwsFQgGTAwevryMzAwsDIz'
                'MFTMuRtxN++ygUjZ7m23GXKTRTOEFjyd1G146ePiWL5jLFNvWG3MbPI2L7RW'
                'vr4uKT1q5zPPCiG9CgG9Sp43prlB68/8zL6RdrKr7/Hnn3PO26kfj2RieGPD'
                'tEBtPvem6Qc3hDLvFHnpZefklHDraWgAY/bBH/43/Re0J85eIOxRc+Tgjt5j'
                's7cc7mP/+3F17bNfLLIH4ko25bDXre+1ufeqR6T5evsPmYnKf3gbT5rN+sY5'
                'N7FPkvcLc01tyR494Vu7vDf9j+Or1Mi0CUlZ737s6h9ZgXPrfR7mqhrVq744'
                '/iLG5rZ1e9+GqgCL5Vzptxs2L3iosfPpw3fRnqWCv0x995dt9OdzrDjhPOtr'
                'ub/i0o4c0Y37POuPL9i/4qX2Tk/WRTLdjgZxR+xCTGtVzd/0TjfreH5c1jf+'
                '5PsZrjM31PQ52dtJFLyS8rN9sPi1Y+Mdz7RPW06bPTjj++BMo77Rsc8x9q//'
                'iR11rmRazsMXsfF3wKwDmo1sO2KfHxCRXy3VN+n7OQFV5u7vC5tjVvdq8n/8'
                'tOl0S3H44cAi74L7Vf4R0pMOvZ05LTw+98bs6E35zX9+elbnrfe+I360Wc7+'
                'm9tjtyN79/3rnlE1lXMv1wbzt25Se2tZXgpxvD17Ol/Y784vldX2y/aUx292'
                'Pjlj1w+e3XYqYu8Ovtvx6tzl9Xsq+k4H33r/2qT+3o+mk90MZf8W21fdyTm2'
                'RXmd/S9j3nVrdNnvNLK4Fly6xuj5c2ap7cQr8yKvl0f+XHg+cd3vyVPFZdNz'
                '/72qrjPpTz37XNxz/WvD/fE257ZU9W/dr+Te4hD5uZrl/KSdcl+FP0y9Yid8'
                'b8t/XcV858+THqhu1V+5t/Cjr0Xl3Om3T93P/Wr/bt/5U/8jH5/7VRn81u/z'
                '/8afTQldeqqvMn/M9/7/z+7V95Ytz78oN+cZFyl2PL7OzOt+0lc8sNG62Zn/'
                '+6Kt6TJ72qXmTXo6t/23XFZsS+aTTYLHz6fy+p/czVKk91aa6ZN3qc/ZA/mR'
                'Xqu//jyq8/39kteh76eKf/kvp7bXy+Bf1d/fv/rV98tp/7VXuGlZVXX0T+Du'
                'dZ//M3WZT9YGAF3qrqE=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))    
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        self.assertEqual(module.contrast_choice, C.BY_SINGLE_MEASUREMENTS)
        self.assertEqual(len(module.single_measurements), 1)
        group = module.single_measurements[0]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement, "Intensity_IntegratedIntensity_OrigBlue_1")
        self.assertEqual(group.bin_choice, C.BC_EVEN)
        self.assertEqual(group.bin_count, 5)
        self.assertAlmostEqual(group.low_threshold.value, 0.2)
        self.assertAlmostEqual(group.high_threshold.value, 0.8)
        self.assertFalse(group.wants_custom_names)
        self.assertFalse(group.wants_images)
        
    def test_01_02_load_matlab_classify_by_two_measurements(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylRwSU1WMDBRMDS1MrawMjRTMDIwsFQgGTAwevryMzAw/GBi'
                'YKiYczfibt5lA5Gy3duk7Wef0hIQdz6tKp68pEFgeiqzo+q1sDWzHlzKvXst'
                'OCl8o3HUukqBI5PfFFbwvDHLDRZ+psKbae0dViJTP7/f3vr48ZvMDCZ8DBod'
                'z/tWRBpoVp4/92PVlvOCz2d8iZB7ciB5/d+oK9wvEm5susC6JebIgTU+t1cv'
                '8Z69+OCbq5Jvd4itcrx9P3LKrOztEw6/uqt5ofruAeeUuY/fc+hMy9x+/gWX'
                'sU9n+/cfV6/d3DFF+/G7zl+bLa4rLn4xQXjrk+zpf7UT7tmLL4i+eH3/i6tq'
                'Up/S37dvOMsr/aks/tOeix/eaNhcWme9o1xpiR9zfMCH1o9WW3Vb1K2OSP7f'
                'ws/peK3b5Nj1h9pSf2RZ9d9qPewwOOMeVy31ekL4h0t6s93X+wjkyZWL7tly'
                'P7A5IG8Hz9nnHwMPBwdVeM2Ts2UvuhIZoua534f3/DWm82I/tPZ9VDkeX1Z7'
                '+ZLi7gOTWU1nvbNKEaswWsCTd93uwseWXvHMn8X+DyeWzTW+/iPhotiWoO/p'
                '4Q8nfpNsOrL+otWb66HzT2eejf8bd2bxp6BdS1urJYWmRx942lCZM+3xgydy'
                'sWzhFx9c6rPPPB8/Pdwm4M80MzXZmlZrvS08OYfv6WTfP2Kx59VnW4/+7XrL'
                'Hyk13ZZe0y4l3ufcJx/G/1WyPfzJMeVrbWKfPh45mxxSHzaX3/Xr8/rny0+/'
                'cp532OV6emRsyDz92+b/l384flHjt5nH/cILby2vyC45Lzxl+6VrqlmfLSfW'
                '1xWeE5efr3hOyqRjQXjI/5Iax//dz3b53uk+JH51/4p/bU+vf1TZGp+3um57'
                '6b/XYvmLXm1/lR70+aZJbElhXfzH8FtXnoqpvVjvIH/k/vk/v2tC35sXfn+u'
                'zmUder9V7cweOS1gDJUJPNxhnWVx49vzdYk/5DRrLomVmz6zWmH370ztdNWT'
                '1y98kOSav8Vf98SMD/lOJ/zczuZfkT4sNs/YVF8r48s5ruz9W/bOvlLfZvD4'
                '6fv0za/+y8jqPd+63+5P4Y/XJ782HvnPdvHb1cvSv/xP7/r2n7ns6/KJAHF9'
                'smE=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()),4)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        self.assertEqual(module.contrast_choice, C.BY_TWO_MEASUREMENTS)
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.first_measurement, "Intensity_IntegratedIntensity_OrigBlue_1")
        self.assertEqual(module.second_measurement, "Intensity_MeanIntensity_OrigBlue_1")
        self.assertEqual(module.first_threshold_method, C.TM_MEAN)
        self.assertEqual(module.second_threshold_method, C.TM_MEAN)
        self.assertFalse(module.wants_custom_names)
        self.assertFalse(module.wants_image)
    
    def test_01_03_load_v1(self):
        data = ('eJztXNtu2zYYphwna1pgSLeLFRiG6SoYCkdQTluWmymHZjEQJ8EcdO1NM0Wi'
                'bQ4yaUhUEu9J9lh7lD7CRFu2ZEaObFmWLJdCFJs0v/8jP/48/JKg2tHNxdGx'
                'vK+ocu3oZquBLChfWzptELt9KGNakU9sqFNoygQfyjWC5VNoyOovsnpwuPvz'
                'obov76jqryDZIVVrX3sf/34CYM37fOGdJf+nVT8thU6WrkNKEW46q6AM3vj5'
                'n73zvW4j/c6C73XLhU5AMciv4ga56XaGP9WI6VrwUm+HC3vHpdu+g7Zz1RgA'
                '/Z+v0SO06ugfyDVhUOwPeI8cRLCP9+3zuUNeQjneeos8nNledTj7xzo1WnXq'
                '9cBofk+3SqCbxOnGPjdD+az8OQjKlyN0fh0qv+GnETbRPTJd3ZJRW28Oa83s'
                'HcTYe8HZY+krGzWPvS5KAx/Xnm85PEvXPefxXLwNdce1YRti6gzbo8XY2+Ds'
                'sfPmgYwYS2wHPtKtd4+6QeU26/JJ9Fnj7LD0pWtYEE3WnpccnqVPiYwJlV3H'
                '9zdmR42xI43YkcBOQtx2QtwumMwfV7n2svS2WtlTJ8TH6T0rPq7d5RF8GVwS'
                'DLPs55UROyvgozcb8Lg1Djc4Brh1EPB9iOELz18bfrqKKcQOot1b9q1ps6Up'
                'yBvMDrPpGYcrjeBKHm4yvklw0+gXN/99w+nH0u/uIba6stPRDW9Nv0PYyc/e'
                'd5w9lj6FDd21qFxli418imxoUGJ3c/HPaeeh/YS43YQ4BUw2z7/idGbpK+q4'
                '8u8WudOtyPamqRPfL6qyk6i9g3l6XuNzkvVovroczHXdTKrLk3pWtrPVJYIv'
                'TVxSXfj1Q1XU7Sx1iZrfslh3tBi+9RG+fnq4Ro9p7zT81zH8P3L8LH2GbIdW'
                '6tAg2KzctJBtVs6Ia9NW5Qw1aGum/UKSOOakpWMMrZ1F0DNrv5s17puWd1ud'
                'r79HxXMnlu44qIGg2d/hp2lnVn9LY18zjX5Fjauy3m8mmdeC2KemP0YEQkmv'
                '+1wQQ6eI4NsT6Bm1bz+E6jkvPSfBpRkfRV0fOnEdStpbJmwgHA5okvphDZpI'
                'x8Xyw3nFL0/3u/u5r1vz3c+rqY2/j2PqsUx6bft6pTnOsujfea2LUeu4slNR'
                '9ivKQR7jTYvhy/s6w6Jcl1s0vqRxy1dgtD9Z2iIPt965VPNQmnEK06eFmq0s'
                '6+sN64WIb5LoxbRiDpX1vMTu2xmsDTDY3+UV38TVPyr+7+nG/oXqMa99ctT+'
                '5E/ocbPHAu7ZDXBswBTqkef9mjT9Lep+2xmxYdMmLjaz16kGdZxH/LXo8/ys'
                '82ae95WyHldZ67II/TDJOCpK+5Z9HC2iLmnu6xcZl2T/c0EevL9itXNe+kTF'
                'Wefe9uoixThrkXFJ9PG0YRIVqp1ZXidi4pynGIcWBSfm5QCnged1mTSuT8r/'
                '39sAJ3E49rkZys86Hus9zMwCsk76doriH+cx7Y2K+8nd39CgvQbLCJuwk2P9'
                'i4LTwPM6R12fCum8sHaKor/ACZzACZzACZzACZzAPY/TQrhJ48YgDuqHBUWu'
                'd1HbvyzxSVF0KwpOA1+WPwucwAmcwC0aTgvhvoTrywIncMuI00I4sZ8aj9PA'
                '8zqJeEDg5oHTgBifAidwAidwAidwAidwAidwy4LTQrg89vetUoCTOJzkf5dC'
                '5f+Kqe9brr4sbUDL6tiEvY/ZVtq9lwY7ikV0s/8WXuXC+1oNvZCX8XRieDSO'
                'RxvHg0yIKWp0O7bH5lLS1ikylKqfe+3lHg1yGe9jDO8xx3s8jtd/kW7/GgAa'
                'vMxGqfWzr3rZ3Iuvev0Rw7/H8e+N4zf6Dz93+xVwFP9h6G6f2cmLL/z8/3oE'
                'X9jfSn769Q8rm9+D58cZAKP+Hfj959+S8pbLJUkCT8fpqxg80/EleHowO2+k'
                '6cbbT2B8+UGbl6l8kn6S2AFm1zfgKw/rNuBZhvL/A/z19oA=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()),5)
        module = pipeline.modules()[-2]
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        self.assertEqual(module.contrast_choice, C.BY_SINGLE_MEASUREMENTS)
        self.assertEqual(len(module.single_measurements), 2)
        
        group = module.single_measurements[0]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement.value, "Intensity_IntegratedIntensity_OrigBlue")
        self.assertEqual(group.bin_choice, C.BC_EVEN)
        self.assertEqual(group.bin_count, 5)
        self.assertAlmostEqual(group.low_threshold.value, 0.2)
        self.assertAlmostEqual(group.high_threshold.value, 0.8)
        self.assertTrue(group.wants_custom_names)
        for name, expected in zip(group.bin_names.value.split(','),
                                  ('First','Second','Third','Fourth','Fifth')):
            self.assertEqual(name, expected)
        self.assertTrue(group.wants_images)
        self.assertEqual(group.image_name, "ClassifiedNuclei")
        
        group = module.single_measurements[1]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement.value, "Intensity_MaxIntensity_OrigBlue")
        self.assertEqual(group.bin_choice, C.BC_CUSTOM)
        self.assertEqual(group.custom_thresholds, ".2,.5,.8")
        self.assertFalse(group.wants_custom_names)
        self.assertFalse(group.wants_images)
        
    def make_workspace(self, labels, contrast_choice,
                       measurement1=None, measurement2=None):
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)
        
        measurements = cpmeas.Measurements()
        module = C.ClassifyObjects()
        m_names = []
        if measurement1 is not None:
            measurements.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME_1,
                                         measurement1)
            m_names.append(MEASUREMENT_NAME_1)
        if measurement2 is not None:
            measurements.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME_2,
                                         measurement2)
            module.add_single_measurement()
            m_names.append(MEASUREMENT_NAME_2)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        
        module.contrast_choice.value = contrast_choice
        if module.contrast_choice == C.BY_SINGLE_MEASUREMENTS:
            for i, m in enumerate(m_names):
                group = module.single_measurements[i]
                group.object_name.value = OBJECTS_NAME
                group.measurement.value = m
                group.image_name.value = IMAGE_NAME
        else:
            module.object_name.value = OBJECTS_NAME
            module.image_name.value = IMAGE_NAME
            module.first_measurement.value = MEASUREMENT_NAME_1
            module.second_measurement.value = MEASUREMENT_NAME_2
        module.module_num = 1
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, measurements,
                                  image_set_list)
        return workspace, module
        
    def test_02_01_classify_single_none(self):
        '''Make sure the single measurement mode can handle no objects'''
        workspace, module = self.make_workspace(
            np.zeros((10,10),int),
            C.BY_SINGLE_MEASUREMENTS,
            np.zeros((0,), float))
        module.run(workspace)
        for m_name in ("Classify_Measurement1_Bin_1",
                       "Classify_Measurement1_Bin_2",
                        "Classify_Measurement1_Bin_3"):

            m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                               m_name)
            self.assertEqual(len(m), 0)
    
    def test_02_02_classify_single_even(self):
        m = np.array((.5,0,1,.1))
        labels = np.zeros((20,10),int)
        labels[2:5,3:7] = 1
        labels[12:15,1:4] = 2
        labels[6:11,5:9] = 3
        labels[16:19,5:9] = 4
        workspace, module = self.make_workspace(labels,
                                                C.BY_SINGLE_MEASUREMENTS, m)
        module.single_measurements[0].bin_choice.value = C.BC_EVEN
        module.single_measurements[0].low_threshold.value = .2
        module.single_measurements[0].high_threshold.value = .7
        module.single_measurements[0].bin_count.value = 3
        module.single_measurements[0].wants_images.value = True
        
        expected = dict(Classify_Measurement1_Bin_1 = (0,1,0,1),
                        Classify_Measurement1_Bin_2 = (1,0,0,0),
                        Classify_Measurement1_Bin_3 = (0,0,1,0))
        module.run(workspace)
        for measurement, expected_values in expected.iteritems():
            values = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                    measurement)
            self.assertEqual(len(values), 4)
            self.assertTrue(np.all(values == np.array(expected_values)))
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertTrue(np.all(pixel_data[labels==0,:] == 0))
        colors = [pixel_data[x,y,:] for x,y in ((2,3),(12,1),(6,5))]
        for i,color in enumerate(colors + [colors[1]]):
            self.assertTrue(np.all(pixel_data[labels==i+1,:] == color))
            
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 3)
        self.assertEqual(len(set([column[1] for column in columns])), 3) # no duplicates
        for column in columns:
            self.assertEqual(column[0], OBJECTS_NAME)
            self.assertTrue(column[1] in expected.keys())
            self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER)
            
        categories = module.get_categories(None, cpmeas.IMAGE)
        self.assertEqual(len(categories), 0)
        categories = module.get_categories(None, OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], C.M_CATEGORY)
        names = module.get_measurements(None, OBJECTS_NAME, "foo")
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, "foo", C.M_CATEGORY)
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, OBJECTS_NAME, C.M_CATEGORY)
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(['_'.join((C.M_CATEGORY, name)) in expected.keys()
                             for name in names]))
        
    def test_02_03_classify_single_custom(self):
        m = np.array((.5,0,1,.1))
        labels = np.zeros((20,10),int)
        labels[2:5,3:7] = 1
        labels[12:15,1:4] = 2
        labels[6:11,5:9] = 3
        labels[16:19,5:9] = 4
        workspace, module = self.make_workspace(labels,
                                                C.BY_SINGLE_MEASUREMENTS, m)
        module.single_measurements[0].bin_choice.value = C.BC_CUSTOM
        module.single_measurements[0].custom_thresholds.value = ".2,.7"
        module.single_measurements[0].bin_count.value = 14 # should ignore
        module.single_measurements[0].wants_custom_names.value = True
        module.single_measurements[0].bin_names.value = "Three,Blind,Mice"
        module.single_measurements[0].wants_images.value = True
        
        expected = dict(Classify_Three = (0,1,0,1),
                        Classify_Blind = (1,0,0,0),
                        Classify_Mice = (0,0,1,0))
        module.run(workspace)
        for measurement, expected_values in expected.iteritems():
            values = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                    measurement)
            self.assertEqual(len(values), 4)
            self.assertTrue(np.all(values == np.array(expected_values)))
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertTrue(np.all(pixel_data[labels==0,:] == 0))
        colors = [pixel_data[x,y,:] for x,y in ((2,3),(12,1),(6,5))]
        for i,color in enumerate(colors + [colors[1]]):
            self.assertTrue(np.all(pixel_data[labels==i+1,:] == color))
            
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 3)
        self.assertEqual(len(set([column[1] for column in columns])), 3) # no duplicates
        for column in columns:
            self.assertEqual(column[0], OBJECTS_NAME)
            self.assertTrue(column[1] in expected.keys())
            self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER)
            
        categories = module.get_categories(None, cpmeas.IMAGE)
        self.assertEqual(len(categories), 0)
        categories = module.get_categories(None, OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], C.M_CATEGORY)
        names = module.get_measurements(None, OBJECTS_NAME, "foo")
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, "foo", C.M_CATEGORY)
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, OBJECTS_NAME, C.M_CATEGORY)
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(['_'.join((C.M_CATEGORY, name)) in expected.keys()
                             for name in names]))
        
    def test_03_01_two_none(self):
        workspace, module = self.make_workspace(
            np.zeros((10,10),int),
            C.BY_TWO_MEASUREMENTS,
            np.zeros((0,), float),np.zeros((0,), float))
        module.run(workspace)
        for lh1 in ("low","high"):
            for lh2 in ("low","high"):
                m_name = ("Classify_Measurement1_%s_Measurement2_%s" %
                          (lh1,lh2))
                m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                   m_name)
                self.assertEqual(len(m), 0)
        
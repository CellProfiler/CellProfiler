'''test_measureobjectradialdistribution.py

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import base64
import numpy as np
from StringIO import StringIO
import unittest
import zlib

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
import cellprofiler.modules.measureobjectradialdistribution as M

OBJECT_NAME = 'objectname'
CENTER_NAME = 'centername'
IMAGE_NAME = 'imagename'

def feature_frac_at_d(bin, bin_count):
    return M.M_CATEGORY + "_"+M.FF_FRAC_AT_D % (IMAGE_NAME, bin, bin_count)
def feature_mean_frac(bin, bin_count):
    return M.M_CATEGORY + "_"+M.FF_MEAN_FRAC % (IMAGE_NAME, bin, bin_count)
def feature_radial_cv(bin, bin_count):
    return M.M_CATEGORY + "_"+M.FF_RADIAL_CV % (IMAGE_NAME, bin, bin_count)

class TestMeasureObjectRadialDistribution(unittest.TestCase):
    def test_01_01_load_matlab(self):
        data = ('eJwBhAR7+01BVExBQiA1LjAgTUFULWZpbGUsIFBsYXRmb3JtOiBQQ1dJTiwg'
                'Q3JlYXRlZCBvbjogVHVlIEF1ZyAxOCAxNTozODowMiAyMDA5ICAgICAgICAg'
                'ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAAAUlN'
                'DwAAAPwDAAB4nOxZ0W6bMBR10iTqNqnKtJftjQ9oO9g6qY+tkmnrQ9OqrSrt'
                '0SEu9WRsFEyV7Kv2Cfu0QQIELFKDQyDrQLIsG59zj6+vr2U4AACc6gD0/Hrf'
                'L22wfLphu5UoQfsWcY6p5XZBB7wP+//45R5OMRwTdA+Jh1wQP1H/BX1gd3Mn'
                'fnXJJh5BI2gnB/vPyLPHaOpePUTA8PU1niFyi38hkH6iYTfoCbuY0RAf8ou9'
                'sV3GBbsHfvn9YeWHluCHoLxL9Afjz8BqfCfDb/3E+H5Y7tCMH32dQZNrNuTm'
                'Y8CjS3j2Ujx7YDg6X9g/leB6gv3ewr8mQXipX9WubN6vBLtBezDnzCHQtRP+'
                'q9v+pv6T6Xgj4IP24BFSiohxpBul6ZDhuwI+aA8QIW5F9uvWv238mQRfVjxm'
                '6TD0wxM9p46seLyeMgdakPtJMtYh43kt8ATtIdMo45rnIlA6T93xWZU/nskX'
                'n5L5omje/OGfdVXm7VaKpwVO/jGczE9511v1fJPh2ilcG4yYenxdcdfTvhE2'
                'hiTWHdVl+aHhKTdOov1cNL4N0KzvLvLk3ZdF40Q/NJr1roFHdV/KcJ0UrgP0'
                'Y91o1nf31nfdPix6rht6Ol8XjavjF477nBO3bt8o4L78D/utaHxf3gwrnUfW'
                'PeWCckRdzOegfH/krb9L7L0V7AVtTCf4CU88SDRsQyv+KlmmH1TvE3nnva37'
                'Stb8zj3ObMixWUBfUb3r8lNdelXsWlM4d01IUIJH9Z5RVO+2vk+UvV/z6lWN'
                '37r0iv6lH2Gl659X56Z5J6r3+8//t0nm3U3P70WStqbMc/LzZN3v2PgnMvmK'
                'qEo9L5Unr5+jWuW8TvBp/tmNnC3ylTXfhmcznqz/uKv4XC6byrm9C/P8CwAA'
                '//9jZWBg4ABiRiDmhtIgIADl5ydlpSaXpBfllxaAxfmA2AGI2aD6WKDqB9Ic'
                'dJqQuVxo5oL4mbmJ6akIY0kyb6iGG73MEUAzRwAlvBUy81JSC7DF41D170Cn'
                '/9HwG02/Q8kcQvRgc+9QM4fS9Etv2oOAf4TQ/APi55eW5GTmIfto4P0xUumB'
                'jr8dTAj7YfYg248sTm570Sc/McUTlIeKife3KJo5IL5nSmpeSWZaZUBRZq5j'
                'aUl+bmJJZjKR5gmimSeIZF5wanJ+XkpiUSWSPwMImCeJZh6I75uaWFxalBqU'
                'mJKZmOOSWVxSlJlUWpKZnzdqLoa5DgTM5UUzF8R3rSjILyoJyXetSE7NgZpj'
                'gWQOGxZzkNMvE5QvJMzDAgRcIP0GBNzByIDsDkYGQwrsZeVhZgQCZnT/E9LP'
                'Ag2DbbLpspdlveUg8K4MyJwVjKTlXw0G3OphYFQ9bdQDAIDboH02qiiq')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 8)
        for i, image_name, object_name, center_name in \
            ((3, "DNA", "Nuclei", None),
             (4, "Cytoplasm", "Nuclei", None),
             (5, "DNA", "Cells", "Nuclei"),
             (6, "Cytoplasm","Cells","Nuclei")):
            module = pipeline.modules()[i]
            self.assertTrue(isinstance(module, M.MeasureObjectRadialDistribution))
            for seq in (module.images, module.objects, module.bin_counts):
                self.assertEqual(len(seq),1)
            self.assertEqual(module.images[0].image_name, image_name)
            self.assertEqual(module.objects[0].object_name,  object_name)
            if center_name is None:
                self.assertEqual(module.objects[0].center_choice, M.C_SELF)
            else:
                self.assertEqual(module.objects[0].center_choice, M.C_OTHER)
                self.assertEqual(module.objects[0].center_object_name, center_name)
            self.assertEqual(module.bin_counts[0].bin_count, 4)

    def test_01_02_load_v1(self):
        data = ('eJztW92O2kYUHgi7yjZttOlNqqiR5jLbLsiQXXWzijZQ6A9qYNEuShRFaTvg'
                'ASYae5A93iytIvWyj9HH6GUfp5d9hHrABjPxYmPMAhsjWXCO55vvzJkz5wwD'
                'rpWaz0vfwsOcAmulZrZDKIYNiniHGdox1Pk+LBsYcaxCph/DpoVhyerC/BHM'
                'K8ePj44LB7CgKE9AtFeqWrtrvzWeArBtv9+2r7Rza8uRU55LyOeYc6J3zS2Q'
                'AV84+n/s6wUyCGpR/AJRC5sTCldf1TusOeiPb9WYalFcR5q3sf2qW1oLG+Zp'
                'xwU6txvkEtNz8huWhuA2O8MXxCRMd/BO/7J2zMu4xCv88NeXEz+kJD9k7Ouh'
                'Ry/a/wgm7TM+frvnab/ryERXyQVRLUQh0VB3bIXoTwno79ZUf7dApV4a4o4C'
                'cNuSHdtDP7cpJuF4U1P4FCg49hYDcLsSr7ia+JJnv7tEbQ41xNu9OOwPwm9J'
                'eCGXMaVmSL9fNf55cY9BdHvzyv6BEtLvdyS8kBsG66Mu4vZiGOqj2J8Pyf+J'
                'xC/kCoM649AynQUcJd5f2aslDP+OxC/k8oCzPkWmBqLzu+stCJeewqVBnYXj'
                'uwoXNF6/+T7lpgV/oKyF6Hi8cfktKO/dl/oRcgV3kEU5rIqkByvEwG3OjMFC'
                'cTAvLp9TYs+X2xLefbn4HY/figG8YecxSt1RcsrwtZ93Psxh12dSf0Ju9rCJ'
                'IWu9tecxbB6Nex6DcJkpXEb4IB8FV2c6XmT9xj1f89qRV66nTkbNe3HNk407'
                'jBrPp7yHjRnxHOc698uzVZ1j3SR8MEc/YevsovkuynjKPaTrmBayS/DLvPlC'
                'CZkv5P3OQUS+qPsEd53GkZ/mideo+8B5cd+ExIXJI3GOL2q+j7o/9avzzXcM'
                'tu19lul8I1xkvH8E8P8k8Qv550fPGk/FwQM+yX2994uQXtqp/4y9O3ldyjbe'
                '7LmaMqOWpp+8VrJP3vye3y+8HzU+JzZyqNwb2xHkh7D5K0r9fIlJtyeOTS7E'
                'AYHeds8NFvFrL8COI8kOIQvfvMLIcBx28H4vK1Q1pvOeoys4ugoaTDTLjD8/'
                'v3/PDNw1mKWri/spiH9Gvch760Uc+8moeWLp443h++iqxrus85IwdXsdxhe1'
                'Xiyzbm9ifVjlPuAm5v1VrpM419eHeeBw5XYu+5wnzv3fdePWZX+2bvO87H3W'
                'uq/bm5aX5H3N4Yrs/PvBBJeScH6/U15nfA9/1BQB3g/fj18+HJ3ETTralLzm'
                'sRsSXcX9Jfa3ievqJuKK4HrWSdh+Pja/3ZTxrmseTHDJfH6M490UXNC+4nMw'
                'PS9CZhanRMcfbCySeQ6PK4LZftoF034S16R+jbye1M8Et2m4uPPNpow7wa0H'
                'rghmx9+y826CWw9cEcyOg6T+JbgEl+ASXIJLcOuDu52e4FISTsje/8mI9r96'
                'ePzq/Fee9ruO3MaU9g0mnqc0ctrwoT8zRxlSR0/d5Z7bH6ueB/AETz+Apyjx'
                'FK/iISrWOekM+obNZnGmIU7auaqjbdjakqsVvL0AXr//l8zkNXGb6SoyBmPO'
                'c1cj+P4M4GtKfM2r+DSMTMvAo7Mfw/YtoioxuUFalnjQLFcb3T8d3j8b3q94'
                '7rt+98bRjo893nhI29L9h/c+nRV/AEzH3SQe/3sWhS+TSafugun/0d8JwGXA'
                '9DoQ+H/BfHH/aEZ7d4zr2v5/jp9zeQ==')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))            
        self.assertEqual(len(pipeline.modules()),4)
        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module, M.MeasureObjectRadialDistribution))
        self.assertEqual(len(module.images), 2)
        self.assertEqual(len(module.objects), 2)
        self.assertEqual(len(module.bin_counts), 1)
        for image, name in zip(module.images,("DNA", "Cytoplasm")):
            self.assertEqual(image.image_name, name)
        for o, name, center_name in ((module.objects[0], "Nuclei",None),
                                     (module.objects[1], "Cells","Nuclei")):
            self.assertEqual(o.object_name, name)
            if center_name is None:
                self.assertEqual(o.center_choice, M.C_SELF)
            else:
                self.assertEqual(o.center_choice, M.C_OTHER)
                self.assertEqual(o.center_object_name, center_name)
        self.assertEqual(module.bin_counts[0].bin_count, 4)
    
    def test_02_01_get_measurement_columns(self):
        module = M.MeasureObjectRadialDistribution()
        for i,image_name in ((0, "DNA"),(1, "Cytoplasm"),(2,"Actin")):
            if i:
                module.add_image()
            module.images[i].image_name.value = image_name
        for i,object_name, center_name in ((0, "Nucleii", None),
                                           (1, "Cells", "Nucleii"),
                                           (2, "Cytoplasm", "Nucleii")):
            if i:
                module.add_object()
            module.objects[i].object_name.value = object_name
            if center_name is None:
                module.objects[i].center_choice.value = M.C_SELF
            else:
                module.objects[i].center_choice.value = M.C_OTHER
                module.objects[i].center_object_name.value = center_name
        for i,bin_count in ((0,4),(0,5),(0,6)):
            if i:
                module.add_bin_count()
            module.bin_counts[i].bin_count.value = bin_count
        
        columns = module.get_measurement_columns(None)
        column_dictionary = {}
        for object_name, feature, coltype in columns:
            key = (object_name, feature)
            self.assertFalse(column_dictionary.has_key(key))
            self.assertEqual(coltype, cpmeas.COLTYPE_FLOAT)
            column_dictionary[key] = (object_name, feature, coltype)
        
        for object_name in [x.object_name.value for x in module.objects]:
            for image_name in [x.image_name.value for x in module.images]:
                for bin_count in [x.bin_count.value for x in module.bin_counts]:
                    for bin in range(1,bin_count+1):
                        for feature in M.F_ALL:
                            measurement = "_".join((M.M_CATEGORY,
                                                    feature,
                                                    image_name,
                                                    "%dof%d"%(bin, bin_count)))
                            key = (object_name, measurement)
                            self.assertTrue(column_dictionary.has_key(key))
                            del column_dictionary[key]
        self.assertEqual(len(column_dictionary), 0)
    
    def test_02_02_get_measurements(self):
        module = M.MeasureObjectRadialDistribution()
        for i,image_name in ((0, "DNA"),(1, "Cytoplasm"),(2,"Actin")):
            if i:
                module.add_image()
            module.images[i].image_name.value = image_name
        for i,object_name, center_name in ((0, "Nucleii", None),
                                           (1, "Cells", "Nucleii"),
                                           (2, "Cytoplasm", "Nucleii")):
            if i:
                module.add_object()
            module.objects[i].object_name.value = object_name
            if center_name is None:
                module.objects[i].center_choice.value = M.C_SELF
            else:
                module.objects[i].center_choice.value = M.C_OTHER
                module.objects[i].center_object_name.value = center_name
        for i,bin_count in ((0,4),(0,5),(0,6)):
            if i:
                module.add_bin_count()
            module.bin_counts[i].bin_count.value = bin_count
        
        for object_name in [x.object_name.value for x in module.objects]:
            self.assertEqual(tuple(module.get_categories(None, object_name)),
                             (M.M_CATEGORY,))
            for feature in M.F_ALL:
                self.assertTrue(feature in module.get_measurements(
                    None, object_name, M.M_CATEGORY))
            for image_name in [x.image_name.value for x in module.images]:
                for feature in M.F_ALL:
                    self.assertTrue(image_name in 
                                    module.get_measurement_images(None, 
                                                                  object_name, 
                                                                  M.M_CATEGORY, 
                                                                  feature))
                for bin_count in [x.bin_count.value for x in module.bin_counts]:
                    for bin in range(1,bin_count+1):
                        for feature in M.F_ALL:
                            self.assertTrue("%dof%d"%(bin, bin_count) in
                                            module.get_measurement_scales(
                                                None, object_name,
                                                M.M_CATEGORY, feature,
                                                image_name))

    def run_module(self, image, labels, center_labels = None, bin_count = 4):
        '''Run the module, returning the measurements
        
        image - matrix representing the image to be analyzed
        labels - labels matrix of objects to be analyzed
        center_labels - labels matrix of alternate centers or None for self
                        centers
        bin_count - # of radial bins
        '''
        module = M.MeasureObjectRadialDistribution()
        module.images[0].image_name.value = IMAGE_NAME
        module.objects[0].object_name.value = OBJECT_NAME
        object_set = cpo.ObjectSet()
        main_objects = cpo.Objects()
        main_objects.segmented = labels
        object_set.add_objects(main_objects, OBJECT_NAME)
        if center_labels is None:
            module.objects[0].center_choice.value = M.C_SELF
        else:
            module.objects[0].center_choice.value = M.C_OTHER
            module.objects[0].center_object_name.value = CENTER_NAME
            center_objects = cpo.Objects()
            center_objects.segmented = center_labels
            object_set.add_objects(center_objects, CENTER_NAME)
        module.bin_counts[0].bin_count.value = bin_count
        pipeline = cpp.Pipeline()
        measurements = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        img = cpi.Image(image)
        image_set.add(IMAGE_NAME, img)
        workspace = cpw.Workspace(pipeline, module, image_set, object_set,
                                  measurements, image_set_list)
        module.run(workspace)
        return measurements
        
    def test_03_01_zeros_self(self):
        '''Test the module on an empty labels matrix, self-labeled'''
        m = self.run_module(np.zeros((10,10)), np.zeros((10,10),int))
        for bin in range(1,5):
            for feature in (feature_frac_at_d(bin, 4),
                            feature_mean_frac(bin, 4),
                            feature_radial_cv(bin, 4)):
                self.assertTrue(feature in m.get_feature_names(OBJECT_NAME))
                data = m.get_current_measurement(OBJECT_NAME, feature)
                self.assertEqual(len(data), 0)
    
    def test_03_02_circle(self):
        '''Test the module on a uniform circle'''
        i,j = np.mgrid[-50:51,-50:51]
        labels = (np.sqrt(i*i+j*j) <= 40).astype(int)
        m = self.run_module(np.ones(labels.shape), labels)
        for bin in range(1,5):
            data = m.get_current_measurement(OBJECT_NAME, 
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 1)
            area = (float(bin) * 2.0 - 1.0)/16.0
            self.assertTrue(data[0] > area - .1)
            self.assertTrue(data[0] < area + .1)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_mean_frac(bin, 4))
            self.assertEqual(len(data), 1)
            self.assertAlmostEqual(data[0], 1, 2)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_radial_cv(bin, 4))
            self.assertEqual(len(data), 1)
            self.assertAlmostEqual(data[0], 0, 2)
  
    def test_03_03_half_circle(self):
        '''Test the module on a circle and an image that's 1/2 zeros
        
        The measurements here are somewhat considerably off because
        the propagate function uses a Manhattan distance with jaywalking
        allowed instead of the Euclidean distance.
        '''
        i,j = np.mgrid[-50:51,-50:51]
        labels = (np.sqrt(i*i+j*j) <= 40).astype(int)
        image = np.zeros(labels.shape)
        image[i>0] = (np.sqrt(i*i+j*j) / 100)[i>0]
        image[j==0] = 0
        image[i==j] = 0
        image[i==-j] = 0
        # 1/2 of the octants should be pretty much all zero and 1/2
        # should be all one
        x=[0,0,0,0,1,1,1,1]
        expected_cv = np.std(x)/np.mean(x)
        m = self.run_module(image, labels)
        bin_labels = (np.sqrt(i*i+j*j)*4 / 40.001).astype(int)
        mask = i*i+j*j <=40*40
        total_intensity = np.sum(image[mask])
        for bin in range(1,5):
            data = m.get_current_measurement(OBJECT_NAME, 
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 1)
            bin_count = np.sum(bin_labels[mask] == bin-1)
            frac_in_bin = float(bin_count) / np.sum(mask)
            bin_intensity = np.sum(image[mask & (bin_labels == bin-1)])
            expected = bin_intensity / total_intensity
            self.assertTrue(np.abs(expected-data[0]) < .2 * expected)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_mean_frac(bin, 4))
            self.assertEqual(len(data), 1)
            expected = expected / frac_in_bin
            self.assertTrue(np.abs(data[0]-expected) < .2 * expected)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_radial_cv(bin, 4))
            self.assertEqual(len(data), 1)
            self.assertTrue(np.abs(data[0] - expected_cv) < .2*expected_cv)
    
    def test_03_04_line(self):
        '''Test the alternate centers with a line'''
        labels = np.array([[0,0,0,0,0,0,0,0,0,0],
                           [0,1,1,1,1,1,1,1,1,0],
                           [0,1,1,1,1,1,1,1,1,0],
                           [0,1,1,1,1,1,1,1,1,0],
                           [0,0,0,0,0,0,0,0,0,0]])
        centers = np.zeros(labels.shape, int)
        centers[2,1] = 1
        distance_to_center = np.array([[0,0,0,0,0,0,0,0,0,0],
                                       [0,1,1.4,2.4,3.4,4.4,5.4,6.4,7.4,0],
                                       [0,0,1  ,2  ,3,  4,  5  ,6  ,7  ,0],
                                       [0,1,1.4,2.4,3.4,4.4,5.4,6.4,7.4,0],
                                       [0,0,0,0,0,0,0,0,0,0]])
        distance_to_edge = np.array([[0,0,0,0,0,0,0,0,0,0],
                                     [0,1,1,1,1,1,1,1,1,0],
                                     [0,1,2,2,2,2,2,2,2,0],
                                     [0,1,1,1,1,1,1,1,1,0],
                                     [0,0,0,0,0,0,0,0,0,0]])
        np.random.seed(0)
        image = np.random.uniform(size=labels.shape)
        m = self.run_module(image, labels, centers)
        total_intensity = np.sum(image[labels==1])
        normalized_distance = distance_to_center / (distance_to_center + distance_to_edge + .001)
        bin_labels = (normalized_distance * 4).astype(int)
        for bin in range(1,5):
            data = m.get_current_measurement(OBJECT_NAME, 
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 1)
            bin_intensity = np.sum(image[(labels==1) & 
                                         (bin_labels == bin-1)])
            expected = bin_intensity / total_intensity
            self.assertTrue(np.abs(expected-data[0]) < .1 * expected)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_mean_frac(bin, 4))
            expected = expected * np.sum(labels==1) / np.sum((labels == 1) &
                                                             (bin_labels == bin-1))
            self.assertTrue(np.abs(data[0]-expected) < .1 * expected)
            data = m.get_current_measurement(OBJECT_NAME,
                                             feature_radial_cv(bin, 4))
            self.assertEqual(len(data), 1)
            
    def test_04_01_img_607(self):
        '''Regression test of bug IMG-607
        
        MeasureObjectRadialDistribution fails if there are no pixels for
        some of the objects.
        '''
        np.random.seed(41)
        labels = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                           [ 0, 1, 1, 1, 0, 0, 3, 3, 3, 0 ],
                           [ 0, 1, 1, 1, 0, 0, 3, 3, 3, 0 ],
                           [ 0, 1, 1, 1, 0, 0, 3, 3, 3, 0 ],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]])
        
        image = np.random.uniform(size=labels.shape)
        for center_labels in (labels, None):
            m = self.run_module(image, labels, center_labels, 4)
            for bin in range(1,5):
                data = m.get_current_measurement(OBJECT_NAME, 
                                                 feature_frac_at_d(bin, 4))
                self.assertEqual(len(data), 3)
                self.assertTrue(np.isnan(data[1]))
                
    def test_04_02_center_outside_of_object(self):
        '''Make sure MeasureObjectRadialDistribution can handle oddly shaped objects'''
        np.random.seed(42)
        labels = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                           [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
                           [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                           [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                           [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                           [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                           [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]])
        
        center_labels = np.zeros(labels.shape, int)
        center_labels[int(center_labels.shape[0]/2),
                      int(center_labels.shape[1]/2)] = 1
        
        image = np.random.uniform(size=labels.shape)
        m = self.run_module(image, labels, center_labels, 4)
        for bin in range(1,5):
            data = m.get_current_measurement(OBJECT_NAME, 
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 1)

        m = self.run_module(image, labels, None, 4)
        for bin in range(1,5):
            data = m.get_current_measurement(OBJECT_NAME, 
                                             feature_frac_at_d(bin, 4))
            self.assertEqual(len(data), 1)
    
    def test_04_03_wrong_size(self):
        '''Regression test for IMG-961: objects & image of different sizes
        
        Make sure that the module executes without exception with and
        without centers and with similarly and differently shaped centers
        '''
        np.random.seed(43)
        labels = np.ones((30,40), int)
        image = np.random.uniform(size=(20,50))
        m = self.run_module(image, labels)
        centers = np.zeros(labels.shape)
        centers[15,20] = 1
        m = self.run_module(image, labels, centers)
        centers = np.zeros((35,35), int)
        centers[15,20] = 1
        m = self.run_module(image, labels, centers)
